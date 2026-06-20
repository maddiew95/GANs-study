"""
1D DCGAN for GPVS fault-detection augmentation (Interpretation B).

Design notes
------------
* Per-class, unconditional: one DCGAN is trained per class (F0..F7). Generating
  all 8 classes with an unconditional GAN means 8 separate generators; the
  architecture is identical across classes so it is not a confound.
* Length-preserving convolutions: 13 features map poorly onto MNIST-style spatial
  up/downsampling, so the latent is projected straight to length-13 and every
  generator conv keeps stride 1, pad 1. Depth comes from channels, not resizing.
* Leakage firewall: the GAN and its scaler see ONLY the real per-class training
  features. Synthetic rows are added to the training set after generation; val
  and test are never touched.
* Reproducible + stability-measuring: every random draw flows from the run seed,
  so re-running gives identical output, while the five recorded seeds give the
  Monte Carlo spread.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int):
    """Pin every RNG so a run is byte-reproducible (Interpretation B)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ----------------------------------------------------------------------
# Architecture (MLP — the right inductive bias for 13 UNORDERED features)
# ----------------------------------------------------------------------
# Convolutions assume adjacent indices are correlated; GPVS features are not
# ordered, so Conv1d models structure that isn't there. These are plain MLPs.
class Generator1D(nn.Module):
    def __init__(self, latent_dim=64, n_features=13, hidden=128):
        super().__init__()

        def block(in_f, out_f, normalize=True):
            layers = [nn.Linear(in_f, out_f)]
            if normalize:                       # BatchNorm is fine in the GENERATOR
                layers.append(nn.BatchNorm1d(out_f, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, hidden, normalize=False),
            *block(hidden, hidden * 2),
            *block(hidden * 2, hidden * 2),
            nn.Linear(hidden * 2, n_features),
            nn.Tanh(),                          # output in [-1, 1] -> matches scaler
        )

    def forward(self, z):
        return self.model(z)                    # (B, n_features)


class Discriminator1D(nn.Module):
    def __init__(self, n_features=13, hidden=128):
        super().__init__()
        # CRITICAL: no BatchNorm in D. For tabular data the real/fake signal lives
        # in per-feature means and scales; BatchNorm (with separate real/fake
        # passes) normalizes exactly that away and collapses D to a constant 0.5.
        self.model = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),                       # kept for BCE compatibility
        )

    def forward(self, x):
        return self.model(x)                    # x is (B, n_features)


# ----------------------------------------------------------------------
# Train / generate (per class)
# ----------------------------------------------------------------------
def train_dcgan(X_raw, seed, device, n_epochs=300, batch_size=64,
                latent_dim=64, lr=2e-4, b1=0.5, b2=0.999, verbose=False):
    """Train one DCGAN on a single class's REAL training features.

    X_raw: (N, n_features) raw feature array for ONE class (real data only).
    Returns (generator, scaler) ready for generation.
    """
    set_seed(seed)
    N, n_features = X_raw.shape
    if N < 2:
        raise ValueError(f"Need >=2 real samples to train a GAN, got {N}.")

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_raw)
    Xs = scaler.transform(X_raw).astype("float32")

    bs = min(batch_size, N)
    # BatchNorm fails on a final batch of size 1 -> drop it when that would happen
    drop_last = (N % bs == 1)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xs)),
                        batch_size=bs, shuffle=True,
                        drop_last=drop_last, generator=g)

    G = Generator1D(latent_dim, n_features).to(device)
    D = Discriminator1D(n_features).to(device)
    G.apply(_weights_init)
    D.apply(_weights_init)

    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

    # Per-epoch training diagnostics. IMPORTANT: d_acc ~0.5 is ambiguous —
    #   healthy: d_loss BELOW ln(2)=0.693 and d_real > d_fake (D learning, G good)
    #   DEAD:    d_loss PINNED at 0.693 with d_real == d_fake == 0.5 (D collapsed
    #            to a constant, G gets no gradient). Read d_loss + the gap, not d_acc alone.
    history = {"d_loss": [], "g_loss": [], "d_acc": [],
               "d_real": [], "d_fake": []}

    for epoch in range(n_epochs):
        ep_d = ep_g = ep_real = ep_fake = 0.0
        ep_correct = ep_total = 0
        n_batches = 0
        for (real,) in loader:
            real = real.to(device)
            n = real.size(0)
            valid = torch.ones(n, 1, device=device)
            fake = torch.zeros(n, 1, device=device)

            # Generator
            opt_G.zero_grad()
            z = torch.randn(n, latent_dim, device=device)
            gen = G(z)
            g_loss = bce(D(gen), valid)
            g_loss.backward()
            opt_G.step()

            # Discriminator
            opt_D.zero_grad()
            d_real = D(real)
            d_fake = D(gen.detach())
            real_loss = bce(d_real, valid)
            fake_loss = bce(d_fake, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            # ---- diagnostics ----
            ep_d += d_loss.item()
            ep_g += g_loss.item()
            ep_real += d_real.mean().item()
            ep_fake += d_fake.mean().item()
            # D "accuracy": real scored >0.5 and fake scored <0.5
            ep_correct += (d_real > 0.5).sum().item() + (d_fake < 0.5).sum().item()
            ep_total += 2 * n
            n_batches += 1

        history["d_loss"].append(ep_d / n_batches)
        history["g_loss"].append(ep_g / n_batches)
        history["d_real"].append(ep_real / n_batches)
        history["d_fake"].append(ep_fake / n_batches)
        history["d_acc"].append(ep_correct / ep_total)

        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(f"    [ep {epoch+1:3d}/{n_epochs}] "
                  f"D_loss {history['d_loss'][-1]:.3f} "
                  f"G_loss {history['g_loss'][-1]:.3f} "
                  f"D_acc {history['d_acc'][-1]:.3f} "
                  f"D(real) {history['d_real'][-1]:.2f} "
                  f"D(fake) {history['d_fake'][-1]:.2f}")

    return G, scaler, history


def generate(G, scaler, n_samples, seed, device, latent_dim=64):
    """Generate n_samples synthetic rows in RAW feature space.

    Uses a local CPU generator for the latent draw so the global RNG state is
    untouched; still fully reproducible from `seed`.
    """
    if n_samples <= 0:
        return np.empty((0, scaler.n_features_in_), dtype="float32")
    G.eval()
    g = torch.Generator().manual_seed(seed + 10_000)  # distinct, reproducible stream
    z = torch.randn(n_samples, latent_dim, generator=g).to(device)
    with torch.no_grad():
        xs = G(z).cpu().numpy()
    return scaler.inverse_transform(xs).astype("float32")


# ----------------------------------------------------------------------
# Augmentation glue (train GANs once per scene, reuse across ratios)
# ----------------------------------------------------------------------
def train_class_gans(train_array, seed, device, n_classes=8,
                     feat_slice=(1, 14), label_col=-1, **gan_kw):
    """Train one DCGAN per class on the REAL training set.

    Returns (gans, histories):
      gans      = {class_idx: (generator, scaler, n_real)}
      histories = {class_idx: history dict from train_dcgan}
    Train this ONCE per (seed, scene); the same generators serve every ratio.
    The histories let you plot G/D loss and D accuracy per class, which is the
    direct evidence of synthetic fidelity (or its collapse) at each scene."""
    feats = train_array[:, feat_slice[0]:feat_slice[1]].astype("float32")
    labels = train_array[:, label_col].astype(int)
    gans, histories = {}, {}
    for c in range(n_classes):
        Xc = feats[labels == c]
        if len(Xc) < 2:
            continue
        G, scaler, hist = train_dcgan(Xc, seed=seed, device=device, **gan_kw)
        gans[c] = (G, scaler, len(Xc))
        histories[c] = hist
    return gans, histories


def plot_gan_history(history, title=""):
    """Plot G/D loss and discriminator diagnostics for one class.
    Returns the matplotlib figure so you can save it for the paper."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["d_loss"], label="D loss")
    ax[0].plot(history["g_loss"], label="G loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_title(f"{title} — loss"); ax[0].legend()

    ax[1].plot(history["d_acc"], label="D accuracy")
    ax[1].plot(history["d_real"], label="D(real)")
    ax[1].plot(history["d_fake"], label="D(fake)")
    ax[1].axhline(0.5, ls="--", c="gray", lw=0.8)
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xlabel("epoch")
    ax[1].set_title(f"{title} — discriminator"); ax[1].legend()
    fig.tight_layout()
    return fig


def augment(train_array, gans, ratio, seed, device,
            feat_slice=(1, 14), label_col=-1, latent_dim=64):
    """Return real + synthetic training array. ratio is synthetic-per-real
    (0.5 -> +50%, 2.0 -> +200%). ratio=0 returns the real array unchanged."""
    if ratio <= 0 or not gans:
        return train_array

    width = train_array.shape[1]
    blocks = [train_array]
    for c, (G, scaler, n_real) in gans.items():
        n_gen = int(round(ratio * n_real))
        gen = generate(G, scaler, n_gen, seed=seed, device=device,
                       latent_dim=latent_dim)
        if len(gen) == 0:
            continue
        block = np.zeros((len(gen), width), dtype=train_array.dtype)
        block[:, feat_slice[0]:feat_slice[1]] = gen
        block[:, label_col] = c
        blocks.append(block)
    return np.vstack(blocks)