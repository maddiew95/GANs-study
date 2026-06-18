"""
1D Conditional GAN (cGAN) for GPVS fault-detection augmentation — Interpretation B.

Adapted from eriklindernoren's cgan.py. Differences from the DCGAN module:
  * Single conditional model (label embedding) generates all 8 classes, instead
    of 8 separate per-class generators. Shared layers train on the POOLED
    multi-class training set, so the model is less starved at severe scarcity.
  * MLP architecture (no convolutions) — appropriate for a 13-feature vector.
  * Least-squares (MSE) adversarial loss, per the reference cGAN. Loss magnitudes
    are therefore NOT comparable to the BCE-based DCGAN, but d_acc still is.
  * One GLOBAL scaler (fit on all training features) since one model emits every
    class; class structure is carried by the embedding, not by separate scalers.

Interface mirrors dcgan_1d so it drops into the same notebook loop:
  bundle, history = train_cgan(train_array, seed, device)
  aug_array       = augment_cgan(train_array, bundle, ratio, seed, device)
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Architecture (MLP, conditional via label embedding)
# ----------------------------------------------------------------------
class ConditionalGenerator1D(nn.Module):
    def __init__(self, latent_dim=64, n_classes=8, n_features=13, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or n_classes
        self.label_emb = nn.Embedding(n_classes, embed_dim)

        def block(in_f, out_f, normalize=True):
            layers = [nn.Linear(in_f, out_f)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_f, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + embed_dim, 128, normalize=False),
            *block(128, 256),
            nn.Linear(256, n_features),
            nn.Tanh(),                         # output in [-1, 1] -> matches scaler
        )

    def forward(self, z, labels):
        x = torch.cat((self.label_emb(labels), z), dim=-1)
        return self.model(x)                   # (B, n_features)


class ConditionalDiscriminator1D(nn.Module):
    def __init__(self, n_classes=8, n_features=13, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or n_classes
        self.label_emb = nn.Embedding(n_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(n_features + embed_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),                 # no sigmoid: LSGAN uses raw scores
        )

    def forward(self, x, labels):
        d_in = torch.cat((x, self.label_emb(labels)), dim=-1)
        return self.model(d_in)


# ----------------------------------------------------------------------
# Train (one model, all classes pooled)
# ----------------------------------------------------------------------
def train_cgan(train_array, seed, device, n_classes=8, feat_slice=(1, 14),
               label_col=-1, n_epochs=300, batch_size=64, latent_dim=64,
               lr=2e-4, b1=0.5, b2=0.999, verbose=False):
    set_seed(seed)
    feats = train_array[:, feat_slice[0]:feat_slice[1]].astype("float32")
    labels = train_array[:, label_col].astype(int)
    counts = {c: int((labels == c).sum()) for c in range(n_classes)}

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(feats)   # global
    Xs = scaler.transform(feats).astype("float32")
    N, n_features = Xs.shape

    bs = min(batch_size, N)
    drop_last = (N % bs == 1)                  # BatchNorm fails on a size-1 batch
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xs), torch.from_numpy(labels)),
        batch_size=bs, shuffle=True, drop_last=drop_last, generator=g)

    G = ConditionalGenerator1D(latent_dim, n_classes, n_features).to(device)
    D = ConditionalDiscriminator1D(n_classes, n_features).to(device)

    mse = nn.MSELoss()                         # least-squares GAN
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(b1, b2))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

    # d_acc uses a 0.5 midpoint threshold (targets are 1.0 / 0.0) so it stays
    # comparable to the DCGAN's d_acc even though the loss differs.
    history = {"d_loss": [], "g_loss": [], "d_acc": [], "d_real": [], "d_fake": []}

    for epoch in range(n_epochs):
        ep_d = ep_g = ep_real = ep_fake = 0.0
        ep_correct = ep_total = 0
        n_batches = 0
        for real, lab in loader:
            real, lab = real.to(device), lab.to(device)
            n = real.size(0)
            valid = torch.ones(n, 1, device=device)
            fake = torch.zeros(n, 1, device=device)

            # Generator: random target labels, as in the reference cGAN
            opt_G.zero_grad()
            z = torch.randn(n, latent_dim, device=device)
            gen_labels = torch.randint(0, n_classes, (n,), device=device)
            gen = G(z, gen_labels)
            g_loss = mse(D(gen, gen_labels), valid)
            g_loss.backward()
            opt_G.step()

            # Discriminator
            opt_D.zero_grad()
            d_real = D(real, lab)
            d_fake = D(gen.detach(), gen_labels)
            d_loss = (mse(d_real, valid) + mse(d_fake, fake)) / 2
            d_loss.backward()
            opt_D.step()

            with torch.no_grad():
                ep_correct += (d_real > 0.5).sum().item() + (d_fake < 0.5).sum().item()
                ep_total += 2 * n
                ep_real += d_real.mean().item()
                ep_fake += d_fake.mean().item()
            ep_d += d_loss.item(); ep_g += g_loss.item(); n_batches += 1

        history["d_loss"].append(ep_d / n_batches)
        history["g_loss"].append(ep_g / n_batches)
        history["d_acc"].append(ep_correct / ep_total)
        history["d_real"].append(ep_real / n_batches)
        history["d_fake"].append(ep_fake / n_batches)

        if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
            print(f"    [ep {epoch+1:3d}/{n_epochs}] "
                  f"D_loss {history['d_loss'][-1]:.3f} "
                  f"G_loss {history['g_loss'][-1]:.3f} "
                  f"D_acc {history['d_acc'][-1]:.3f}")

    bundle = {"G": G, "scaler": scaler, "counts": counts,
              "latent_dim": latent_dim, "n_classes": n_classes}
    return bundle, history


# ----------------------------------------------------------------------
# Generate / augment (condition on label)
# ----------------------------------------------------------------------
def generate_cgan(bundle, n_samples, label, seed, device):
    if n_samples <= 0:
        return np.empty((0, bundle["scaler"].n_features_in_), dtype="float32")
    G, scaler = bundle["G"], bundle["scaler"]
    latent_dim = bundle["latent_dim"]
    G.eval()
    # distinct, reproducible noise stream per (seed, label)
    g = torch.Generator().manual_seed(seed + 10_000 + int(label))
    z = torch.randn(n_samples, latent_dim, generator=g).to(device)
    labels = torch.full((n_samples,), int(label), dtype=torch.long, device=device)
    with torch.no_grad():
        xs = G(z, labels).cpu().numpy()
    return scaler.inverse_transform(xs).astype("float32")


def augment_cgan(train_array, bundle, ratio, seed, device,
                 feat_slice=(1, 14), label_col=-1):
    """Return real + synthetic training array. ratio is synthetic-per-real,
    applied per class. ratio=0 returns the real array unchanged."""
    if ratio <= 0:
        return train_array
    counts = bundle["counts"]
    width = train_array.shape[1]
    blocks = [train_array]
    for c in range(bundle["n_classes"]):
        n_gen = int(round(ratio * counts.get(c, 0)))
        gen = generate_cgan(bundle, n_gen, c, seed=seed, device=device)
        if len(gen) == 0:
            continue
        block = np.zeros((len(gen), width), dtype=train_array.dtype)
        block[:, feat_slice[0]:feat_slice[1]] = gen
        block[:, label_col] = c
        blocks.append(block)
    return np.vstack(blocks)


def plot_cgan_history(history, title=""):
    """Loss + discriminator diagnostics for the (single) conditional model."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["d_loss"], label="D loss")
    ax[0].plot(history["g_loss"], label="G loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE loss")
    ax[0].set_title(f"{title} — loss"); ax[0].legend()
    ax[1].plot(history["d_acc"], label="D accuracy")
    ax[1].axhline(0.5, ls="--", c="grey", lw=1, label="chance (0.5)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title(f"{title} — D accuracy"); ax[1].legend()
    fig.tight_layout()
    return fig