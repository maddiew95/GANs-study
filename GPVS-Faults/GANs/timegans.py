"""
1D TimeGAN for GPVS fault-detection augmentation.

PyTorch port of jsyoon0823's TimeGAN (NeurIPS 2019), adapted to this domain:
  * Five GRU networks — embedder, recovery, generator, supervisor, discriminator —
    trained in three phases: (1) autoencoder/embedding, (2) supervised next-step,
    (3) joint adversarial. Same scheme as the reference.
  * Per-class, like the DCGAN/WGAN modules: one TimeGAN per fault class, exposing
    the SAME public API (train_class_gans / augment) so it drops into the registry.
  * Sequences in, ROWS out. TimeGAN works on length-`seq_len` windows, but your
    classifiers consume single 13-feature rows, so generated windows are UNROLLED
    (each timestep -> one row, tagged with the class) before augmentation.

================================  READ THIS  ================================
TimeGAN only means anything if the windows are TEMPORALLY CONTIGUOUS. If you pass
it the randomly-sampled scene `train.csv` (rows drawn out of time order), the
"sequences" are noise and TimeGAN degenerates to a feature autoencoder.
  * For real temporal modeling: feed per-class feature rows that are CONTIGUOUS in
    time (sampled as blocks from the original F{i}M recordings), and use seq_len>1.
  * seq_len=1 is the degenerate fallback (no temporal structure modeled).
This module windows whatever row order it is given; it cannot restore time order
that was lost upstream. Decide the data path before trusting the output.
============================================================================
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


class _GRUNet(nn.Module):
    """GRU stack + linear head. activation: nn.Sigmoid() for the latent/feature
    nets (TimeGAN scales data to [0,1]); nn.Identity() for the discriminator."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = activation

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.act(self.fc(out))


def _make_windows(X_rows, seq_len):
    """(n_rows, n_features) -> (n_windows, seq_len, n_features) via stride-1 windows.
    Assumes rows are time-ordered; see module header."""
    n = len(X_rows)
    if n < seq_len:
        seq_len = max(2, n)
    idx = np.arange(seq_len)[None, :] + np.arange(n - seq_len + 1)[:, None]
    return X_rows[idx], seq_len


def train_timegan(X_rows, seed, device, seq_len=24, hidden_dim=24, num_layers=3,
                  n_epochs_emb=300, n_epochs_sup=300, n_epochs_joint=500,
                  batch_size=64, lr=1e-3, gamma=1.0, verbose=False):
    """Train one TimeGAN on a single class's REAL feature rows.
    Returns (handle, history). handle holds the nets + scaler + seq_len."""
    set_seed(seed)
    n_features = X_rows.shape[1]

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_rows)   # sigmoid outputs -> [0,1]
    Xs = scaler.transform(X_rows).astype("float32")

    windows, seq_len = _make_windows(Xs, seq_len)
    if len(windows) < 2:
        raise ValueError(f"Too few rows ({len(X_rows)}) to form windows at seq_len.")

    z_dim = n_features
    bs = min(batch_size, len(windows))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(windows)),
                        batch_size=bs, shuffle=True, drop_last=False, generator=g)

    sig, idn = nn.Sigmoid(), nn.Identity()
    embedder   = _GRUNet(n_features, hidden_dim, hidden_dim, num_layers, sig).to(device)
    recovery   = _GRUNet(hidden_dim, hidden_dim, n_features, num_layers, sig).to(device)
    generator  = _GRUNet(z_dim,      hidden_dim, hidden_dim, num_layers, sig).to(device)
    supervisor = _GRUNet(hidden_dim, hidden_dim, hidden_dim, max(1, num_layers - 1), sig).to(device)
    discrim    = _GRUNet(hidden_dim, hidden_dim, 1,          num_layers, idn).to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    opt_e = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr)
    opt_g = torch.optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=lr)
    opt_d = torch.optim.Adam(discrim.parameters(), lr=lr)
    opt_er = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr)

    history = {"e_loss": [], "s_loss": [], "g_loss": [], "d_loss": []}

    def rand_z(n, L):
        return torch.rand(n, L, z_dim, device=device)

    # ---- Phase 1: embedding network (reconstruction) ----
    for ep in range(n_epochs_emb):
        tot = nb = 0.0
        for (x,) in loader:
            x = x.to(device)
            opt_e.zero_grad()
            h = embedder(x)
            x_tilde = recovery(h)
            e_loss = mse(x_tilde, x)
            (10 * e_loss).backward()
            opt_e.step()
            tot += e_loss.item(); nb += 1
        if ep == n_epochs_emb - 1:
            history["e_loss"].append(tot / nb)
        if verbose and (ep == 0 or (ep + 1) % 100 == 0):
            print(f"    [emb {ep+1}/{n_epochs_emb}] recon {tot/nb:.4f}")

    # ---- Phase 2: supervised loss (next-step in latent) ----
    for ep in range(n_epochs_sup):
        tot = nb = 0.0
        for (x,) in loader:
            x = x.to(device)
            opt_g.zero_grad()
            h = embedder(x).detach()
            h_sup = supervisor(h)
            s_loss = mse(h_sup[:, :-1, :], h[:, 1:, :])
            s_loss.backward()
            opt_g.step()
            tot += s_loss.item(); nb += 1
        if ep == n_epochs_sup - 1:
            history["s_loss"].append(tot / nb)
        if verbose and (ep == 0 or (ep + 1) % 100 == 0):
            print(f"    [sup {ep+1}/{n_epochs_sup}] s_loss {tot/nb:.4f}")

    # ---- Phase 3: joint adversarial ----
    for ep in range(n_epochs_joint):
        tg = td = nb = 0.0
        for (x,) in loader:
            x = x.to(device)
            n, L = x.size(0), x.size(1)

            # --- generator + embedder (twice, as in reference) ---
            for _ in range(2):
                opt_g.zero_grad()
                h = embedder(x)
                e_hat = generator(rand_z(n, L))
                h_hat = supervisor(e_hat)
                h_sup = supervisor(h)
                x_hat = recovery(h_hat)

                y_fake = discrim(h_hat)
                y_fake_e = discrim(e_hat)
                g_u = bce(y_fake, torch.ones_like(y_fake))
                g_u_e = bce(y_fake_e, torch.ones_like(y_fake_e))
                g_s = mse(h_sup[:, :-1, :], h[:, 1:, :].detach())
                # moment matching (mean + std) between real and recovered-synthetic
                g_v = (torch.mean(torch.abs(x_hat.mean(0) - x.mean(0)))
                       + torch.mean(torch.abs(x_hat.std(0) + 1e-6 - (x.std(0) + 1e-6))))
                g_loss = g_u + gamma * g_u_e + 100 * torch.sqrt(g_s + 1e-8) + 100 * g_v
                g_loss.backward()
                opt_g.step()

                # embedder/recovery refinement with supervised term
                opt_er.zero_grad()
                h = embedder(x)
                x_tilde = recovery(h)
                h_sup = supervisor(h)
                e_loss0 = mse(x_tilde, x)
                s_loss = mse(h_sup[:, :-1, :], h[:, 1:, :].detach())
                e_loss = 10 * e_loss0 + 0.1 * s_loss
                e_loss.backward()
                opt_er.step()

            # --- discriminator ---
            opt_d.zero_grad()
            h = embedder(x).detach()
            e_hat = generator(rand_z(n, L)).detach()
            h_hat = supervisor(e_hat).detach()
            y_real = discrim(h)
            y_fake = discrim(h_hat)
            y_fake_e = discrim(e_hat)
            d_loss = (bce(y_real, torch.ones_like(y_real))
                      + bce(y_fake, torch.zeros_like(y_fake))
                      + gamma * bce(y_fake_e, torch.zeros_like(y_fake_e)))
            d_loss.backward()
            opt_d.step()

            tg += g_loss.item(); td += d_loss.item(); nb += 1

        history["g_loss"].append(tg / nb)
        history["d_loss"].append(td / nb)
        if verbose and (ep == 0 or (ep + 1) % 100 == 0):
            print(f"    [joint {ep+1}/{n_epochs_joint}] G {tg/nb:.3f} D {td/nb:.3f}")

    handle = {"generator": generator, "supervisor": supervisor, "recovery": recovery,
              "scaler": scaler, "seq_len": seq_len, "z_dim": z_dim}
    return handle, history


def generate(handle, n_rows, seed, device):
    """Generate n_rows synthetic rows (windows unrolled into individual rows)."""
    if n_rows <= 0:
        return np.empty((0, handle["scaler"].n_features_in_), dtype="float32")
    G, S, R = handle["generator"], handle["supervisor"], handle["recovery"]
    seq_len, z_dim = handle["seq_len"], handle["z_dim"]
    n_windows = int(np.ceil(n_rows / seq_len))
    G.eval(); S.eval(); R.eval()
    gen = torch.Generator().manual_seed(seed + 10_000)
    z = torch.rand(n_windows, seq_len, z_dim, generator=gen).to(device)
    with torch.no_grad():
        x_hat = R(S(G(z))).cpu().numpy()           # (n_windows, seq_len, n_features)
    rows = x_hat.reshape(-1, x_hat.shape[-1])[:n_rows]   # unroll windows -> rows
    return handle["scaler"].inverse_transform(rows).astype("float32")


# ---- public API mirrors dcgans.py so the registry treats it the same ----
def train_class_gans(train_array, seed, device, n_classes=8,
                     feat_slice=(1, 14), label_col=-1, **gan_kw):
    feats = train_array[:, feat_slice[0]:feat_slice[1]].astype("float32")
    labels = train_array[:, label_col].astype(int)
    gans, histories = {}, {}
    for c in range(n_classes):
        Xc = feats[labels == c]
        if len(Xc) < 2:
            continue
        handle, hist = train_timegan(Xc, seed=seed, device=device, **gan_kw)
        gans[c] = (handle, len(Xc))
        histories[c] = hist
    return gans, histories


def augment(train_array, gans, ratio, seed, device,
            feat_slice=(1, 14), label_col=-1):
    if ratio <= 0 or not gans:
        return train_array
    width = train_array.shape[1]
    blocks = [train_array]
    for c, (handle, n_real) in gans.items():
        n_gen = int(round(ratio * n_real))
        gen = generate(handle, n_gen, seed=seed, device=device)
        if len(gen) == 0:
            continue
        block = np.zeros((len(gen), width), dtype=train_array.dtype)
        block[:, feat_slice[0]:feat_slice[1]] = gen
        block[:, label_col] = c
        blocks.append(block)
    return np.vstack(blocks)
