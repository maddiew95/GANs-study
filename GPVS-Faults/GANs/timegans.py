"""
1D TimeGAN for GPVS fault-detection augmentation.

PyTorch redesign of jsyoon0823/TimeGAN (Yoon et al., NeurIPS 2019) for this domain:
  * Five GRU networks — embedder, recovery, generator, supervisor, discriminator —
    trained in the reference's three phases:
      1) embedding (autoencoder reconstruction),
      2) supervised next-step prediction in latent space,
      3) joint adversarial training (G+S updated twice per D update, with the
         supervised and moment-matching terms, weights 100/100/gamma as in the paper).
  * PER-CLASS, like dcgans/wgans: one TimeGAN per fault class. Public API mirrors
    those modules (train_class_gans / augment) so it drops into the same notebooks.
  * SEQUENCES IN, ROWS OUT: trains on stride-1 windows of length seq_len cut from
    the class's (time-ordered!) rows; generated windows are UNROLLED back into
    individual rows so the row-based classifiers consume them unchanged.

DATA REQUIREMENT (critical)
---------------------------
Feed this module the CONTIGUOUS scene files from TimeGANs_csv/ (subsample_timegan),
NOT the random-sampled CSV_Files scenes. Random rows destroy within-window
chronology and reduce TimeGAN to an expensive autoencoder.

Scarce-scene behaviour: if a class has fewer rows than seq_len, the window length
shrinks to that class's row count (recorded in history["seq_len_used"]); with
seq_len effectively 1–10 the temporal modelling is degenerate — report those scenes
honestly rather than as genuine TimeGAN results.
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
    """GRU stack + linear head. Sigmoid for latent/feature nets (data scaled to
    [0,1] as in the reference); Identity for the discriminator (BCE-with-logits)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid=True):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.act(self.fc(out))


def _make_windows(X_rows, seq_len):
    """(n_rows, F) -> (n_windows, L, F) stride-1 windows. Assumes rows are
    time-ordered. Shrinks L to n_rows when the class is too small (degenerate)."""
    n = len(X_rows)
    L = min(seq_len, n)
    idx = np.arange(L)[None, :] + np.arange(n - L + 1)[:, None]
    return X_rows[idx], L


def train_timegan(X_rows, seed, device, seq_len=24, hidden_dim=24, num_layers=3,
                  n_epochs_emb=300, n_epochs_sup=300, n_epochs_joint=400,
                  batch_size=64, lr=1e-3, gamma=1.0, verbose=False):
    """Train one TimeGAN on ONE class's time-ordered feature rows.
    Returns (handle, history)."""
    set_seed(seed)
    n_features = X_rows.shape[1]

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X_rows)
    Xs = scaler.transform(X_rows).astype("float32")
    windows, L = _make_windows(Xs, seq_len)

    z_dim = n_features
    bs = min(batch_size, len(windows))
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(windows)),
                        batch_size=bs, shuffle=True, drop_last=False, generator=g)

    embedder   = _GRUNet(n_features, hidden_dim, hidden_dim, num_layers).to(device)
    recovery   = _GRUNet(hidden_dim, hidden_dim, n_features, num_layers).to(device)
    generator  = _GRUNet(z_dim,      hidden_dim, hidden_dim, num_layers).to(device)
    supervisor = _GRUNet(hidden_dim, hidden_dim, hidden_dim, max(1, num_layers - 1)).to(device)
    discrim    = _GRUNet(hidden_dim, hidden_dim, 1, num_layers, sigmoid=False).to(device)

    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    opt_e  = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr)
    opt_g  = torch.optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=lr)
    opt_d  = torch.optim.Adam(discrim.parameters(), lr=lr)
    opt_er = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=lr)

    history = {"e_loss": [], "s_loss": [], "g_loss": [], "d_loss": [],
               "seq_len_used": L, "n_windows": len(windows)}

    def rand_z(n, T):
        return torch.rand(n, T, z_dim, device=device)

    # ---- Phase 1: embedding (reconstruction) ----
    for ep in range(n_epochs_emb):
        tot = nb = 0.0
        for (x,) in loader:
            x = x.to(device)
            opt_e.zero_grad()
            e_loss = mse(recovery(embedder(x)), x)
            (10 * torch.sqrt(e_loss + 1e-8)).backward()
            opt_e.step()
            tot += e_loss.item(); nb += 1
        history["e_loss"].append(tot / nb)
        if verbose and (ep == 0 or (ep + 1) % 100 == 0):
            print(f"    [emb {ep+1}/{n_epochs_emb}] recon {tot/nb:.5f}")

    # ---- Phase 2: supervised next-step ----
    for ep in range(n_epochs_sup):
        tot = nb = 0.0
        for (x,) in loader:
            x = x.to(device)
            opt_g.zero_grad()
            h = embedder(x).detach()
            s_loss = mse(supervisor(h)[:, :-1, :], h[:, 1:, :])
            s_loss.backward()
            opt_g.step()
            tot += s_loss.item(); nb += 1
        history["s_loss"].append(tot / nb)
        if verbose and (ep == 0 or (ep + 1) % 100 == 0):
            print(f"    [sup {ep+1}/{n_epochs_sup}] s_loss {tot/nb:.5f}")

    # ---- Phase 3: joint adversarial ----
    for ep in range(n_epochs_joint):
        tg = td = nb = 0.0
        for (x,) in loader:
            x = x.to(device)
            n, T = x.size(0), x.size(1)

            for _ in range(2):                       # G (+ E/R refinement) twice per D
                opt_g.zero_grad()
                h = embedder(x)
                e_hat = generator(rand_z(n, T))
                h_hat = supervisor(e_hat)
                x_hat = recovery(h_hat)
                g_u   = bce(discrim(h_hat), torch.ones(n, T, 1, device=device))
                g_u_e = bce(discrim(e_hat), torch.ones(n, T, 1, device=device))
                g_s   = mse(supervisor(h)[:, :-1, :], h[:, 1:, :].detach())
                g_v   = (torch.mean(torch.abs(x_hat.mean(dim=(0, 1)) - x.mean(dim=(0, 1)))) +
                         torch.mean(torch.abs(x_hat.std(dim=(0, 1)) - x.std(dim=(0, 1)))))
                (g_u + gamma * g_u_e + 100 * torch.sqrt(g_s + 1e-8) + 100 * g_v).backward()
                opt_g.step()

                opt_er.zero_grad()
                h = embedder(x)
                e0 = mse(recovery(h), x)
                s  = mse(supervisor(h)[:, :-1, :], h[:, 1:, :].detach())
                (10 * torch.sqrt(e0 + 1e-8) + 0.1 * s).backward()
                opt_er.step()
                tg += g_u.item()

            opt_d.zero_grad()
            h = embedder(x).detach()
            e_hat = generator(rand_z(n, T)).detach()
            h_hat = supervisor(e_hat).detach()
            d_loss = (bce(discrim(h),     torch.ones(n, T, 1, device=device)) +
                      bce(discrim(h_hat), torch.zeros(n, T, 1, device=device)) +
                      gamma * bce(discrim(e_hat), torch.zeros(n, T, 1, device=device)))
            d_loss.backward()
            opt_d.step()
            td += d_loss.item(); nb += 1

        history["g_loss"].append(tg / (2 * nb))
        history["d_loss"].append(td / nb)
        if verbose and (ep == 0 or (ep + 1) % 100 == 0):
            print(f"    [joint {ep+1}/{n_epochs_joint}] G_adv {history['g_loss'][-1]:.4f} "
                  f"D {history['d_loss'][-1]:.4f}")

    handle = {"generator": generator, "supervisor": supervisor, "recovery": recovery,
              "scaler": scaler, "seq_len": L, "z_dim": z_dim}
    return handle, history


def generate(handle, n_rows, seed, device):
    """Generate n_rows synthetic rows (windows unrolled)."""
    if n_rows <= 0:
        return np.empty((0, handle["scaler"].n_features_in_), dtype="float32")
    G, S, R = handle["generator"], handle["supervisor"], handle["recovery"]
    L, z_dim = handle["seq_len"], handle["z_dim"]
    n_windows = int(np.ceil(n_rows / L))
    G.eval(); S.eval(); R.eval()
    gen = torch.Generator().manual_seed(seed + 10_000)
    z = torch.rand(n_windows, L, z_dim, generator=gen).to(device)
    with torch.no_grad():
        x_hat = R(S(G(z))).cpu().numpy()
    rows = x_hat.reshape(-1, x_hat.shape[-1])[:n_rows]
    return handle["scaler"].inverse_transform(rows).astype("float32")


# ---------- public API mirrors dcgans/wgans ----------
def train_class_gans(train_array, seed, device, n_classes=8,
                     feat_slice=(1, 14), label_col=-1, **gan_kw):
    """One TimeGAN per class. train_array rows MUST be time-ordered within each
    class (use TimeGANs_csv). Returns (gans, histories) shaped like dcgans."""
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
    """Real + synthetic rows; ratio = synthetic-per-real per class; 0 -> unchanged.
    Timestamp column of synthetic rows is 0 (never read by the classifiers)."""
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


def plot_timegan_history(history, title=""):
    """Reconstruction, supervised, and joint-phase curves for one class."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(history["e_loss"]); ax[0].set_title(f"{title} — recon (phase 1)")
    ax[1].plot(history["s_loss"], color="tab:orange"); ax[1].set_title("supervised (phase 2)")
    ax[2].plot(history["g_loss"], label="G adv"); ax[2].plot(history["d_loss"], label="D")
    ax[2].set_title("joint (phase 3)"); ax[2].legend()
    for a in ax: a.set_xlabel("epoch"); a.grid(alpha=0.2)
    fig.tight_layout()
    return fig
