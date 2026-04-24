"""
F0-only Autoencoder for unsupervised anomaly scoring on GPVS-Faults.
Adapted from O'Quinn-style AE anomaly detection, scaled to 13-D tabular input.

Training: on F0 (healthy) samples only.
Inference: reconstruction MSE -> anomaly score. High score = likely fault.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAE(nn.Module):
    """
    Symmetric MLP autoencoder for 13-D GPVS feature vectors.
    Dimension ladder:  13 -> 32 -> 16 -> 8 -> 16 -> 32 -> 13
    """
    def __init__(self, in_dim: int = 13, hidden_dims=(32, 16), latent_dim: int = 8):
        super().__init__()

        # ---- Encoder ----
        enc_layers = []
        prev = in_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim)]   # latent is linear, no activation
        self.encoder = nn.Sequential(*enc_layers)

        # ---- Decoder (mirror) ----
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, in_dim)]       # linear output (z-scored input range)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):                     # x: (B, 1, 13) or (B, 13)
        if x.dim() == 3:
            x = x.squeeze(1)                  # -> (B, 13)
        z = self.encoder(x)                   # (B, latent_dim)
        x_hat = self.decoder(z)               # (B, 13)
        return x_hat, z

    def encode(self, x):
        """Expose latent features -- useful if later used as feature extractor."""
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.encoder(x)


class AnomalyDetector:
    """
    Wrapper that trains AnomalyAE on F0-only data and scores new samples
    by reconstruction error (per-sample MSE across the 13 features).

    Decision threshold is calibrated on held-out F0 validation samples
    at a target false-alarm rate (e.g. 95th percentile of F0 errors).
    """
    def __init__(self, in_dim=13, hidden_dims=(32, 16), latent_dim=8,
                 device="cpu"):
        self.device = device
        self.model  = AnomalyAE(in_dim, hidden_dims, latent_dim).to(device)
        self.threshold_ = None                   # set by .calibrate()

    # ------------------------------------------------------------------
    # Training -- on F0 samples only
    # ------------------------------------------------------------------
    def fit(self, X_f0, epochs=200, lr=1e-3, batch_size=50, verbose=False):
        """
        X_f0: tensor of shape (N, 1, 13) or (N, 13) containing ONLY F0 samples.
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_f0),
            batch_size=batch_size, shuffle=True,
        )
        self.model.train()
        for ep in range(epochs):
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                x_hat, _ = self.model(xb)
                target = xb.squeeze(1) if xb.dim() == 3 else xb
                loss = loss_fn(x_hat, target)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            if verbose and (ep + 1) % 20 == 0:
                print(f"epoch {ep+1:4d}  recon_mse={total/len(X_f0):.6f}")

    # ------------------------------------------------------------------
    # Scoring -- per-sample reconstruction MSE
    # ------------------------------------------------------------------
    def score(self, X):
        """Returns numpy array of per-sample reconstruction MSE. Higher = more anomalous."""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            x_hat, _ = self.model(X)
            target = X.squeeze(1) if X.dim() == 3 else X
            per_sample = ((x_hat - target) ** 2).mean(dim=1)     # (B,)
        return per_sample.cpu().numpy()

    # ------------------------------------------------------------------
    # Threshold calibration -- on held-out F0 validation samples
    # ------------------------------------------------------------------
    def calibrate(self, X_f0_val, target_far=0.05):
        """
        Pick threshold so that `target_far` fraction of F0 validation samples
        exceed it (i.e. a target false-alarm rate).
        """
        scores = self.score(X_f0_val)
        self.threshold_ = float(np.quantile(scores, 1.0 - target_far))
        return self.threshold_

    # ------------------------------------------------------------------
    # Binary prediction: 0 = normal (F0), 1 = anomaly (fault)
    # ------------------------------------------------------------------
    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Call .calibrate() before .predict().")
        return (self.score(X) > self.threshold_).astype(int)