import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class AnomalyAE(nn.Module):
    def __init__(self, in_dim: int = 13, hidden_dims=(32, 16),
                 latent_dim: int = 8):
        super().__init__()
        # encoder
        enc = []
        prev = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc += [nn.Linear(prev, latent_dim)]   # linear latent
        self.encoder = nn.Sequential(*enc)

        # decoder (mirror)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec += [nn.Linear(prev, in_dim)]       # linear output
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)                   # (B, 1, 13) -> (B, 13)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.encoder(x)


class AnomalyDetector:
    def __init__(self, in_dim=13, hidden_dims=(32, 16), latent_dim=8,
                 device="cpu", seed=0, score_mode="mse"):
        assert score_mode in ("mse", "mahalanobis", "combined")
        self.device = device
        torch.manual_seed(seed)
        self.score_mode = score_mode
        self.model = AnomalyAE(in_dim, hidden_dims, latent_dim).to(device)
        # Xavier init for Linear layers (matches MATLAB Glorot default)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.threshold_ = None
        # Mahalanobis stats — fitted in fit_mahalanobis()
        self.mu_         = None
        self.cov_inv_    = None
        # Score normalization stats — fitted in _fit_score_normalizers()
        self._mse_mean   = None
        self._mse_std    = None
        self._mah_mean   = None
        self._mah_std    = None

    def fit(self, X_f0, X_f0_val=None, epochs=70, lr=1e-2,
            weight_decay=1e-4, batch_size=50, seed=0, verbose=False):
        opt = optim.Adam(self.model.parameters(), lr=lr,
                         betas=(0.9, 0.999), eps=1e-8,
                         weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        loss_fn = nn.MSELoss()

        # MATLAB-style shuffle once
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X_f0), generator=g)
        loader = DataLoader(TensorDataset(X_f0[perm]),
                            batch_size=batch_size, shuffle=False)

        for ep in range(1, epochs + 1):
            self.model.train()
            tr_loss = tr_n = 0
            for (xb,) in loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                x_hat, _ = self.model(xb)
                target = xb.squeeze(1) if xb.dim() == 3 else xb
                loss = loss_fn(x_hat, target)
                loss.backward()
                opt.step()
                tr_loss += loss.item() * xb.size(0)
                tr_n    += xb.size(0)
            scheduler.step()

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                msg = f"  ep {ep:3d} | train MSE {tr_loss/tr_n:.6f}"
                if X_f0_val is not None:
                    val_mse = self.score(X_f0_val).mean()
                    msg += f" | val MSE {val_mse:.6f}"
                print(msg)
        # After training, fit the latent statistics on the F0 training set
        # (needed for Mahalanobis scoring; harmless if mode == "mse")
        self._fit_mahalanobis_stats(X_f0)
        # Also fit normalizers on F0 training set so combined scoring
        # weights MSE and Mahalanobis comparably
        self._fit_score_normalizers(X_f0)
    
    @torch.no_grad()
    def _fit_mahalanobis_stats(self, X_f0):
        """Compute mean and inverse covariance of F0 latent representations."""
        self.model.eval()
        z = self.model.encode(X_f0.to(self.device)).cpu().numpy()  # (N, latent_dim)
        self.mu_ = z.mean(axis=0)                                  # (latent_dim,)
        cov = np.cov(z.T) + 1e-6 * np.eye(z.shape[1])              # ridge for stability
        self.cov_inv_ = np.linalg.pinv(cov)

    @torch.no_grad()
    def _fit_score_normalizers(self, X_f0):
        """Fit mean/std of each scorer on F0 training samples, so that
        scores are on comparable scales when combining."""
        mse = self._mse_score(X_f0)
        mah = self._mahalanobis_score(X_f0)
        self._mse_mean, self._mse_std = float(mse.mean()), float(mse.std() + 1e-8)
        self._mah_mean, self._mah_std = float(mah.mean()), float(mah.std() + 1e-8)

    @torch.no_grad()
    def _mse_score(self, X):
        self.model.eval()
        X = X.to(self.device)
        x_hat, _ = self.model(X)
        target = X.squeeze(1) if X.dim() == 3 else X
        return ((x_hat - target) ** 2).mean(dim=1).cpu().numpy()

    @torch.no_grad()
    def _mahalanobis_score(self, X):
        if self.mu_ is None or self.cov_inv_ is None:
            raise RuntimeError(".fit() must be called before mahalanobis scoring.")
        self.model.eval()
        z = self.model.encode(X.to(self.device)).cpu().numpy()     # (N, latent_dim)
        diff = z - self.mu_                                        # (N, latent_dim)
        # quadratic form: sqrt((diff @ cov_inv * diff).sum(axis=1))
        return np.sqrt(np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff))

    def score(self, X):
        if self.score_mode == "mse":
            return self._mse_score(X)
        elif self.score_mode == "mahalanobis":
            return self._mahalanobis_score(X)
        elif self.score_mode == "combined":
            mse = self._mse_score(X)
            mah = self._mahalanobis_score(X)
            mse_z = (mse - self._mse_mean) / self._mse_std
            mah_z = (mah - self._mah_mean) / self._mah_std
            return mse_z + mah_z
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")

    def calibrate(self, X_f0_val, target_far=0.05):
        scores = self.score(X_f0_val)
        self.threshold_ = float(np.quantile(scores, 1.0 - target_far))
        return self.threshold_

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Call .calibrate() before .predict().")
        return (self.score(X) > self.threshold_).astype(int)