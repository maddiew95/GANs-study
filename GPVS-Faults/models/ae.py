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
                 device="cpu", seed=0):
        self.device = device
        torch.manual_seed(seed)
        self.model = AnomalyAE(in_dim, hidden_dims, latent_dim).to(device)
        # Xavier init for Linear layers (matches MATLAB Glorot default)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.threshold_ = None

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

    @torch.no_grad()
    def score(self, X):
        self.model.eval()
        X = X.to(self.device)
        x_hat, _ = self.model(X)
        target = X.squeeze(1) if X.dim() == 3 else X
        per_sample = ((x_hat - target) ** 2).mean(dim=1)
        return per_sample.cpu().numpy()

    def calibrate(self, X_f0_val, target_far=0.05):
        scores = self.score(X_f0_val)
        self.threshold_ = float(np.quantile(scores, 1.0 - target_far))
        return self.threshold_

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Call .calibrate() before .predict().")
        return (self.score(X) > self.threshold_).astype(int)