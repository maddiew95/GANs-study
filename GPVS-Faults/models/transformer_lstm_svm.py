import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC

from models.transformer_lstm import Transformer_LSTM


class Transformer_LSTM_SVM:
    def __init__(self, n_classes: int = 8, device: str = "cpu",
                 seed: int = 0, **backbone_kw):
        self.device = device
        torch.manual_seed(seed)
        self.backbone = Transformer_LSTM(
            n_classes=n_classes, **backbone_kw).to(device)
        # Xavier init on Linear layers (match MATLAB's Glorot default)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # RBF-kernel SVM (Li Table IV). gamma='scale' matches sklearn default.
        self.svm = SVC(
            kernel="rbf", gamma="scale",
            decision_function_shape="ovr",
            random_state=seed,
        )

    def fit_backbone(self, X_train, y_train, X_val=None, y_val=None,
                     epochs=70, lr=1e-3, weight_decay=1e-4,
                     batch_size=50, seed=0, verbose=False):
        opt = optim.Adam(self.backbone.parameters(), lr=lr,
                         betas=(0.9, 0.999), eps=1e-8,
                         weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        loss_fn = nn.CrossEntropyLoss()

        # MATLAB-style shuffle once
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(X_train), generator=g)
        loader = DataLoader(TensorDataset(X_train[perm], y_train[perm]),
                            batch_size=batch_size, shuffle=False)

        for ep in range(1, epochs + 1):
            self.backbone.train()
            tr_loss = tr_correct = tr_n = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.backbone(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                tr_loss    += loss.item() * xb.size(0)
                tr_correct += (logits.argmax(1) == yb).sum().item()
                tr_n       += xb.size(0)
            scheduler.step()

            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                msg = (f"  [stage1] ep {ep:3d} | "
                       f"loss {tr_loss/tr_n:.4f} acc {tr_correct/tr_n:.3f}")
                if X_val is not None and y_val is not None:
                    va_loss, va_acc = self._eval_backbone(X_val, y_val, loss_fn)
                    msg += f" | val loss {va_loss:.4f} acc {va_acc:.3f}"
                print(msg)

    @torch.no_grad()
    def _eval_backbone(self, X, y, loss_fn):
        self.backbone.eval()
        logits = self.backbone(X.to(self.device))
        loss = loss_fn(logits, y.to(self.device)).item()
        acc = (logits.argmax(1) == y.to(self.device)).float().mean().item()
        return loss, acc

    @torch.no_grad()
    def _features(self, X):
        """Returns FC(8) logits per sample -- the inputs to what would have
        been the softmax in pure Transformer-LSTM."""
        self.backbone.eval()
        return self.backbone(X.to(self.device)).cpu().numpy()

    def fit_svm(self, X, y):
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y
        self.svm.fit(self._features(X), y_np)

    def predict(self, X):
        return self.svm.predict(self._features(X))