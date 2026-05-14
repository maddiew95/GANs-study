import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC


class TransformerLSTMBackbone(nn.Module):
    def __init__(self, d_model: int = 13, n_heads: int = 1,
                 n_attn_layers: int = 2, dropout: float = 0.41,
                 lstm_hidden: int = 32, seq_len: int = 13,
                 # we still need a softmax head for stage-1 training
                 n_classes: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_attn_layers)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden,
                            batch_first=True)
        # Auxiliary FC head only used during stage-1 cross-entropy training
        self.aux_fc = nn.Linear(lstm_hidden, n_classes)

    def features(self, x):
        """Returns LSTM hidden state — what Li feeds to the SVM."""
        x = x.transpose(1, 2)
        x = self.input_proj(x) + self.pos_emb
        x = self.transformer(x)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)                # (B, lstm_hidden=32)

    def forward(self, x):
        """Returns logits for stage-1 cross-entropy training."""
        return self.aux_fc(self.features(x))


class Transformer_LSTM_SVM:
    def __init__(self, n_classes: int = 8, device: str = "cpu",
                 seed: int = 0, **backbone_kw):
        self.device = device
        torch.manual_seed(seed)
        # Per Li Table IV for the SVM variant: dropout=0.41, lstm=32
        defaults = dict(dropout=0.41, lstm_hidden=32, d_model=13)
        defaults.update(backbone_kw)
        self.backbone = TransformerLSTMBackbone(
            n_classes=n_classes, **defaults).to(device)
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.svm = SVC(
            kernel="rbf", gamma="scale",
            decision_function_shape="ovr",
            random_state=seed,
        )

    def fit_backbone(self, X_train, y_train, X_val=None, y_val=None,
                     epochs=200, lr=1e-3,
                     batch_size=50, seed=0, verbose=False):
        # NOTE: no weight_decay, no scheduler — matches working Transformer-LSTM setup
        opt = optim.Adam(self.backbone.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

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

            if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
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
        """Returns LSTM hidden state — what Li feeds to the SVM."""
        self.backbone.eval()
        return self.backbone.features(X.to(self.device)).cpu().numpy()

    def fit_svm(self, X, y):
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y
        self.svm.fit(self._features(X), y_np)

    def predict(self, X):
        return self.svm.predict(self._features(X))