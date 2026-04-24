import torch.nn as nn
from xgboost import XGBClassifier

class LSTM(nn.Module):
    def __init__(self, n_classes: int = 8, lstm_hidden: int = 32):
        super().__init__()
        # Input (B,1,13) -> (B, 13, 1): 13 timesteps, 1 feature each
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden,
                            batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)   # used only in stage (a)

    def features(self, x):                # x: (B, 1, 13)
        x = x.transpose(1, 2)             # (B, 13, 1)
        _, (h_n, _) = self.lstm(x)        # (1, B, 32)
        return h_n.squeeze(0)             # (B, 32)

    def forward(self, x):
        return self.head(self.features(x))



class LSTM_XGB:
    def __init__(self, n_classes: int = 8, lstm_hidden: int = 32,
                 device: str = "cpu"):
        self.device  = device
        self.backbone = LSTM(n_classes, lstm_hidden).to(device)
        self.xgb = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", tree_method="hist",
        )

    def fit_lstm(self, X, y, epochs=100, lr=1e-3, batch_size=50):
        opt = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=batch_size, shuffle=True)
        self.backbone.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.backbone(xb.to(self.device)),
                               yb.to(self.device))
                loss.backward()
                opt.step()

    def _extract(self, X):
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone.features(X.to(self.device)).cpu().numpy()

    def fit_xgb(self, X, y):
        self.xgb.fit(self._extract(X), y)

    def predict(self, X):
        return self.xgb.predict(self._extract(X))