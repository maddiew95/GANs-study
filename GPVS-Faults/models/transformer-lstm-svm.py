import torch.nn as nn
from sklearn.svm import SVC

class Transformer_LSTM_SVM:
    """
    Two-stage wrapper.
      stage a) train LiTransformer_LSTM end-to-end with softmax head
      stage b) freeze, extract FC(8) activations, fit RBF SVM on them
    """
    def __init__(self, n_classes: int = 8, device: str = "cpu", **kw):
        self.device   = device
        self.backbone = LiTransformer_LSTM(n_classes=n_classes, **kw).to(device)
        self.svm      = SVC(kernel="rbf", gamma="scale",
                            decision_function_shape="ovr")

    def fit_backbone(self, X, y, epochs=100, lr=1e-3, batch_size=50):
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

    def _features(self, X):
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone(X.to(self.device)).cpu().numpy()  # FC(8) logits

    def fit_svm(self, X, y):
        self.svm.fit(self._features(X), y)

    def predict(self, X):
        return self.svm.predict(self._features(X))