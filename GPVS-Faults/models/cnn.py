import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_classes: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=12,
                              kernel_size=2, stride=1)   # -> (B, 12, 12)
        self.bn   = nn.BatchNorm1d(12)
        self.pool = nn.MaxPool1d(kernel_size=2)          # -> (B, 12, 6)
        self.flat = nn.Flatten()
        self.fc   = nn.Linear(12 * 6, n_classes)         # -> (B, 8)

    def forward(self, x):                 # x: (B, 1, 13)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.fc(x)                 # raw logits -> CrossEntropyLoss