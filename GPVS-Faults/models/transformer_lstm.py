import torch
import torch.nn as nn


class Transformer_LSTM(nn.Module):
    def __init__(self, n_classes: int = 8, d_model: int = 3,
                 n_heads: int = 1, n_attn_layers: int = 2,
                 dropout: float = 0.41, lstm_hidden: int = 32,
                 seq_len: int = 13):
        super().__init__()
        # Project 1-D per timestep -> d_model embedding
        self.input_proj = nn.Linear(1, d_model)
        # Learned positional embedding (paper specifies dim 3)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_attn_layers)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):                 # x: (B, 1, 13)
        x = x.transpose(1, 2)             # (B, 13, 1)
        x = self.input_proj(x) + self.pos_emb   # (B, 13, 3)
        x = self.transformer(x)                 # (B, 13, 3)
        _, (h_n, _) = self.lstm(x)              # (1, B, 32)
        return self.fc(h_n.squeeze(0))          # (B, 8) logits