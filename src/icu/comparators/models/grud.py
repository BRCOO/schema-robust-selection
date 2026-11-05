from __future__ import annotations

import torch
from torch import nn


class GRUDLite(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim * 3, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.head = nn.Linear(self.hidden_size, 1)

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        features = torch.cat([values, observed_mask, deltas], dim=-1)
        encoded = self.encoder(features)
        outputs, _ = self.gru(encoded)
        last_hidden = outputs[:, -1, :]
        return self.head(last_hidden).squeeze(-1)
