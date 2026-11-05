"""AdaCare baseline: channel reweighting with causal aggregation (AAAI 2020)."""

from __future__ import annotations

import torch

from .sequence_bundle import SequenceBundle, build_sequence_bundle, fit_binary_sequence_model
from .data_contract import EpisodeRecord


class AdaCareClassifier(torch.nn.Module):
    """Simplified AdaCare-style model with channel-wise reweighting.

    Uses feature-wise attention and temporal aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        total_input_dim = input_dim * 3
        self.feature_gate = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, input_dim),
            torch.nn.Sigmoid(),
        )
        self.temporal_gru = torch.nn.GRU(
            input_dim, hidden_size, batch_first=True, bidirectional=False, num_layers=1,
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([values, observed_mask, deltas], dim=-1)
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Channel-wise reweighting (AdaCare-style)
        gates = self.feature_gate(x)
        gated_values = values * gates[:, :, :values.shape[-1]]

        # Temporal aggregation with GRU
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        key_padding_mask = arange >= lengths.unsqueeze(1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            gated_values, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.temporal_gru(packed)
        context = hidden.squeeze(0)
        return self.head(context).squeeze(-1)


def train_evaluate_adacare(
    records: list[EpisodeRecord],
    split: dict,
    random_state: int = 42,
    epochs: int = 8,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
) -> dict:
    """Train and evaluate an AdaCare classifier."""
    bundle = build_sequence_bundle(records, split)
    model = AdaCareClassifier(input_dim=bundle.feature_dim, hidden_size=int(hidden_size), max_len=bundle.max_len)
    result = fit_binary_sequence_model(model=model, bundle=bundle, random_state=random_state, epochs=epochs, learning_rate=learning_rate)
    result["model_name"] = "adacare_seq"
    return result
