"""RETAIN baseline: reverse-time visit attention (NeurIPS 2016)."""

from __future__ import annotations

import torch

from .sequence_bundle import SequenceBundle, build_sequence_bundle, fit_binary_sequence_model
from .data_contract import EpisodeRecord


class RETAINClassifier(torch.nn.Module):
    """Simplified RETAIN-style model with reverse-time attention.

    Uses two GRUs: one to generate attention weights (reverse-time),
    one to generate context vectors (forward-time).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim * 3, hidden_size)
        self.attn_gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        self.context_gru = torch.nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=False)
        self.attn_out = torch.nn.Linear(hidden_size, 1)
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
        x = self.input_proj(x)

        batch_size, seq_len, _ = x.shape
        device = x.device

        # Reverse the sequence for attention GRU (RETAIN-style)
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        mask_3d = arange >= lengths.unsqueeze(1)

        x_rev = torch.flip(x, dims=[1])
        len_rev = lengths

        attn_out, _ = self.attn_gru(x_rev)
        attn_weights = self.attn_out(attn_out)
        attn_weights = torch.flip(attn_weights, dims=[1])

        # Context GRU forward
        context_out, _ = self.context_gru(x)

        # Apply attention
        attn_logits = attn_weights.squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask_3d, float("-inf"))
        attn_scores = torch.softmax(attn_logits, dim=1).unsqueeze(-1)

        weighted = (context_out * attn_scores).sum(dim=1)
        return self.head(weighted).squeeze(-1)


def train_evaluate_retain(
    records: list[EpisodeRecord],
    split: dict,
    random_state: int = 42,
    epochs: int = 8,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
) -> dict:
    """Train and evaluate a RETAIN classifier."""
    bundle = build_sequence_bundle(records, split)
    model = RETAINClassifier(input_dim=bundle.feature_dim, hidden_size=int(hidden_size), max_len=bundle.max_len)
    result = fit_binary_sequence_model(model=model, bundle=bundle, random_state=random_state, epochs=epochs, learning_rate=learning_rate)
    result["model_name"] = "retain_seq"
    return result
