"""mTAN baseline: multi-time attention network (ICLR 2021)."""

from __future__ import annotations

import torch

from .sequence_bundle import SequenceBundle, build_sequence_bundle, fit_binary_sequence_model
from .data_contract import EpisodeRecord


class MTANClassifier(torch.nn.Module):
    """Simplified mTAN-style model with multi-time attention.

    Uses a time-aware attention mechanism over the sequence dimension.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim * 3, latent_dim)
        self.time_proj = torch.nn.Linear(1, latent_dim)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(latent_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_dim, 1),
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

        # Create time embeddings from lengths
        time_pos = torch.arange(seq_len, device=device).float().unsqueeze(0).unsqueeze(-1)
        time_embed = self.time_proj(time_pos)

        # Apply attention
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        key_padding_mask = arange >= lengths.unsqueeze(1)

        x = x + time_embed
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (attn_out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def train_evaluate_mtan(
    records: list[EpisodeRecord],
    split: dict,
    random_state: int = 42,
    epochs: int = 8,
    latent_dim: int = 64,
    learning_rate: float = 1e-3,
) -> dict:
    """Train and evaluate an mTAN classifier."""
    bundle = build_sequence_bundle(records, split)
    model = MTANClassifier(input_dim=bundle.feature_dim, latent_dim=int(latent_dim), max_len=bundle.max_len)
    result = fit_binary_sequence_model(model=model, bundle=bundle, random_state=random_state, epochs=epochs, learning_rate=learning_rate)
    result["model_name"] = "mtan_seq"
    return result
