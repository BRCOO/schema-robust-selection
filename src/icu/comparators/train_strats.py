"""STraTS baseline: sparse irregular transformer (TKDD 2022)."""

from __future__ import annotations

import torch

from .sequence_bundle import SequenceBundle, build_sparse_event_bundle, fit_binary_sequence_model
from .data_contract import EpisodeRecord


class STraTSClassifier(torch.nn.Module):
    """Simplified STraTS-style transformer for sparse irregular time series.

    Uses time embeddings and self-attention over the observation dimension.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
        use_time_embedding: bool = True,
        use_self_supervised_pretraining: bool = False,
    ) -> None:
        super().__init__()
        self.use_time_embedding = use_time_embedding
        self.input_proj = torch.nn.Linear(input_dim * 3, d_model)
        if use_time_embedding:
            self.time_embed = torch.nn.Linear(1, d_model)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, 1),
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

        if self.use_time_embedding:
            time_pos = torch.arange(seq_len, device=device).float().unsqueeze(0).unsqueeze(-1)
            x = x + self.time_embed(time_pos)

        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        key_padding_mask = arange >= lengths.unsqueeze(1)

        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (enc * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def train_evaluate_strats(
    records: list[EpisodeRecord],
    split: dict,
    random_state: int = 42,
    epochs: int = 8,
    d_model: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
) -> dict:
    """Train and evaluate a STraTS-style classifier.

    Note: This is a supervised-only variant (no self-supervised pretraining).
    """
    bundle = build_sparse_event_bundle(records, split, time_encoding="relative_hour", include_observation_mask=True)
    model = STraTSClassifier(
        input_dim=bundle.feature_dim,
        d_model=int(d_model),
        num_layers=int(num_layers),
        max_len=bundle.max_len,
        use_time_embedding=True,
        use_self_supervised_pretraining=False,
    )
    result = fit_binary_sequence_model(model=model, bundle=bundle, random_state=random_state, epochs=epochs, learning_rate=learning_rate)
    result["model_name"] = "strats_seq"
    return result
