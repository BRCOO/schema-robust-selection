from __future__ import annotations

from pathlib import Path

import pandas as pd

from .mimic_iv_schema import get_feature_specs


def _vasopressor_itemids() -> set[int]:
    for spec in get_feature_specs("core"):
        if spec.name == "vasopressor_on":
            return set(spec.itemids)
    raise ValueError("vasopressor_on feature spec not found")


def extract_vasopressor_anchors(
    inputevents: pd.DataFrame,
    escalation_ratio: float = 1.5,
) -> pd.DataFrame:
    frame = inputevents.copy()
    frame = frame[frame["itemid"].isin(_vasopressor_itemids())].copy()
    if "statusdescription" in frame.columns:
        mask = frame["statusdescription"].astype(str).str.contains("rewrite|stop|paused", case=False, na=False)
        frame = frame[~mask].copy()
    if frame.empty:
        return pd.DataFrame(columns=["stay_id", "anchor_time", "anchor_kind", "previous_rate", "current_rate"])

    frame["anchor_time"] = pd.to_datetime(frame["starttime"], errors="coerce")
    rate = pd.to_numeric(frame.get("rate"), errors="coerce")
    amount = pd.to_numeric(frame.get("amount"), errors="coerce")
    frame["current_rate"] = rate.fillna(amount).fillna(0.0)
    frame = frame.dropna(subset=["stay_id", "anchor_time"]).copy()
    frame["stay_id"] = frame["stay_id"].astype(int)
    frame = (
        frame.groupby(["stay_id", "anchor_time"], as_index=False)["current_rate"]
        .sum()
        .sort_values(["stay_id", "anchor_time"])
        .reset_index(drop=True)
    )

    anchors: list[dict] = []
    for stay_id, group in frame.groupby("stay_id", sort=True):
        previous_rate = 0.0
        for row in group.itertuples(index=False):
            current_rate = float(row.current_rate)
            if current_rate <= 0.0:
                previous_rate = 0.0
                continue
            anchor_kind = None
            if previous_rate <= 0.0:
                anchor_kind = "initiation"
            elif current_rate >= previous_rate * float(escalation_ratio):
                anchor_kind = "escalation"
            if anchor_kind is not None:
                anchors.append(
                    {
                        "stay_id": int(stay_id),
                        "anchor_time": pd.Timestamp(row.anchor_time),
                        "anchor_kind": anchor_kind,
                        "previous_rate": float(previous_rate),
                        "current_rate": current_rate,
                    }
                )
            previous_rate = current_rate

    return pd.DataFrame(anchors)
