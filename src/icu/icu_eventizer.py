from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


DEFAULT_MISSINGNESS_FEATURES = {
    "HR",
    "RR",
    "SpO2",
    "SBP",
    "DBP",
    "MAP",
    "Temp",
    "Lactate",
    "pH",
    "Creatinine",
    "Glucose",
    "Bilirubin_Total",
    "WBC",
    "Platelets",
}

DEFAULT_INTERVENTION_FEATURES = {
    "ventilation_on",
    "vasopressor_on",
    "sedation_on",
}

DEFAULT_OBSERVATION_DENSITY_FEATURES = set(DEFAULT_MISSINGNESS_FEATURES)

DEFAULT_DETERIORATION_SIGN = {
    "HR": 1.0,
    "RR": 1.0,
    "SpO2": -1.0,
    "SBP": -1.0,
    "DBP": -1.0,
    "MAP": -1.0,
    "Temp": 1.0,
    "Lactate": 1.0,
    "pH": -1.0,
    "Creatinine": 1.0,
    "Glucose": 1.0,
    "Bilirubin_Total": 1.0,
    "WBC": 1.0,
    "Platelets": -1.0,
}


class ICUEventizer:
    def __init__(
        self,
        feature_names: list[str] | None = None,
        n_bins: int = 4,
        window_size: int = 200,
        quantile_stride: int = 10,
        change_threshold: float = 0.1,
        support_decay_lambda: float | None = None,
        support_min: float | None = 0.01,
        support_max: float | None = 0.95,
        missingness_features: list[str] | None = None,
        intervention_features: list[str] | None = None,
        observation_density_features: list[str] | None = None,
        deterioration_multiplier: float = 0.75,
        recovery_multiplier: float = 1.25,
        intervention_on_threshold: float = 0.5,
        sparse_observation_threshold: float = 0.5,
        min_observed_for_levels: int = 3,
        missing_repeat_stride: int = 3,
    ):
        self.feature_names = list(feature_names or [])
        self.n_bins = int(n_bins)
        self.window_size = int(window_size)
        self.quantile_stride = max(1, int(quantile_stride))
        self.change_threshold = float(change_threshold)
        self.support_decay_lambda = support_decay_lambda
        self.support_min = support_min
        self.support_max = support_max
        self.missingness_features = set(missingness_features or DEFAULT_MISSINGNESS_FEATURES)
        self.intervention_features = set(intervention_features or DEFAULT_INTERVENTION_FEATURES)
        self.observation_density_features = set(observation_density_features or DEFAULT_OBSERVATION_DENSITY_FEATURES)
        self.deterioration_multiplier = float(deterioration_multiplier)
        self.recovery_multiplier = float(recovery_multiplier)
        self.intervention_on_threshold = float(intervention_on_threshold)
        self.sparse_observation_threshold = float(sparse_observation_threshold)
        self.min_observed_for_levels = max(1, int(min_observed_for_levels))
        self.missing_repeat_stride = max(1, int(missing_repeat_stride))
        self.event_labels: dict[int, str] = {}
        self._custom_event_ids: dict[str, int] = {}

    @property
    def _events_per_var(self) -> int:
        return self.n_bins + 2

    def _encode_level_event(self, var_idx: int, bin_idx: int) -> int:
        return var_idx * self._events_per_var + int(bin_idx)

    def _encode_change_event(self, var_idx: int, is_up: bool) -> int:
        return var_idx * self._events_per_var + (self.n_bins if is_up else self.n_bins + 1)

    def _custom_event_start(self) -> int:
        return len(self.feature_names) * self._events_per_var

    def _get_custom_event(self, label: str) -> int:
        event_id = self._custom_event_ids.get(label)
        if event_id is not None:
            return event_id
        event_id = self._custom_event_start() + len(self._custom_event_ids)
        self._custom_event_ids[label] = event_id
        self.event_labels[event_id] = label
        return event_id

    def describe_event(self, event_id: int) -> str:
        if event_id in self.event_labels:
            return self.event_labels[event_id]
        if not self.feature_names:
            return str(event_id)
        var_idx, offset = divmod(int(event_id), self._events_per_var)
        if var_idx >= len(self.feature_names):
            return str(event_id)
        feature_name = self.feature_names[var_idx]
        if offset < self.n_bins:
            return f"{feature_name}:level_{offset}"
        if offset == self.n_bins:
            return f"{feature_name}:up"
        return f"{feature_name}:down"

    def _resolve_feature_names(self, feature_meta: dict | None, width: int) -> list[str]:
        if feature_meta and feature_meta.get("feature_names"):
            names = list(feature_meta["feature_names"])
        else:
            names = list(self.feature_names)
        if len(names) != width:
            raise ValueError(f"feature_names width mismatch: expected {width}, got {len(names)}")
        self.feature_names = names
        return names

    def _resolve_missing_mask(self, X: np.ndarray, feature_meta: dict | None) -> np.ndarray:
        if feature_meta and feature_meta.get("missing_mask") is not None:
            mask = np.asarray(feature_meta["missing_mask"], dtype=bool)
            if mask.shape != X.shape:
                raise ValueError(f"missing_mask shape mismatch: expected {X.shape}, got {mask.shape}")
            return mask
        return np.isnan(X)

    def _resolve_feature_set(self, feature_meta: dict | None, key: str, default: set[str]) -> set[str]:
        if feature_meta and feature_meta.get(key) is not None:
            return set(feature_meta[key])
        return set(default)

    def _fill_missing(self, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        filled = X.astype(float, copy=True)
        for column_idx in range(filled.shape[1]):
            column = pd.Series(filled[:, column_idx])
            if missing_mask[:, column_idx].all():
                filled[:, column_idx] = 0.0
                continue
            column = column.interpolate(limit_direction="both").ffill().bfill()
            filled[:, column_idx] = column.to_numpy(dtype=float)
        return filled

    def _filter_by_support(self, events_sequence: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        T = len(events_sequence)
        if T == 0:
            return events_sequence

        min_support = 0.0 if self.support_min is None else float(self.support_min)
        max_support = 1.0 if self.support_max is None else float(self.support_max)
        if min_support <= 0.0 and max_support >= 1.0:
            return events_sequence

        if self.support_decay_lambda is None:
            half_life = max(1.0, T / 5.0)
            decay_lambda = np.log(2) / half_life
        else:
            decay_lambda = float(self.support_decay_lambda)

        ts = np.arange(T)
        weights = np.exp(-decay_lambda * (T - ts))
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            return events_sequence

        event_weights = defaultdict(float)
        for t, events in enumerate(events_sequence):
            weight = weights[t]
            for event_id in events:
                event_weights[event_id] += weight

        filtered = []
        for events in events_sequence:
            filtered.append(
                tuple(
                    event_id
                    for event_id in events
                    if min_support <= (event_weights[event_id] / weights_sum) <= max_support
                )
            )
        return filtered

    def transform(self, X: np.ndarray, feature_meta: dict | None = None) -> list[tuple[int, ...]]:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"expected 2D input, got shape {matrix.shape}")
        feature_names = self._resolve_feature_names(feature_meta, matrix.shape[1])
        missing_mask = self._resolve_missing_mask(matrix, feature_meta)
        filled = self._fill_missing(matrix, missing_mask)
        missingness_features = self._resolve_feature_set(feature_meta, "missingness_features", self.missingness_features)
        intervention_features = self._resolve_feature_set(feature_meta, "intervention_features", self.intervention_features)
        observation_density_features = self._resolve_feature_set(
            feature_meta,
            "observation_density_features",
            self.observation_density_features,
        )
        self.event_labels = {}
        self._custom_event_ids = {}

        T, d = filled.shape
        events_sequence: list[tuple[int, ...]] = []
        last_bins: list[np.ndarray | None] = [None] * d
        last_intervention_state: dict[str, bool] = {}
        last_missing_state: dict[str, bool] = {}
        last_missing_streak: dict[str, int] = {}

        for t in range(T):
            events = set()
            start = max(0, t - self.window_size + 1)

            for var_idx, feature_name in enumerate(feature_names):
                window_values = filled[start : t + 1, var_idx]
                current_is_missing = bool(missing_mask[t, var_idx])
                window_missing_mask = missing_mask[start : t + 1, var_idx]
                observed_count = int((~window_missing_mask).sum())
                window_length = int(len(window_missing_mask))

                if feature_name in missingness_features:
                    previous_missing = last_missing_state.get(feature_name, False)
                    if current_is_missing:
                        streak = last_missing_streak.get(feature_name, 0) + 1
                        should_emit_missing = (not previous_missing) or (streak % self.missing_repeat_stride == 0)
                        if should_emit_missing:
                            events.add(self._get_custom_event(f"{feature_name}:missing"))
                        last_missing_streak[feature_name] = streak
                    else:
                        last_missing_streak[feature_name] = 0
                    last_missing_state[feature_name] = current_is_missing

                if feature_name in observation_density_features and window_length >= 3 and 0 < observed_count < window_length:
                    observed_fraction = observed_count / max(1, window_length)
                    if observed_fraction <= self.sparse_observation_threshold:
                        events.add(self._get_custom_event(f"{feature_name}:sparse_measurement"))

                if feature_name in intervention_features:
                    current_state = bool(filled[t, var_idx] > self.intervention_on_threshold)
                    previous_state = last_intervention_state.get(feature_name)
                    if previous_state is not None and current_state != previous_state:
                        label = f"{feature_name}:onset" if current_state else f"{feature_name}:offset"
                        events.add(self._get_custom_event(label))
                    last_intervention_state[feature_name] = current_state

                if current_is_missing:
                    continue

                required_observed = min(window_length, max(2, self.min_observed_for_levels))
                if observed_count >= required_observed and (
                    t % self.quantile_stride == 0 or last_bins[var_idx] is None
                ):
                    quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
                    bins = np.quantile(window_values, quantiles)
                    last_bins[var_idx] = np.unique(bins)

                bins = last_bins[var_idx]
                if bins is not None and len(bins) > 1:
                    bin_idx = np.searchsorted(bins[1:-1], filled[t, var_idx], side="right")
                    bin_idx = min(int(bin_idx), self.n_bins - 1)
                    events.add(self._encode_level_event(var_idx, bin_idx))

                if t > 0:
                    delta = filled[t, var_idx] - filled[t - 1, var_idx]
                    window_std = float(np.std(window_values)) + 1e-8
                    if abs(delta) > self.change_threshold * window_std:
                        events.add(self._encode_change_event(var_idx, delta > 0))

                    sign = DEFAULT_DETERIORATION_SIGN.get(feature_name)
                    if sign is not None:
                        signed_delta = delta * sign
                        if signed_delta > self.change_threshold * self.deterioration_multiplier * window_std:
                            events.add(self._get_custom_event(f"{feature_name}:deterioration"))
                        elif signed_delta < -self.change_threshold * self.recovery_multiplier * window_std:
                            events.add(self._get_custom_event(f"{feature_name}:recovery"))

            events_sequence.append(tuple(sorted(events)))

        return self._filter_by_support(events_sequence)



