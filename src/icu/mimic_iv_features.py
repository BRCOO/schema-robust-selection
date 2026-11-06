import numpy as np
import pandas as pd


def _build_full_index(start_time, end_time, resample_minutes: int):
    if start_time is None or end_time is None:
        return None
    start = pd.to_datetime(start_time, errors="coerce")
    end = pd.to_datetime(end_time, errors="coerce")
    if pd.isna(start) or pd.isna(end) or end < start:
        return None
    freq = f"{int(resample_minutes)}min"
    start_bucket = start.floor(freq)
    end_bucket = end.floor(freq)
    if end_bucket < start_bucket:
        return None
    return pd.date_range(start_bucket, end_bucket, freq=freq)


def _empty_matrix_result(
    feature_names: list[str] | None,
    resample_minutes: int,
    full_index=None,
    include_missing_mask: bool = False,
    fill_map: dict[str, str] | None = None,
):
    ordered = list(feature_names or [])
    row_count = 0 if full_index is None else int(len(full_index))
    matrix = np.zeros((row_count, len(ordered)), dtype=float)
    meta = {
        "feature_names": ordered,
        "resample_minutes": int(resample_minutes),
        "start_time": None if full_index is None or row_count == 0 else full_index[0].isoformat(),
        "end_time": None if full_index is None or row_count == 0 else full_index[-1].isoformat(),
        "shape": [row_count, len(ordered)],
    }
    if include_missing_mask:
        missing_mask = np.ones((row_count, len(ordered)), dtype=bool)
        for idx, feature_name in enumerate(ordered):
            if (fill_map or {}).get(feature_name) == "zero":
                missing_mask[:, idx] = False
        meta["missing_mask"] = missing_mask
    return matrix, meta


def _robust_scale_columns(frame: pd.DataFrame, fill_map: dict[str, str]) -> pd.DataFrame:
    scaled = frame.copy()
    numeric_cols = [col for col in scaled.columns if fill_map.get(col, "interpolate") == "interpolate"]
    if not numeric_cols:
        return scaled
    arr = scaled[numeric_cols].to_numpy(dtype=float)
    med = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    iqr = q3 - q1
    iqr = np.where(iqr < 1e-9, 1.0, iqr)
    scaled[numeric_cols] = (arr - med) / (iqr + 1e-9)
    return scaled


def build_feature_matrix(
    rows: pd.DataFrame,
    resample_minutes: int = 60,
    feature_names: list[str] | None = None,
    window_start=None,
    window_end=None,
    fill_map: dict[str, str] | None = None,
    include_missing_mask: bool = False,
):
    full_index = _build_full_index(window_start, window_end, resample_minutes)
    resolved_fill_map = dict(fill_map or {})
    if rows.empty:
        return _empty_matrix_result(
            feature_names,
            resample_minutes,
            full_index=full_index,
            include_missing_mask=include_missing_mask,
            fill_map=resolved_fill_map,
        )

    data = rows.copy()
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce")
    data = data.dropna(subset=["event_time", "feature_name", "value"]).copy()
    if data.empty:
        return _empty_matrix_result(
            feature_names,
            resample_minutes,
            full_index=full_index,
            include_missing_mask=include_missing_mask,
            fill_map=resolved_fill_map,
        )

    data["bucket"] = data["event_time"].dt.floor(f"{int(resample_minutes)}min")
    if feature_names is None:
        feature_names = list(dict.fromkeys(data["feature_name"].tolist()))

    resolved_fill_map.update(data.groupby("feature_name")["fill"].first().to_dict())
    agg_map = data.groupby("feature_name")["agg"].first().to_dict()

    series_list = []
    for feature_name in feature_names:
        feature_rows = data[data["feature_name"] == feature_name]
        if feature_rows.empty:
            continue
        agg_name = agg_map.get(feature_name, "mean")
        series = feature_rows.groupby("bucket")["value"].agg(agg_name).rename(feature_name)
        series_list.append(series)

    frame = pd.concat(series_list, axis=1).sort_index() if series_list else pd.DataFrame(columns=feature_names)
    if frame.empty:
        return _empty_matrix_result(
            feature_names,
            resample_minutes,
            full_index=full_index,
            include_missing_mask=include_missing_mask,
            fill_map=resolved_fill_map,
        )

    if full_index is None:
        full_index = pd.date_range(frame.index.min(), frame.index.max(), freq=f"{int(resample_minutes)}min")
    frame = frame.reindex(full_index)
    frame = frame.reindex(columns=feature_names)
    missing_mask = pd.DataFrame(False, index=frame.index, columns=frame.columns, dtype=bool)

    for column in frame.columns:
        fill_mode = resolved_fill_map.get(column, "interpolate")
        if fill_mode == "zero":
            missing_mask[column] = False
            frame[column] = frame[column].fillna(0.0)
        else:
            missing_mask[column] = frame[column].isna()
            if frame[column].notna().any():
                frame[column] = frame[column].interpolate(limit_direction="both").ffill().bfill()
            else:
                frame[column] = 0.0

    frame = _robust_scale_columns(frame, resolved_fill_map)
    meta = {
        "feature_names": list(frame.columns),
        "resample_minutes": int(resample_minutes),
        "start_time": full_index[0].isoformat(),
        "end_time": full_index[-1].isoformat(),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
    }
    if include_missing_mask:
        meta["missing_mask"] = missing_mask.to_numpy(dtype=bool)
    return frame.to_numpy(dtype=float), meta
