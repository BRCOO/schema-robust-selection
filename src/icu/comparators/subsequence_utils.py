from __future__ import annotations

import math

import numpy as np


def contains_subsequence(sequence: list[str], pattern: list[str]) -> bool:
    """Return True if pattern appears in order within sequence."""
    if not pattern:
        return False
    cursor = 0
    for token in sequence:
        if token == pattern[cursor]:
            cursor += 1
            if cursor >= len(pattern):
                return True
    return False


def rule_fires(
    sequence: list[str],
    antecedent: list[str],
    consequent: list[str],
) -> bool:
    """Return True if antecedent appears before consequent."""
    if not antecedent or not consequent:
        return False
    if not contains_subsequence(sequence, antecedent + consequent):
        return False
    return True


def impute_series(series: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Forward/backward fill a 1D series; default to zeros if fully missing."""
    arr = np.asarray(series, dtype=float).copy()
    if mask is not None:
        miss = np.asarray(mask, dtype=bool)
        arr[miss] = np.nan
    if arr.size == 0:
        return arr
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=float)
    first_idx = int(np.flatnonzero(valid)[0])
    arr[:first_idx] = arr[first_idx]
    for idx in range(first_idx + 1, arr.shape[0]):
        if not np.isfinite(arr[idx]):
            arr[idx] = arr[idx - 1]
    for idx in range(arr.shape[0] - 2, -1, -1):
        if not np.isfinite(arr[idx]):
            arr[idx] = arr[idx + 1]
    return arr.astype(float)


def z_normalize(values: np.ndarray) -> np.ndarray:
    """Z-normalize a 1D array with a stable zero-variance fallback."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std


def sliding_windows(series: np.ndarray, length: int, step: int = 1) -> list[np.ndarray]:
    """Extract fixed-length sliding windows from a 1D series."""
    arr = np.asarray(series, dtype=float)
    if length <= 0 or arr.shape[0] < length:
        return []
    windows: list[np.ndarray] = []
    for start in range(0, arr.shape[0] - length + 1, max(1, int(step))):
        windows.append(arr[start : start + length])
    return windows


def min_window_distance(series: np.ndarray, template: np.ndarray) -> float:
    """Minimum z-normalized Euclidean distance between a series and template."""
    temp = z_normalize(np.asarray(template, dtype=float))
    length = int(temp.shape[0])
    candidates = sliding_windows(np.asarray(series, dtype=float), length, step=1)
    if not candidates:
        return float("inf")
    best = float("inf")
    for window in candidates:
        dist = float(np.linalg.norm(z_normalize(window) - temp)) / math.sqrt(max(length, 1))
        if dist < best:
            best = dist
    return best


def min_window_distance_raw(series: np.ndarray, template: np.ndarray) -> float:
    """Minimum raw Euclidean distance between a series and template."""
    temp = np.asarray(template, dtype=float)
    length = int(temp.shape[0])
    candidates = sliding_windows(np.asarray(series, dtype=float), length, step=1)
    if not candidates:
        return float("inf")
    best = float("inf")
    scale = max(float(np.std(temp)), 1.0)
    for window in candidates:
        dist = float(np.linalg.norm(np.asarray(window, dtype=float) - temp)) / (math.sqrt(max(length, 1)) * scale)
        if dist < best:
            best = dist
    return best
