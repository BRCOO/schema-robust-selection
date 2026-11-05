
import numpy as np
import os
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
from bisect import bisect_left
import time
import heapq
import array
try:
    from bitarray import bitarray
    BITARRAY_AVAILABLE = True
except Exception:
    BITARRAY_AVAILABLE = False
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')


_MISSING = object()
_INTERSECT_NUMPY_THRESHOLD = 512

_EVENT_N_BINS = None
_EVENTS_PER_VAR = None
_EVENT_ID_LABELS = {}


def _dataclass_with_slots(*args, **kwargs):
    def decorator(cls):
        try:
            return dataclass(*args, **kwargs, slots=True)(cls)
        except TypeError:
            return dataclass(*args, **kwargs)(cls)
    return decorator


def _set_event_encoder(n_bins: int) -> None:
    global _EVENT_N_BINS, _EVENTS_PER_VAR, _EVENT_ID_LABELS
    _EVENT_N_BINS = int(n_bins)
    _EVENTS_PER_VAR = _EVENT_N_BINS + 2
    _EVENT_ID_LABELS = {}


def _set_custom_event_labels(labels: Optional[Dict[int, str]]) -> None:
    global _EVENT_ID_LABELS
    _EVENT_ID_LABELS = dict(labels or {})


def _encode_level_event(v: int, bin_idx: int) -> int:
    return v * _EVENTS_PER_VAR + bin_idx


def _encode_change_event(v: int, is_up: bool) -> int:
    return v * _EVENTS_PER_VAR + (_EVENT_N_BINS if is_up else _EVENT_N_BINS + 1)


def _event_id_to_str(event_id: int) -> str:
    if event_id in _EVENT_ID_LABELS:
        return _EVENT_ID_LABELS[event_id]
    if _EVENT_N_BINS is None or _EVENTS_PER_VAR is None:
        return str(event_id)
    v, offset = divmod(event_id, _EVENTS_PER_VAR)
    if offset < _EVENT_N_BINS:
        return f"v{v}_L{offset}"
    if offset == _EVENT_N_BINS:
        return f"v{v}_↑"
    return f"v{v}_↓"


def _format_itemset(itemset: Tuple[int, ...]) -> str:
    parts = [_event_id_to_str(e) for e in itemset]
    return "{" + ", ".join(parts) + "}"


def _itemset_key(itemset: Iterable[int]) -> Tuple[int, ...]:
    if isinstance(itemset, tuple):
        return itemset
    return tuple(sorted(itemset))


class LRUCache:
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._data = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return _MISSING
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if self.maxsize is not None and len(self._data) > self.maxsize:
            self._data.popitem(last=False)

# ==================== 数据结构定义 ====================

@_dataclass_with_slots(frozen=True)
class Motif:
    """Motif (Episode) 结构"""
    itemsets: Tuple[Tuple[int, ...], ...]

    def __len__(self):
        return len(self.itemsets)

    def __repr__(self):
        return f"<{' → '.join(_format_itemset(s) for s in self.itemsets)}>"


@_dataclass_with_slots()
class Occurrence:
    """Motif/Rule的一次出现"""
    positions: Tuple[int, ...]

    @property
    def start(self):
        return self.positions[0]

    @property
    def end(self):
        return self.positions[-1]

@_dataclass_with_slots()
class Rule:
    """时序规则"""
    antecedent: Motif
    consequent: Motif
    delta_min: int
    delta_max: int

    sup_P: float = 0.0
    sup_rule: float = 0.0
    conf: float = 0.0
    lift: float = 0.0

    sup_P_lambda: float = 0.0
    sup_rule_lambda: float = 0.0
    conf_lambda: float = 0.0
    lift_lambda: float = 0.0

    stability: float = 0.0
    conf_per_regime: List[float] = field(default_factory=list)

    def __repr__(self):
        return f"{self.antecedent} ⇒ {self.consequent} [{self.delta_min},{self.delta_max}] (conf={self.conf:.3f})"


def _occ_start(occ) -> int:
    if isinstance(occ, Occurrence):
        return occ.start
    return occ[0]


def _occ_end(occ) -> int:
    if isinstance(occ, Occurrence):
        return occ.end
    return occ[-1]

# ==================== JS散度（NumPy/Numba实现） ====================

def _js_divergence_np(p: np.ndarray, q: np.ndarray) -> float:
    """NumPy实现的JS散度，避免scipy调用开销"""
    eps = 1e-10
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return 0.5 * (kl_pm + kl_qm)


if NUMBA_AVAILABLE:
    @njit(fastmath=True)
    def _js_divergence_numba(p: np.ndarray, q: np.ndarray) -> float:
        eps = 1e-10
        kl_pm = 0.0
        kl_qm = 0.0
        for i in range(p.shape[0]):
            m = 0.5 * (p[i] + q[i])
            kl_pm += p[i] * np.log((p[i] + eps) / (m + eps))
            kl_qm += q[i] * np.log((q[i] + eps) / (m + eps))
        return 0.5 * (kl_pm + kl_qm)

    _js_divergence = _js_divergence_numba
else:
    _js_divergence = _js_divergence_np



# ==================== 1. Eventizer: 事件化模块 ====================

class Eventizer:
    """将连续多变量序列转换为事件序列"""

    def __init__(self,
                 n_bins: int = 4,
                 window_size: int = 200,
                 quantile_stride: int = 10,
                 change_threshold: float = 0.1,
                 support_decay_lambda: Optional[float] = None,
                 support_min: float = 0.01,
                 support_max: float = 0.95):
        self.n_bins = n_bins
        self.window_size = window_size
        self.quantile_stride = max(1, quantile_stride)
        self.change_threshold = change_threshold
        self.support_decay_lambda = support_decay_lambda
        self.support_min = support_min
        self.support_max = support_max
        _set_event_encoder(self.n_bins)

    def transform(self, X: np.ndarray) -> List[Tuple[int, ...]]:
        """将 X (T, d) 转换为事件序列"""
        T, d = X.shape
        events_sequence: List[Tuple[int, ...]] = []
        last_bins = [None] * d

        for t in range(T):
            E_t = set()
            start = max(0, t - self.window_size + 1)

            for v in range(d):
                window_data = X[start:t + 1, v]
                if len(window_data) < 10:
                    continue
                if (t % self.quantile_stride == 0) or (last_bins[v] is None):
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    bins = np.quantile(window_data, quantiles)
                    bins = np.unique(bins)
                    last_bins[v] = bins

                bins = last_bins[v]
                if bins is not None and len(bins) > 1:
                    bin_idx = np.searchsorted(bins[1:-1], X[t, v])
                    bin_idx = min(bin_idx, self.n_bins - 1)
                    E_t.add(_encode_level_event(v, int(bin_idx)))

            if t > 0:
                for v in range(d):
                    delta = X[t, v] - X[t - 1, v]
                    window_std = np.std(X[start:t + 1, v]) + 1e-8
                    if abs(delta) > self.change_threshold * window_std:
                        E_t.add(_encode_change_event(v, delta > 0))

            events_sequence.append(tuple(sorted(E_t)))

        if T == 0:
            return events_sequence

        if self.support_min is None and self.support_max is None:
            return events_sequence

        min_support = 0.0 if self.support_min is None else self.support_min
        max_support = 1.0 if self.support_max is None else self.support_max
        if min_support <= 0.0 and max_support >= 1.0:
            return events_sequence

        if self.support_decay_lambda is None:
            half_life = max(1.0, T / 5.0)
            decay_lambda = np.log(2) / half_life
        else:
            decay_lambda = self.support_decay_lambda

        ts = np.arange(T)
        weights = np.exp(-decay_lambda * (T - ts))
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            return events_sequence

        event_weights = defaultdict(float)
        for t, E_t in enumerate(events_sequence):
            w = weights[t]
            for e in E_t:
                event_weights[e] += w

        filtered_sequence = []
        for E_t in events_sequence:
            filtered_sequence.append(tuple(
                e for e in E_t
                if min_support <= (event_weights[e] / weights_sum) <= max_support
            ))

        return filtered_sequence


# ==================== 1b. Incremental Eventizer ====================

class IncrementalEventizer(Eventizer):
    """Eventizer with incremental window stats to reduce O(T*d*W) cost."""

    def transform(self, X: np.ndarray) -> List[Tuple[int, ...]]:
        T, d = X.shape
        events_sequence: List[Tuple[int, ...]] = []
        last_bins = [None] * d
        windows = [deque(maxlen=self.window_size) for _ in range(d)]
        running_sum = np.zeros(d, dtype=float)
        running_sum_sq = np.zeros(d, dtype=float)

        for t in range(T):
            E_t = set()

            for v in range(d):
                x_new = float(X[t, v])
                if len(windows[v]) == self.window_size:
                    x_old = windows[v][0]
                    running_sum[v] -= x_old
                    running_sum_sq[v] -= x_old * x_old

                windows[v].append(x_new)
                running_sum[v] += x_new
                running_sum_sq[v] += x_new * x_new

                n = len(windows[v])
                if n >= 2:
                    mean = running_sum[v] / n
                    var = (running_sum_sq[v] / n) - (mean * mean)
                    if var < 0.0:
                        var = 0.0
                    window_std = np.sqrt(var) + 1e-8
                else:
                    window_std = 1e-8

                if n >= 10 and ((t % self.quantile_stride == 0) or (last_bins[v] is None)):
                    window_array = np.array(windows[v], dtype=float)
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    bins = np.quantile(window_array, quantiles)
                    bins = np.unique(bins)
                    last_bins[v] = bins

                bins = last_bins[v]
                if bins is not None and len(bins) > 1:
                    bin_idx = np.searchsorted(bins[1:-1], x_new)
                    if bin_idx >= self.n_bins:
                        bin_idx = self.n_bins - 1
                    E_t.add(_encode_level_event(v, int(bin_idx)))

                if t > 0:
                    delta = x_new - X[t - 1, v]
                    if abs(delta) > self.change_threshold * window_std:
                        E_t.add(_encode_change_event(v, delta > 0))

            events_sequence.append(tuple(sorted(E_t)))

        if T == 0:
            return events_sequence

        if self.support_min is None and self.support_max is None:
            return events_sequence

        min_support = 0.0 if self.support_min is None else self.support_min
        max_support = 1.0 if self.support_max is None else self.support_max
        if min_support <= 0.0 and max_support >= 1.0:
            return events_sequence

        if self.support_decay_lambda is None:
            half_life = max(1.0, T / 5.0)
            decay_lambda = np.log(2) / half_life
        else:
            decay_lambda = self.support_decay_lambda

        ts = np.arange(T)
        weights = np.exp(-decay_lambda * (T - ts))
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            return events_sequence

        event_weights = defaultdict(float)
        for t, E_t in enumerate(events_sequence):
            w = weights[t]
            for e in E_t:
                event_weights[e] += w

        filtered_sequence = []
        for E_t in events_sequence:
            filtered_sequence.append(tuple(
                e for e in E_t
                if min_support <= (event_weights[e] / weights_sum) <= max_support
            ))

        return filtered_sequence

# ==================== 2. Segmenter: 分段模块 ====================

class Segmenter:
    """基于事件分布的JS散度变点检测"""

    def __init__(self, window_size: int = 1000, threshold: float = 0.1, min_gap: int = None):
        self.window_size = window_size
        self.threshold = threshold
        self.min_gap = min_gap or window_size // 2

    def detect_changepoints(self, event_sequence: List[Tuple[int, ...]]) -> List[int]:
        """检测变点"""
        T = len(event_sequence)
        if T <= 2 * self.window_size:
            return [0, T]

        all_events = set()
        for E_t in event_sequence:
            all_events.update(E_t)
        if not all_events:
            return [0, T]

        event_list = sorted(all_events)
        event_to_idx = {e: i for i, e in enumerate(event_list)}
        m = len(event_list)

        eps = 1e-10
        pre_counts = np.full(m, eps, dtype=float)
        post_counts = np.full(m, eps, dtype=float)
        w = self.window_size

        for s in range(0, w):
            for e in event_sequence[s]:
                pre_counts[event_to_idx[e]] += 1
        for s in range(w, 2 * w):
            for e in event_sequence[s]:
                post_counts[event_to_idx[e]] += 1

        js_scores = []
        for t in range(w, T - w):
            pre_sum = float(pre_counts.sum())
            post_sum = float(post_counts.sum())
            pre_dist = pre_counts / pre_sum
            post_dist = post_counts / post_sum
            js = _js_divergence(pre_dist, post_dist)
            js_scores.append((t, js))

            out_pre = t - w
            in_pre = t
            out_post = t
            in_post = t + w

            for e in event_sequence[out_pre]:
                pre_counts[event_to_idx[e]] -= 1
            for e in event_sequence[in_pre]:
                pre_counts[event_to_idx[e]] += 1
            for e in event_sequence[out_post]:
                post_counts[event_to_idx[e]] -= 1
            if in_post < T:
                for e in event_sequence[in_post]:
                    post_counts[event_to_idx[e]] += 1

        changepoints = [0]
        for i in range(1, len(js_scores) - 1):
            t, js = js_scores[i]
            if js > self.threshold:
                if js > js_scores[i - 1][1] and js > js_scores[i + 1][1]:
                    if t - changepoints[-1] >= self.min_gap:
                        changepoints.append(t)

        changepoints.append(T)
        return changepoints

# ==================== 3. IndexBuilder: 索引构建 ====================

class IndexBuilder:
    """构建BitSet和位置列表索引"""

    def __init__(self,
                 event_sequence: List[Tuple[int, ...]],
                 use_bitarray: bool = True,
                 cache_maxsize: int = 10000):
        self.T = len(event_sequence)
        self.event_sequence = event_sequence
        self._use_bitarray = bool(use_bitarray and BITARRAY_AVAILABLE)

        self.bitsets: Dict[int, object] = {}
        self.position_lists = defaultdict(list)
        self._bitset_cache = LRUCache(cache_maxsize)
        self._bitset_numpy_cache = LRUCache(cache_maxsize)
        self._positions_cache = LRUCache(cache_maxsize)

        for t, E_t in enumerate(event_sequence):
            for e in E_t:
                self.position_lists[e].append(t)
        self.position_lists = {
            e: array.array('I', positions)
            for e, positions in self.position_lists.items()
        }

        if self._use_bitarray:
            for e, positions in self.position_lists.items():
                bs = bitarray(self.T)
                bs.setall(False)
                for t in positions:
                    bs[t] = True
                self.bitsets[e] = bs
        else:
            for e, positions in self.position_lists.items():
                arr = np.zeros(self.T, dtype=bool)
                if len(positions) > 0:
                    arr[positions] = True
                self.bitsets[e] = arr

    def _zero_bitset(self):
        if self._use_bitarray:
            bs = bitarray(self.T)
            bs.setall(False)
            return bs
        return np.zeros(self.T, dtype=bool)

    def _one_bitset(self):
        if self._use_bitarray:
            bs = bitarray(self.T)
            bs.setall(True)
            return bs
        return np.ones(self.T, dtype=bool)

    def _bitset_count(self, bs) -> int:
        if self._use_bitarray:
            return bs.count()
        return int(bs.sum())

    def event_support(self, event: int) -> int:
        bs = self.bitsets.get(event)
        if bs is None:
            return 0
        return self._bitset_count(bs)

    def bitset_and_count(self, e1: int, e2: int) -> int:
        b1 = self.bitsets.get(e1)
        b2 = self.bitsets.get(e2)
        if b1 is None or b2 is None:
            return 0
        return self._bitset_count(b1 & b2)

    def get_bitset_numpy(self, itemset: Iterable[int]) -> np.ndarray:
        key = _itemset_key(itemset)
        cached = self._bitset_numpy_cache.get(key)
        if cached is not _MISSING:
            return cached

        bs = self.get_bitset(key)
        if self._use_bitarray:
            arr = np.array(bs, dtype=bool)
        else:
            arr = bs
        self._bitset_numpy_cache.set(key, arr)
        return arr

    def get_bitset(self, itemset: Iterable[int]):
        """获取itemset的bitset（所有事件的AND）"""
        if not itemset:
            return self._one_bitset()

        key = _itemset_key(itemset)
        cached = self._bitset_cache.get(key)
        if cached is not _MISSING:
            return cached

        if len(key) == 1:
            e = key[0]
            result = self.bitsets.get(e, self._zero_bitset())
            self._bitset_cache.set(key, result)
            return result

        result = None
        for e in key:
            if e not in self.bitsets:
                result = self._zero_bitset()
                break
            if result is None:
                result = self.bitsets[e].copy()
            else:
                result &= self.bitsets[e]

        if result is None:
            result = self._zero_bitset()
        self._bitset_cache.set(key, result)
        return result

    def get_positions(self, itemset: Iterable[int]) -> List[int]:
        """获取itemset的位置列表（所有事件的交集）"""
        if not itemset:
            return list(range(self.T))

        key = _itemset_key(itemset)
        cached = self._positions_cache.get(key)
        if cached is not _MISSING:
            return cached

        if len(key) == 1:
            e = key[0]
            out = self.position_lists.get(e, [])
            self._positions_cache.set(key, out)
            return out

        it = iter(key)
        e0 = next(it)
        out = self.position_lists.get(e0, [])
        if len(out) == 0:
            self._positions_cache.set(key, [])
            return []
        for e in it:
            pe = self.position_lists.get(e, [])
            if len(pe) == 0:
                self._positions_cache.set(key, [])
                return []
            out = _intersect_sorted_lists(out, pe)
            if len(out) == 0:
                self._positions_cache.set(key, [])
                return []

        self._positions_cache.set(key, out)
        return out

# ==================== Phase B并行辅助函数 ====================

_OCC_POSITIONS = None
_OCC_GAP_MIN = 0
_OCC_GAP_MAX = 0
_OCC_MIN_SUPPORT = 1
_OCC_POS_CACHE = None
_OCC_SHM_HANDLES = None
_OCC_DELAY_MATERIALIZE = False

_RULE_P_BITSETS = None
_RULE_Q_WINDOWS = None
_RULE_OCC_MAP = None
_RULE_SUP_P = None
_RULE_CONSEQUENTS = None
_RULE_DELTA_MIN = 0
_RULE_DELTA_MAX = 0
_RULE_LAMBDA = 0.0
_RULE_REGIMES = None
_RULE_REGIME_OF_T = None
_RULE_T = 0
_RULE_MIN_CONF = 0.0
_RULE_MIN_RULE_SUPPORT = None
_RULE_TOP_K = None

_RULE_L_P_MOTIFS = None
_RULE_L_P_ENDS = None
_RULE_L_P_WP = None
_RULE_L_P_R = None
_RULE_L_P_REG_COUNTS = None
_RULE_L_P_SUPP_DECAY = None
_RULE_L_P_BITSETS = None
_RULE_L_Q_MOTIFS = None
_RULE_L_Q_WINDOWS = None
_RULE_L_Q_COUNTS = None
_RULE_L_Q_PRIOR = None
_RULE_L_RN = 0
_RULE_L_Q_SHM = None
_RULE_L_Q_SHM_NAME = None
_RULE_L_Q_SHM_SHAPE = None
_RULE_L_Q_SHM_DTYPE = None
_RULE_L_Q_PACKED = False
_RULE_BA_ONE = bitarray("1") if BITARRAY_AVAILABLE else None


def _can_use_process_pool() -> bool:
    if __name__ == "__main__":
        return True
    try:
        import importlib.util
        spec = importlib.util.find_spec(__name__)
        if spec is None or not spec.origin:
            return False
        return os.path.splitext(os.path.basename(spec.origin))[0] == __name__
    except Exception:
        return False


def _intersect_sorted_lists(a: List[int], b: List[int]) -> List[int]:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return []
    if la + lb >= _INTERSECT_NUMPY_THRESHOLD:
        try:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            return np.intersect1d(a_arr, b_arr, assume_unique=True).tolist()
        except Exception:
            pass
    i = j = 0
    out = []
    while i < la and j < lb:
        va, vb = a[i], b[j]
        if va == vb:
            out.append(va)
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    return out


def _positions_for_itemset(itemset: Iterable[int],
                           pos_map: Dict[int, List[int]]) -> List[int]:
    global _OCC_POS_CACHE
    if not itemset:
        return []
    key = _itemset_key(itemset)
    if _OCC_POS_CACHE is not None:
        cached = _OCC_POS_CACHE.get(key)
        if cached is not None:
            return cached
    if len(key) == 1:
        e = key[0]
        out = pos_map.get(e, [])
        if _OCC_POS_CACHE is not None:
            _OCC_POS_CACHE[key] = out
        return out
    it = iter(key)
    e0 = next(it)
    out = pos_map.get(e0, [])
    if len(out) == 0:
        if _OCC_POS_CACHE is not None:
            _OCC_POS_CACHE[key] = []
        return []
    for e in it:
        pe = pos_map.get(e, [])
        if len(pe) == 0:
            if _OCC_POS_CACHE is not None:
                _OCC_POS_CACHE[key] = []
            return []
        out = _intersect_sorted_lists(out, pe)
        if len(out) == 0:
            if _OCC_POS_CACHE is not None:
                _OCC_POS_CACHE[key] = []
            return []
    if _OCC_POS_CACHE is not None:
        _OCC_POS_CACHE[key] = out
    return out


def _init_occurrence_worker(position_lists, gap_min, gap_max, min_support_count, delay_materialize=False):
    global _OCC_POSITIONS, _OCC_GAP_MIN, _OCC_GAP_MAX, _OCC_MIN_SUPPORT, _OCC_POS_CACHE
    global _OCC_DELAY_MATERIALIZE
    _OCC_POSITIONS = position_lists
    _OCC_GAP_MIN = gap_min
    _OCC_GAP_MAX = gap_max
    _OCC_MIN_SUPPORT = max(1, int(min_support_count or 1))
    _OCC_POS_CACHE = {}
    _OCC_DELAY_MATERIALIZE = bool(delay_materialize)


def _init_occurrence_worker_shm(shm_metadata, gap_min, gap_max, min_support_count, delay_materialize=False):
    global _OCC_POSITIONS, _OCC_GAP_MIN, _OCC_GAP_MAX, _OCC_MIN_SUPPORT, _OCC_POS_CACHE
    global _OCC_SHM_HANDLES, _OCC_DELAY_MATERIALIZE
    from multiprocessing import shared_memory
    _OCC_GAP_MIN = gap_min
    _OCC_GAP_MAX = gap_max
    _OCC_MIN_SUPPORT = max(1, int(min_support_count or 1))
    _OCC_POS_CACHE = {}
    _OCC_DELAY_MATERIALIZE = bool(delay_materialize)
    _OCC_SHM_HANDLES = []
    pos_map = {}
    for event_id, meta in shm_metadata.items():
        name, shape, dtype_str = meta
        shm = shared_memory.SharedMemory(name=name)
        _OCC_SHM_HANDLES.append(shm)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        pos_map[event_id] = arr
    _OCC_POSITIONS = pos_map


def _find_occurrences_from_positions(motif: Motif,
                                     position_lists_map: Dict[int, List[int]],
                                     gap_min: int,
                                     gap_max: int) -> List[Occurrence]:
    global _OCC_DELAY_MATERIALIZE
    position_lists = []
    for itemset in motif.itemsets:
        positions = _positions_for_itemset(itemset, position_lists_map)
        if len(positions) == 0:
            return []
        position_lists.append(positions)

    if len(position_lists) == 1:
        L0 = position_lists[0]
        if _OCC_DELAY_MATERIALIZE:
            return [(int(p),) for p in L0]
        return [Occurrence((int(p),)) for p in L0]

    mins = [pl[0] for pl in position_lists]
    maxs = [pl[-1] for pl in position_lists]
    for i in range(len(position_lists) - 1):
        for j in range(i + 1, len(position_lists)):
            delta = j - i
            if mins[j] > maxs[i] + delta * gap_max:
                return []
            if maxs[j] < mins[i] + delta * gap_min:
                return []

    if len(position_lists) == 2:
        L0, L1 = position_lists
        occurrences = []
        cursor0 = bisect_left(L0, 0, 0)
        cursor1 = 0
        min_support_count = _OCC_MIN_SUPPORT
        next_start_min = 0
        while cursor0 < len(L0):
            if min_support_count > 1:
                remaining = len(L0) - cursor0
                if len(occurrences) + remaining < min_support_count:
                    break
            t0 = L0[cursor0]
            lo = t0 + gap_min
            hi = t0 + gap_max
            cursor1 = bisect_left(L1, lo, cursor1)
            if cursor1 < len(L1) and L1[cursor1] <= hi:
                t1 = L1[cursor1]
                occurrences.append((int(t0), int(t1)))
                next_start_min = t1 + 1
                cursor0 = bisect_left(L0, next_start_min, cursor0)
            else:
                cursor0 += 1
        if _OCC_DELAY_MATERIALIZE:
            return occurrences
        return [Occurrence(occ) for occ in occurrences]

    occurrences = []
    cursors = [0] * len(position_lists)
    next_start_min = 0
    L0 = position_lists[0]

    cursors[0] = bisect_left(L0, next_start_min, cursors[0])

    min_support_count = _OCC_MIN_SUPPORT
    while cursors[0] < len(L0):
        if min_support_count > 1:
            remaining = len(L0) - cursors[0]
            if len(occurrences) + remaining < min_support_count:
                break
        chain = [L0[cursors[0]]]
        valid = True

        for j in range(1, len(position_lists)):
            Lj = position_lists[j]
            lo = chain[-1] + gap_min
            hi = chain[-1] + gap_max

            cursors[j] = bisect_left(Lj, lo, cursors[j])
            if cursors[j] >= len(Lj) or Lj[cursors[j]] > hi:
                valid = False
                break

            chain.append(Lj[cursors[j]])

        if valid:
            occurrences.append(tuple(chain))
            next_start_min = chain[-1] + 1
            cursors[0] = bisect_left(L0, next_start_min, cursors[0])
        else:
            cursors[0] += 1

    if _OCC_DELAY_MATERIALIZE:
        return occurrences
    return [Occurrence(occ) for occ in occurrences]


def _find_occurrences_worker(motif: Motif):
    return motif, _find_occurrences_from_positions(
        motif, _OCC_POSITIONS, _OCC_GAP_MIN, _OCC_GAP_MAX
    )


def _packed_window_mask(packed: np.ndarray, ends: np.ndarray) -> np.ndarray:
    if ends.size == 0:
        return np.zeros(0, dtype=bool)
    byte_idx = ends >> 3
    bit_idx = ends & 7
    bytes_sel = packed[byte_idx]
    return ((bytes_sel >> (7 - bit_idx)) & 1).astype(bool)

def _init_rule_worker(p_bitsets,
                      q_windows,
                      occ_map,
                      sup_p_map,
                      consequents,
                      delta_min,
                      delta_max,
                      lambda_decay,
                      regimes,
                      regime_of_t,
                      total_t,
                      min_conf_per_regime,
                      min_rule_support_count,
                      top_k):
    global _RULE_P_BITSETS, _RULE_Q_WINDOWS, _RULE_OCC_MAP, _RULE_SUP_P, _RULE_CONSEQUENTS
    global _RULE_DELTA_MIN, _RULE_DELTA_MAX, _RULE_LAMBDA, _RULE_REGIMES, _RULE_REGIME_OF_T
    global _RULE_T, _RULE_MIN_CONF, _RULE_MIN_RULE_SUPPORT, _RULE_TOP_K
    _RULE_P_BITSETS = p_bitsets
    _RULE_Q_WINDOWS = q_windows
    _RULE_OCC_MAP = occ_map
    _RULE_SUP_P = sup_p_map
    _RULE_CONSEQUENTS = consequents
    _RULE_DELTA_MIN = delta_min
    _RULE_DELTA_MAX = delta_max
    _RULE_LAMBDA = lambda_decay
    _RULE_REGIMES = regimes
    _RULE_REGIME_OF_T = regime_of_t
    _RULE_T = total_t
    _RULE_MIN_CONF = min_conf_per_regime
    _RULE_MIN_RULE_SUPPORT = min_rule_support_count
    _RULE_TOP_K = top_k


def _init_rule_light_worker(p_motifs,
                            p_ends,
                            p_wp,
                            p_r,
                            p_reg_counts,
                            p_sup_decay,
                            p_bitsets,
                            q_motifs,
                            q_windows,
                            q_counts,
                            q_prior,
                            delta_min,
                            delta_max,
                            lambda_decay,
                            regimes,
                            regime_of_t,
                            total_t,
                            min_conf_per_regime,
                            min_rule_support_count,
                            top_k,
                            q_packed,
                            q_shm_name=None,
                            q_shm_shape=None,
                            q_shm_dtype=None):
    global _RULE_L_P_MOTIFS, _RULE_L_P_ENDS, _RULE_L_P_WP, _RULE_L_P_R
    global _RULE_L_P_REG_COUNTS, _RULE_L_P_SUPP_DECAY, _RULE_L_P_BITSETS
    global _RULE_L_Q_MOTIFS
    global _RULE_L_Q_WINDOWS, _RULE_L_Q_COUNTS, _RULE_L_Q_PRIOR, _RULE_L_RN
    global _RULE_L_Q_SHM, _RULE_L_Q_SHM_NAME, _RULE_L_Q_SHM_SHAPE, _RULE_L_Q_SHM_DTYPE
    global _RULE_DELTA_MIN, _RULE_DELTA_MAX, _RULE_LAMBDA, _RULE_REGIMES
    global _RULE_REGIME_OF_T, _RULE_T, _RULE_MIN_CONF, _RULE_MIN_RULE_SUPPORT, _RULE_TOP_K
    global _RULE_L_Q_PACKED
    _RULE_L_P_MOTIFS = p_motifs
    _RULE_L_P_ENDS = p_ends
    _RULE_L_P_WP = p_wp
    _RULE_L_P_R = p_r
    _RULE_L_P_REG_COUNTS = p_reg_counts
    _RULE_L_P_SUPP_DECAY = p_sup_decay
    _RULE_L_P_BITSETS = p_bitsets
    _RULE_L_Q_MOTIFS = q_motifs
    _RULE_L_Q_SHM = None
    _RULE_L_Q_SHM_NAME = None
    _RULE_L_Q_SHM_SHAPE = None
    _RULE_L_Q_SHM_DTYPE = None
    _RULE_L_Q_PACKED = bool(q_packed)
    if q_windows is None and q_shm_name is not None and q_shm_shape is not None and q_shm_dtype is not None:
        try:
            from multiprocessing import shared_memory
            _RULE_L_Q_SHM = shared_memory.SharedMemory(name=q_shm_name)
            _RULE_L_Q_SHM_NAME = q_shm_name
            _RULE_L_Q_SHM_SHAPE = tuple(q_shm_shape)
            _RULE_L_Q_SHM_DTYPE = q_shm_dtype
            _RULE_L_Q_WINDOWS = np.ndarray(
                _RULE_L_Q_SHM_SHAPE,
                dtype=np.dtype(_RULE_L_Q_SHM_DTYPE),
                buffer=_RULE_L_Q_SHM.buf
            )
        except Exception:
            _RULE_L_Q_WINDOWS = q_windows
    else:
        _RULE_L_Q_WINDOWS = q_windows
    _RULE_L_Q_COUNTS = q_counts
    _RULE_L_Q_PRIOR = q_prior
    _RULE_L_RN = len(regimes) - 1
    _RULE_DELTA_MIN = delta_min
    _RULE_DELTA_MAX = delta_max
    _RULE_LAMBDA = lambda_decay
    _RULE_REGIMES = regimes
    _RULE_REGIME_OF_T = regime_of_t
    _RULE_T = total_t
    _RULE_MIN_CONF = min_conf_per_regime
    _RULE_MIN_RULE_SUPPORT = min_rule_support_count
    _RULE_TOP_K = top_k


def _rule_light_worker(p_idx: int):
    P = _RULE_L_P_MOTIFS[p_idx]
    ends = _RULE_L_P_ENDS[p_idx]
    sup_P = int(ends.size)
    if _RULE_MIN_RULE_SUPPORT is not None and sup_P < _RULE_MIN_RULE_SUPPORT:
        return []

    wP = _RULE_L_P_WP[p_idx]
    r = _RULE_L_P_R[p_idx]
    reg_counts = _RULE_L_P_REG_COUNTS[p_idx]
    supP_decay = _RULE_L_P_SUPP_DECAY[p_idx]

    if _RULE_TOP_K is not None:
        heap = []
        min_keep = None
    else:
        heap = None
        min_keep = None

    results = []
    for q_idx, Q in enumerate(_RULE_L_Q_MOTIFS):
        if P == Q:
            continue
        q_count = _RULE_L_Q_COUNTS[q_idx]
        upper = sup_P if sup_P < q_count else q_count
        if _RULE_MIN_RULE_SUPPORT is not None and upper < _RULE_MIN_RULE_SUPPORT:
            continue
        if heap is not None and len(heap) >= _RULE_TOP_K:
            if min_keep is None:
                min_keep = heap[0][0]
            if upper <= min_keep:
                continue

        window = _RULE_L_Q_WINDOWS[q_idx]
        if _RULE_L_Q_PACKED:
            mask = _packed_window_mask(window, ends)
            sup_rule = int(mask.sum())
            if sup_rule <= 0:
                continue
            if _RULE_MIN_RULE_SUPPORT is not None and sup_rule < _RULE_MIN_RULE_SUPPORT:
                continue
            if heap is not None and len(heap) >= _RULE_TOP_K:
                if min_keep is None:
                    min_keep = heap[0][0]
                if sup_rule <= min_keep:
                    continue

            matches_decay = float(wP[mask].sum()) if supP_decay > 0 else 0.0
            if _RULE_L_RN > 0 and mask.any():
                r_mask = r[mask]
                r_mask = r_mask[r_mask >= 0]
                if r_mask.size > 0:
                    regime_matches = np.bincount(r_mask, minlength=_RULE_L_RN)
                else:
                    regime_matches = np.zeros(_RULE_L_RN, dtype=int)
            else:
                regime_matches = np.zeros(_RULE_L_RN, dtype=int)
        elif _RULE_L_P_BITSETS is not None and BITARRAY_AVAILABLE and isinstance(window, bitarray):
            mask_bits = window & _RULE_L_P_BITSETS[p_idx]
            sup_rule = mask_bits.count()
            if sup_rule <= 0:
                continue
            if _RULE_MIN_RULE_SUPPORT is not None and sup_rule < _RULE_MIN_RULE_SUPPORT:
                continue
            if heap is not None and len(heap) >= _RULE_TOP_K:
                if min_keep is None:
                    min_keep = heap[0][0]
                if sup_rule <= min_keep:
                    continue

            matches_decay = 0.0
            regime_matches = np.zeros(_RULE_L_RN, dtype=int)
            if (supP_decay > 0 or _RULE_L_RN > 0) and sup_rule > 0:
                match_positions = None
                if _RULE_BA_ONE is not None:
                    try:
                        match_positions = mask_bits.search(_RULE_BA_ONE)
                    except Exception:
                        match_positions = None
                if match_positions is not None:
                    match_positions = np.asarray(match_positions, dtype=int)
                    if match_positions.size > 0:
                        chunk_size = 4096
                        for start in range(0, match_positions.size, chunk_size):
                            chunk = match_positions[start:start + chunk_size]
                            idxs = np.searchsorted(ends, chunk)
                            valid = idxs < ends.size
                            if not valid.any():
                                continue
                            chunk = chunk[valid]
                            idxs = idxs[valid]
                            valid = ends[idxs] == chunk
                            if not valid.any():
                                continue
                            idxs = idxs[valid]
                            if supP_decay > 0:
                                matches_decay += float(wP[idxs].sum())
                            if _RULE_L_RN > 0:
                                r_mask = r[idxs]
                                r_mask = r_mask[r_mask >= 0]
                                if r_mask.size > 0:
                                    regime_matches += np.bincount(r_mask, minlength=_RULE_L_RN)
                else:
                    for idx, t in enumerate(ends):
                        if window[t]:
                            if supP_decay > 0:
                                matches_decay += wP[idx]
                            if _RULE_L_RN > 0:
                                ri = r[idx]
                                if ri >= 0:
                                    regime_matches[ri] += 1
        else:
            mask = window[ends] > 0
            sup_rule = int(mask.sum())
            if sup_rule <= 0:
                continue
            if _RULE_MIN_RULE_SUPPORT is not None and sup_rule < _RULE_MIN_RULE_SUPPORT:
                continue
            if heap is not None and len(heap) >= _RULE_TOP_K:
                if min_keep is None:
                    min_keep = heap[0][0]
                if sup_rule <= min_keep:
                    continue

            matches_decay = float(wP[mask].sum()) if supP_decay > 0 else 0.0
            if _RULE_L_RN > 0 and mask.any():
                r_mask = r[mask]
                r_mask = r_mask[r_mask >= 0]
                if r_mask.size > 0:
                    regime_matches = np.bincount(r_mask, minlength=_RULE_L_RN)
                else:
                    regime_matches = np.zeros(_RULE_L_RN, dtype=int)
            else:
                regime_matches = np.zeros(_RULE_L_RN, dtype=int)

        rule = Rule(P, Q, _RULE_DELTA_MIN, _RULE_DELTA_MAX)
        rule.sup_P = sup_P
        rule.sup_rule = sup_rule

        prior_win = _RULE_L_Q_PRIOR[q_idx]
        rule.conf = sup_rule / sup_P if sup_P > 0 else 0.0
        rule.lift = (rule.conf / prior_win) if prior_win > 0 else 0.0

        rule.sup_P_lambda = supP_decay
        rule.sup_rule_lambda = matches_decay
        rule.conf_lambda = matches_decay / supP_decay if supP_decay > 0 else 0.0
        rule.lift_lambda = (rule.conf_lambda / prior_win) if prior_win > 0 else 0.0

        rule.conf_per_regime = []
        stable = 0
        for ri in range(_RULE_L_RN):
            denom = reg_counts[ri] if ri < len(reg_counts) else 0
            if denom > 0:
                cr = regime_matches[ri] / denom
                rule.conf_per_regime.append(cr)
                if cr >= _RULE_MIN_CONF:
                    stable += 1
            else:
                rule.conf_per_regime.append(0.0)
        rule.stability = stable / _RULE_L_RN if _RULE_L_RN > 0 else 0.0

        if heap is None:
            results.append((sup_rule, rule))
        else:
            heapq.heappush(heap, (sup_rule, rule))
            if len(heap) > _RULE_TOP_K:
                heapq.heappop(heap)
            min_keep = heap[0][0]

    if heap is not None:
        results = list(heap)
    return results


def _rule_bitset_count(bs) -> int:
    if BITARRAY_AVAILABLE:
        return bs.count()
    return int(bs.sum())


def _compute_rule_stats_window_worker(P: Motif, Q: Motif, q_window: Dict, sup_rule: int) -> Rule:
    P_occs = _RULE_OCC_MAP.get(P, [])
    rule = Rule(P, Q, _RULE_DELTA_MIN, _RULE_DELTA_MAX)
    rule.sup_P = len(P_occs)
    rule.sup_rule = sup_rule

    prior_win = q_window["prior"]
    rule.conf = sup_rule / rule.sup_P if rule.sup_P > 0 else 0.0
    rule.lift = (rule.conf / prior_win) if prior_win > 0 else 0.0

    Rn = len(_RULE_REGIMES) - 1
    regime_matches = [0] * Rn
    regime_P_counts = [0] * Rn
    window = q_window.get("window")
    window_bs = q_window["bitset"] if window is None else None

    matches_decay = 0.0
    supP_decay = 0.0

    P_ends = sorted(_occ_end(occ) for occ in P_occs)
    for tP in P_ends:
        if tP < 0 or tP >= _RULE_T:
            continue
        wP = np.exp(-_RULE_LAMBDA * (_RULE_T - tP))
        supP_decay += wP

        r = _RULE_REGIME_OF_T[tP]
        if r >= 0:
            regime_P_counts[r] += 1

        if window is not None:
            if window[tP]:
                matches_decay += wP
                if r >= 0:
                    regime_matches[r] += 1
        else:
            if window_bs[tP]:
                matches_decay += wP
                if r >= 0:
                    regime_matches[r] += 1

    rule.sup_P_lambda = supP_decay
    rule.sup_rule_lambda = matches_decay
    rule.conf_lambda = matches_decay / supP_decay if supP_decay > 0 else 0.0
    rule.lift_lambda = (rule.conf_lambda / prior_win) if prior_win > 0 else 0.0

    rule.conf_per_regime = []
    stable = 0
    for r in range(Rn):
        if regime_P_counts[r] > 0:
            cr = regime_matches[r] / regime_P_counts[r]
            rule.conf_per_regime.append(cr)
            if cr >= _RULE_MIN_CONF:
                stable += 1
        else:
            rule.conf_per_regime.append(0.0)
    rule.stability = stable / Rn if Rn > 0 else 0.0
    return rule


def _rule_worker(P: Motif):
    p_bs = _RULE_P_BITSETS.get(P)
    if p_bs is None:
        return []
    sup_P = _RULE_SUP_P.get(P, 0)
    if _RULE_MIN_RULE_SUPPORT is not None and sup_P < _RULE_MIN_RULE_SUPPORT:
        return []

    results = []
    if _RULE_TOP_K is not None:
        heap = []
        min_keep = None
    else:
        heap = None
        min_keep = None

    for Q in _RULE_CONSEQUENTS:
        if P == Q:
            continue
        q_win = _RULE_Q_WINDOWS.get(Q)
        if q_win is None:
            continue
        upper = sup_P if sup_P < q_win["count"] else q_win["count"]
        if _RULE_MIN_RULE_SUPPORT is not None and upper < _RULE_MIN_RULE_SUPPORT:
            continue
        if heap is not None and len(heap) >= _RULE_TOP_K:
            if min_keep is None:
                min_keep = heap[0][0]
            if upper <= min_keep:
                continue
        sup_rule = _rule_bitset_count(p_bs & q_win["bitset"])
        if sup_rule <= 0:
            continue
        if _RULE_MIN_RULE_SUPPORT is not None and sup_rule < _RULE_MIN_RULE_SUPPORT:
            continue
        if heap is not None and len(heap) >= _RULE_TOP_K:
            if min_keep is None:
                min_keep = heap[0][0]
            if sup_rule <= min_keep:
                continue
        rule = _compute_rule_stats_window_worker(P, Q, q_win, sup_rule)
        if rule.sup_rule <= 0:
            continue
        if heap is None:
            results.append((sup_rule, rule))
        else:
            heapq.heappush(heap, (sup_rule, rule))
            if len(heap) > _RULE_TOP_K:
                heapq.heappop(heap)
            min_keep = heap[0][0]

    if heap is not None:
        results = list(heap)
    return results

# ==================== 4. MotifMiner: Motif挖掘 ====================

class MotifMiner:
    """两阶段Motif挖掘"""

    def __init__(self,
                 max_length: int = 6,
                 max_itemset_size: int = 1,
                 gap: int = 3,
                 gap_min: int = 0,
                 gap_max: Optional[int] = None,
                 top_M0: int = 200,
                 n_jobs: int = 1,
                 top_candidates: int = 2000,
                 beam_width: Optional[int] = None,
                 min_support_ratio: Optional[float] = None,
                 min_support_count: Optional[int] = None,
                 time_budget_s: Optional[float] = None,
                 use_incremental_support: bool = True,
                 verify_top_k: Optional[int] = None,
                 max_occurrences: Optional[int] = None,
                 bitarray_gap_threshold: Optional[int] = 8,
                 delay_occurrence_materialization: bool = False,
                 next_pos_top_n: Optional[int] = 200,
                 use_shm_positions: bool = False):
        self.max_length = max_length
        self.max_itemset_size = max_itemset_size
        self.gap_min = max(0, gap_min)
        self.gap_max = gap if gap_max is None else gap_max
        if self.gap_max < self.gap_min:
            self.gap_max = self.gap_min
        self.top_M0 = top_M0
        self.n_jobs = max(1, n_jobs)
        self.top_candidates = top_candidates
        self.beam_width = beam_width
        self.min_support_ratio = min_support_ratio
        self.min_support_count = min_support_count
        self.time_budget_s = time_budget_s
        self.use_incremental_support = bool(use_incremental_support)
        self.verify_top_k = verify_top_k
        self.max_occurrences = max_occurrences
        self.bitarray_gap_threshold = bitarray_gap_threshold
        self.delay_occurrence_materialization = bool(delay_occurrence_materialization)
        self.next_pos_top_n = next_pos_top_n
        self.use_shm_positions = bool(use_shm_positions)
        self._use_occurrence_pool = (self.n_jobs == 1) and (not self.delay_occurrence_materialization)
        self._occurrence_pool = None
        self._phase_a_timeout = False
        self._phase_b_timeout = False
        self._candidate_support = {}
        self._min_support_count = 1
        self.last_phase_times: Dict[str, float] = {}
        self._next_pos = None
        self._T = 0

    def mine(self, index: IndexBuilder) -> Tuple[List[Motif], Dict[Motif, List[Occurrence]]]:
        print("开始Motif挖掘...")
        self._T = index.T

        t0 = time.perf_counter()
        candidates = self._phase_A_search(index)
        t1 = time.perf_counter()
        print(f"阶段A: 生成{len(candidates)}个候选motifs")

        motifs, occurrences = self._phase_B_verify(candidates, index)
        t2 = time.perf_counter()
        print(f"阶段B: 验证得到{len(motifs)}个motifs")

        self.last_phase_times = {
            "MotifA": t1 - t0,
            "MotifB": t2 - t1,
        }
        return motifs, occurrences

    def _should_use_bitarray_backend(self, index: IndexBuilder) -> bool:
        if self.bitarray_gap_threshold is None:
            return False
        if not BITARRAY_AVAILABLE or not index._use_bitarray:
            return False
        return self.gap_max <= self.bitarray_gap_threshold

    def _phase_A_search(self, index: IndexBuilder) -> List[Motif]:
        self._phase_a_timeout = False
        t_start = None
        if self.time_budget_s is not None:
            t_start = time.perf_counter()

        def out_of_time() -> bool:
            return t_start is not None and (time.perf_counter() - t_start) > self.time_budget_s

        min_support_count = self.min_support_count
        if min_support_count is None and self.min_support_ratio is not None:
            min_support_count = int(np.ceil(self.min_support_ratio * index.T))
        if min_support_count is None:
            min_support_count = 1
        self._min_support_count = min_support_count

        use_bitarray_backend = self._should_use_bitarray_backend(index)

        single_events = []
        for e in index.bitsets:
            sup_over = index.event_support(e)
            if sup_over >= min_support_count:
                single_events.append((e, sup_over))

        single_events.sort(key=lambda x: x[1], reverse=True)
        frequent_events = [e for e, _ in single_events[:self.top_M0]]

        print(f"  保留{len(frequent_events)}个频繁事件")

        candidates_heap = []
        min_keep = 0
        counter = 0
        use_incremental = bool(self.use_incremental_support)
        beam_width = self.beam_width if (self.beam_width is not None and self.beam_width > 0) else None
        event_support = {e: sup for e, sup in single_events}
        itemset_support_cache = {}

        def itemset_support_upper(itemset: Tuple[int, ...]) -> int:
            cached = itemset_support_cache.get(itemset)
            if cached is not None:
                return cached
            if len(itemset) == 1:
                sup = event_support.get(itemset[0], 0)
            elif len(itemset) == 2:
                sup = min(event_support.get(itemset[0], 0), event_support.get(itemset[1], 0))
            else:
                sup = 0
            itemset_support_cache[itemset] = sup
            return sup

        def keep_candidate(motif: Motif, sup_over: int) -> bool:
            nonlocal min_keep
            nonlocal counter
            if sup_over < min_support_count:
                return False
            if len(candidates_heap) < self.top_candidates:
                heapq.heappush(candidates_heap, (sup_over, counter, motif))
                counter += 1
                min_keep = candidates_heap[0][0]
                return True
            if sup_over > candidates_heap[0][0]:
                heapq.heapreplace(candidates_heap, (sup_over, counter, motif))
                counter += 1
                min_keep = candidates_heap[0][0]
                return True
            return False

        if beam_width is not None:
            beam = []
            for e in frequent_events:
                motif = Motif(((e,),))
                if use_incremental:
                    if use_bitarray_backend:
                        parent_S = index.get_bitset((e,))
                        sup_over = parent_S.count()
                    else:
                        parent_S = index.get_bitset_numpy((e,))
                        sup_over = int(parent_S.sum())
                else:
                    parent_S = None
                    sup_over = index.event_support(e)
                if keep_candidate(motif, sup_over):
                    beam.append((sup_over, motif, parent_S))

            for _ in range(2, self.max_length + 1):
                if out_of_time():
                    self._phase_a_timeout = True
                    break
                beam.sort(key=lambda x: x[0], reverse=True)
                if len(beam) > beam_width:
                    beam = beam[:beam_width]

                next_beam = []
                for parent_sup, motif, parent_S in beam:
                    if out_of_time():
                        self._phase_a_timeout = True
                        break
                    if len(candidates_heap) >= self.top_candidates and parent_sup <= min_keep:
                        continue
                    extensions = self._generate_extensions(motif, frequent_events)

                    for ext_motif in extensions:
                        last_itemset = ext_motif.itemsets[-1]
                        upper = min(parent_sup, itemset_support_upper(last_itemset))
                        if upper < min_support_count:
                            continue
                        if len(candidates_heap) >= self.top_candidates and upper < min_keep:
                            continue
                        if use_incremental and parent_S is not None:
                            child_S = self._extend_bitset(parent_S, last_itemset, index, use_bitarray_backend)
                            ext_sup = child_S.count() if use_bitarray_backend else int(child_S.sum())
                        else:
                            child_S = None
                            ext_sup = self._fast_support(ext_motif, index, use_bitarray_backend)

                        if ext_sup < min_support_count:
                            continue
                        if len(candidates_heap) >= self.top_candidates and ext_sup < min_keep:
                            continue
                        if keep_candidate(ext_motif, ext_sup):
                            next_beam.append((ext_sup, ext_motif, child_S))

                if self._phase_a_timeout or not next_beam:
                    break
                beam = next_beam
        else:
            queue = deque()
            for e in frequent_events:
                motif = Motif(((e,),))
                if use_incremental:
                    if use_bitarray_backend:
                        parent_S = index.get_bitset((e,))
                        sup_over = parent_S.count()
                    else:
                        parent_S = index.get_bitset_numpy((e,))
                        sup_over = int(parent_S.sum())
                else:
                    parent_S = None
                    sup_over = index.event_support(e)
                if keep_candidate(motif, sup_over):
                    queue.append((motif, sup_over, parent_S))

            while queue:
                if out_of_time():
                    self._phase_a_timeout = True
                    break
                motif, sup_over, parent_S = queue.popleft()

                if len(candidates_heap) >= self.top_candidates and sup_over <= min_keep:
                    continue
                if len(motif) >= self.max_length:
                    continue

                extensions = self._generate_extensions(motif, frequent_events)

                for ext_motif in extensions:
                    last_itemset = ext_motif.itemsets[-1]
                    upper = min(sup_over, itemset_support_upper(last_itemset))
                    if upper < min_support_count:
                        continue
                    if len(candidates_heap) >= self.top_candidates and upper < min_keep:
                        continue
                    if use_incremental and parent_S is not None:
                        child_S = self._extend_bitset(parent_S, last_itemset, index, use_bitarray_backend)
                        ext_sup = child_S.count() if use_bitarray_backend else int(child_S.sum())
                    else:
                        child_S = None
                        ext_sup = self._fast_support(ext_motif, index, use_bitarray_backend)

                    if ext_sup < min_support_count:
                        continue
                    if len(candidates_heap) >= self.top_candidates and ext_sup < min_keep:
                        continue
                    if keep_candidate(ext_motif, ext_sup):
                        queue.append((ext_motif, ext_sup, child_S))

        sorted_candidates = sorted(candidates_heap, key=lambda x: x[0], reverse=True)
        candidates = [m for s, _, m in sorted_candidates]
        self._candidate_support = {m: int(s) for s, _, m in sorted_candidates}

        return candidates

    def _generate_extensions(self, motif: Motif, frequent_events: List[int]) -> List[Motif]:
        extensions = []

        for e in frequent_events:
            new_motif = Motif(motif.itemsets + ((e,),))
            extensions.append(new_motif)

        if self.max_itemset_size >= 2:
            for i, e1 in enumerate(frequent_events):
                for e2 in frequent_events[i + 1:]:
                    itemset = (e1, e2) if e1 < e2 else (e2, e1)
                    new_motif = Motif(motif.itemsets + (itemset,))
                    extensions.append(new_motif)

        return extensions

    @staticmethod
    def _dilate_any_within_gap_numpy(S: np.ndarray, gap_min: int, gap_max: int) -> np.ndarray:
        if gap_max < 0:
            return np.zeros_like(S, dtype=bool)

        gap_min = max(0, gap_min)
        gap_max = max(gap_min, gap_max)
        if gap_max == 0 and gap_min == 0:
            return S.copy()

        x = S.astype(np.int32)
        pref = np.cumsum(x)
        counts = np.zeros_like(pref)

        if gap_min == 0:
            counts[:] = pref
        elif gap_min < len(pref):
            counts[gap_min:] = pref[:-gap_min]

        if gap_max + 1 < len(pref):
            counts[gap_max + 1:] -= pref[:-(gap_max + 1)]

        return counts > 0

    @staticmethod
    def _dilate_any_within_gap_bitarray(S, gap_min: int, gap_max: int):
        if gap_max < 0:
            out = bitarray(len(S))
            out.setall(False)
            return out

        gap_min = max(0, gap_min)
        gap_max = max(gap_min, gap_max)
        if gap_max == 0 and gap_min == 0:
            return S.copy()

        out = bitarray(len(S))
        out.setall(False)
        for g in range(gap_min, gap_max + 1):
            out |= (S << g)
        return out

    def _extend_bitset(self, parent_S, next_itemset: Tuple[int, ...], index: IndexBuilder, use_bitarray_backend: bool):
        if use_bitarray_backend:
            Prev = self._dilate_any_within_gap_bitarray(parent_S, self.gap_min, self.gap_max)
            Curr_pos = index.get_bitset(next_itemset)
            return Prev & Curr_pos
        Prev = self._dilate_any_within_gap_numpy(parent_S, self.gap_min, self.gap_max)
        Curr_pos = index.get_bitset_numpy(next_itemset)
        return Prev & Curr_pos

    def _fast_support(self, motif: Motif, index: IndexBuilder, use_bitarray_backend: bool) -> int:
        if use_bitarray_backend:
            S = index.get_bitset(motif.itemsets[0])
            for j in range(1, len(motif)):
                Prev = self._dilate_any_within_gap_bitarray(S, self.gap_min, self.gap_max)
                Curr_pos = index.get_bitset(motif.itemsets[j])
                S = Prev & Curr_pos
            return S.count()

        S = index.get_bitset_numpy(motif.itemsets[0])
        for j in range(1, len(motif)):
            Prev = self._dilate_any_within_gap_numpy(S, self.gap_min, self.gap_max)
            Curr_pos = index.get_bitset_numpy(motif.itemsets[j])
            S = Prev & Curr_pos
        return int(S.sum())

    def _phase_B_verify(self, candidates: List[Motif], index: IndexBuilder) -> Tuple[List[Motif], Dict]:
        self._phase_b_timeout = False
        t_start = None
        if self.time_budget_s is not None:
            t_start = time.perf_counter()

        if self.verify_top_k is not None and self.verify_top_k > 0 and candidates:
            candidates = sorted(
                candidates,
                key=lambda m: self._candidate_support.get(m, 0),
                reverse=True
            )[:self.verify_top_k]
        if self.time_budget_s is None and candidates:
            # Group by first itemset to improve position cache locality.
            candidates = sorted(candidates, key=lambda m: m.itemsets[0])

        verified_motifs = []
        motif_occurrences = {}
        if self.n_jobs > 1 and candidates:
            try:
                n_jobs = self.n_jobs
                chunksize = max(1, len(candidates) // (n_jobs * 4))
                if _can_use_process_pool():
                    from multiprocessing import Pool, cpu_count
                    n_jobs = min(n_jobs, cpu_count())
                    if n_jobs > 1:
                        if self.use_shm_positions:
                            try:
                                from multiprocessing import shared_memory
                                shm_handles = {}
                                shm_metadata = {}
                                for event_id, positions in index.position_lists.items():
                                    arr = np.asarray(positions, dtype=np.int64)
                                    if arr.nbytes == 0:
                                        continue
                                    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
                                    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                                    shm_arr[:] = arr
                                    shm_handles[event_id] = shm
                                    shm_metadata[event_id] = (shm.name, arr.shape, arr.dtype.str)
                                pool = Pool(
                                    processes=n_jobs,
                                    initializer=_init_occurrence_worker_shm,
                                    initargs=(
                                        shm_metadata,
                                        self.gap_min,
                                        self.gap_max,
                                        self._min_support_count,
                                        self.delay_occurrence_materialization,
                                    )
                                )
                                try:
                                    results_iter = pool.imap_unordered(
                                        _find_occurrences_worker,
                                        candidates,
                                        chunksize=chunksize
                                    )
                                    for motif, occurrences in results_iter:
                                        if t_start is not None and (time.perf_counter() - t_start) > self.time_budget_s:
                                            self._phase_b_timeout = True
                                            break
                                        if occurrences:
                                            verified_motifs.append(motif)
                                            motif_occurrences[motif] = occurrences
                                finally:
                                    if self._phase_b_timeout:
                                        pool.terminate()
                                    else:
                                        pool.close()
                                    pool.join()
                                    for shm in shm_handles.values():
                                        try:
                                            shm.close()
                                            shm.unlink()
                                        except Exception:
                                            pass
                                return verified_motifs, motif_occurrences
                            except Exception:
                                pass
                        pool = Pool(
                            processes=n_jobs,
                            initializer=_init_occurrence_worker,
                            initargs=(
                                index.position_lists,
                                self.gap_min,
                                self.gap_max,
                                self._min_support_count,
                                self.delay_occurrence_materialization,
                            )
                        )
                        try:
                            results_iter = pool.imap_unordered(
                                _find_occurrences_worker,
                                candidates,
                                chunksize=chunksize
                            )
                            for motif, occurrences in results_iter:
                                if t_start is not None and (time.perf_counter() - t_start) > self.time_budget_s:
                                    self._phase_b_timeout = True
                                    break
                                if occurrences:
                                    verified_motifs.append(motif)
                                    motif_occurrences[motif] = occurrences
                        finally:
                            if self._phase_b_timeout:
                                pool.terminate()
                            else:
                                pool.close()
                            pool.join()
                        return verified_motifs, motif_occurrences
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    _init_occurrence_worker(
                        index.position_lists,
                        self.gap_min,
                        self.gap_max,
                        self._min_support_count,
                        self.delay_occurrence_materialization
                    )
                    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                        futures = [executor.submit(_find_occurrences_worker, m) for m in candidates]
                        for future in as_completed(futures):
                            if t_start is not None and (time.perf_counter() - t_start) > self.time_budget_s:
                                self._phase_b_timeout = True
                                break
                            motif, occurrences = future.result()
                            if occurrences:
                                verified_motifs.append(motif)
                                motif_occurrences[motif] = occurrences
                    return verified_motifs, motif_occurrences
            except Exception:
                pass

        if self.next_pos_top_n is not None and self.next_pos_top_n > 0:
            self._prepare_next_pos(index, candidates)
        else:
            self._next_pos = None

        for motif in candidates:
            if t_start is not None and (time.perf_counter() - t_start) > self.time_budget_s:
                self._phase_b_timeout = True
                break
            occurrences = self._find_occurrences(motif, index)

            if occurrences:
                verified_motifs.append(motif)
                motif_occurrences[motif] = occurrences

        return verified_motifs, motif_occurrences

    def _prepare_next_pos(self, index: IndexBuilder, candidates: Optional[List[Motif]] = None) -> None:
        T = index.T
        target_events = None
        if candidates:
            event_set = set()
            for motif in candidates:
                for itemset in motif.itemsets:
                    if len(itemset) == 1:
                        event_set.add(itemset[0])
            if event_set:
                target_events = list(event_set)

        if target_events is None:
            event_support = [(e, len(pos)) for e, pos in index.position_lists.items()]
            event_support.sort(key=lambda x: x[1], reverse=True)
            target_events = [e for e, _ in event_support[:self.next_pos_top_n]]

        next_pos_map = {}
        for e in target_events:
            pos = index.position_lists.get(e, [])
            if len(pos) == 0:
                continue
            nxt = np.full(T + 1, -1, dtype=np.int32)
            p_idx = len(pos) - 1
            next_val = -1
            for t in range(T, -1, -1):
                if p_idx >= 0 and pos[p_idx] == t:
                    next_val = t
                    p_idx -= 1
                nxt[t] = next_val
            next_pos_map[e] = nxt
        self._next_pos = next_pos_map

    def _find_occurrences_next_pos(self, motif: Motif) -> List:
        events = [itemset[0] for itemset in motif.itemsets]
        for e in events:
            if e not in self._next_pos:
                return []
        occurrences = []
        next_start_min = 0
        max_occ = self.max_occurrences
        while True:
            t0 = self._next_pos[events[0]][next_start_min]
            if t0 < 0:
                break
            chain = [int(t0)]
            valid = True
            for e in events[1:]:
                lo = chain[-1] + self.gap_min
                hi = chain[-1] + self.gap_max
                if lo > self._T - 1:
                    valid = False
                    break
                t = self._next_pos[e][lo]
                if t < 0 or t > hi:
                    valid = False
                    break
                chain.append(int(t))

            if valid:
                occurrences.append(tuple(chain))
                if max_occ is not None and len(occurrences) >= max_occ:
                    break
                next_start_min = chain[-1] + 1
            else:
                next_start_min = t0 + 1
        if self.delay_occurrence_materialization:
            return occurrences
        return [Occurrence(occ) for occ in occurrences]

    def _make_occurrence(self, chain: List[int]):
        if self.delay_occurrence_materialization:
            return tuple(chain)
        return Occurrence(tuple(chain))

    def _find_occurrences(self, motif: Motif, index: IndexBuilder) -> List:
        if self._next_pos is not None:
            if all(len(itemset) == 1 for itemset in motif.itemsets):
                occs = self._find_occurrences_next_pos(motif)
                if occs:
                    return occs

        if len(motif.itemsets) == 1:
            positions = index.get_positions(motif.itemsets[0])
            if len(positions) == 0:
                return []
            max_occ = self.max_occurrences
            if max_occ is not None:
                positions = positions[:max_occ]
            if self.delay_occurrence_materialization:
                return [(int(p),) for p in positions]
            return [Occurrence((int(p),)) for p in positions]

        position_lists = []
        for itemset in motif.itemsets:
            positions = index.get_positions(itemset)
            if len(positions) == 0:
                return []
            position_lists.append(positions)

        mins = [pl[0] for pl in position_lists]
        maxs = [pl[-1] for pl in position_lists]
        for i in range(len(position_lists) - 1):
            for j in range(i + 1, len(position_lists)):
                delta = j - i
                if mins[j] > maxs[i] + delta * self.gap_max:
                    return []
                if maxs[j] < mins[i] + delta * self.gap_min:
                    return []

        if len(position_lists) == 2:
            L0, L1 = position_lists
            occurrences = []
            cursor0 = bisect_left(L0, 0, 0)
            cursor1 = 0
            max_occ = self.max_occurrences
            min_support_count = self._min_support_count
            next_start_min = 0
            while cursor0 < len(L0):
                if min_support_count > 1:
                    remaining = len(L0) - cursor0
                    if len(occurrences) + remaining < min_support_count:
                        break
                t0 = L0[cursor0]
                lo = t0 + self.gap_min
                hi = t0 + self.gap_max
                cursor1 = bisect_left(L1, lo, cursor1)
                if cursor1 < len(L1) and L1[cursor1] <= hi:
                    t1 = L1[cursor1]
                    occurrences.append((int(t0), int(t1)))
                    if max_occ is not None and len(occurrences) >= max_occ:
                        break
                    next_start_min = t1 + 1
                    cursor0 = bisect_left(L0, next_start_min, cursor0)
                else:
                    cursor0 += 1
            if self.delay_occurrence_materialization:
                return occurrences
            return [Occurrence(occ) for occ in occurrences]

        occurrences = []
        cursors = [0] * len(position_lists)
        next_start_min = 0
        L0 = position_lists[0]

        cursors[0] = bisect_left(L0, next_start_min, cursors[0])

        max_occ = self.max_occurrences
        min_support_count = self._min_support_count
        while cursors[0] < len(L0):
            if min_support_count > 1:
                remaining = len(L0) - cursors[0]
                if len(occurrences) + remaining < min_support_count:
                    break
            chain = [L0[cursors[0]]]
            valid = True

            for j in range(1, len(position_lists)):
                Lj = position_lists[j]
                lo = chain[-1] + self.gap_min
                hi = chain[-1] + self.gap_max

                cursors[j] = bisect_left(Lj, lo, cursors[j])
                if cursors[j] >= len(Lj) or Lj[cursors[j]] > hi:
                    valid = False
                    break

                chain.append(Lj[cursors[j]])

            if valid:
                occurrences.append(tuple(chain))
                if max_occ is not None and len(occurrences) >= max_occ:
                    break
                next_start_min = chain[-1] + 1
                cursors[0] = bisect_left(L0, next_start_min, cursors[0])
            else:
                cursors[0] += 1

        if self.delay_occurrence_materialization:
            return occurrences
        return [Occurrence(occ) for occ in occurrences]

# ==================== 5. RuleMiner: 规则挖掘 ====================

class RuleMiner:
    """从motifs生成并统计规则"""

    def __init__(self,
                 delta_min: int = 1,
                 delta_max: int = 10,
                 lambda_decay: float = None,
                 regimes: List[int] = None,
                 T: int = None,
                 min_conf_per_regime: float = 0.3,
                 min_rule_support_count: Optional[int] = None,
                 fast_rule_eval: bool = True,
                 rule_top_k: Optional[int] = None,
                 n_jobs: int = 1,
                 use_bitarray_windows: Optional[bool] = None,
                 use_packed_windows: bool = True,
                 batch_windows: bool = True,
                 use_simd_batch: bool = False):
        self.delta_min = delta_min
        self.delta_max = delta_max

        if lambda_decay is None and T is not None:
            half_life = T / 5
            self.lambda_decay = np.log(2) / half_life
        else:
            self.lambda_decay = lambda_decay or 0.001

        self.regimes = regimes or [0, T] if T else [0, 1000]
        self.T = T or self.regimes[-1]
        if self.T > 0:
            self._decay_weights = np.exp(-self.lambda_decay * (self.T - np.arange(self.T)))
        else:
            self._decay_weights = np.array([], dtype=float)
        self.min_conf_per_regime = min_conf_per_regime
        self.min_rule_support_count = min_rule_support_count
        self.fast_rule_eval = bool(fast_rule_eval)
        self.rule_top_k = rule_top_k
        self.n_jobs = max(1, n_jobs)
        if use_bitarray_windows is None:
            self.use_bitarray_windows = bool(BITARRAY_AVAILABLE)
        else:
            self.use_bitarray_windows = bool(use_bitarray_windows)
        self.use_packed_windows = bool(use_packed_windows)
        self.batch_windows = bool(batch_windows)
        self.use_simd_batch = bool(use_simd_batch)
        self._regime_of_t = self._build_regime_map()

    def mine_rules(self,
                   motifs: List[Motif],
                   motif_occurrences: Dict[Motif, List[Occurrence]],
                   top_M1: int = 1000,
                   top_M2: int = 5000,
                   index: Optional[IndexBuilder] = None,
                   max_consequent_itemset_size: int = 2,
                   time_budget_s: Optional[float] = None) -> List[Rule]:
        print("开始规则挖掘...")
        self._rule_timeout = False
        t_start = time.perf_counter() if time_budget_s is not None else None

        motif_scores = []
        for motif in motifs:
            occs = motif_occurrences[motif]
            if occs:
                ends = np.fromiter((_occ_end(occ) for occ in occs), dtype=int, count=len(occs))
                sup_lambda = float(self._decay_weights[ends].sum()) if ends.size > 0 else 0.0
            else:
                sup_lambda = 0.0
            motif_scores.append((motif, sup_lambda, len(occs)))

        motif_scores.sort(key=lambda x: x[1], reverse=True)

        antecedents = [m for m, _, _ in motif_scores[:top_M1]]
        if index is not None:
            consequents = self._build_consequents_from_index(
                index, top_M2, max_consequent_itemset_size
            )
            occurrences_map = dict(motif_occurrences)
            for Q in consequents:
                if Q not in occurrences_map:
                    occurrences_map[Q] = self._build_single_step_occurrences(Q, index)
        else:
            consequents = [
                m for m, _, _ in motif_scores
                if len(m) == 1 and len(m.itemsets[0]) <= 2
            ][:top_M2]
            occurrences_map = motif_occurrences

        if self.min_rule_support_count is not None:
            min_rule = int(self.min_rule_support_count)
            antecedents = [
                m for m in antecedents
                if len(occurrences_map.get(m, [])) >= min_rule
            ]

        print(f"  前因候选: {len(antecedents)}, 后因候选: {len(consequents)}")

        rules = []
        top_k = self.rule_top_k
        rule_heap = []
        heap_counter = 0
        if self.fast_rule_eval:
            p_bitsets_map = self._build_occurrence_bitsets(antecedents, occurrences_map)
            q_windows = self._build_window_bitsets(consequents, occurrences_map)
            use_bitarray_windows = self.use_bitarray_windows and BITARRAY_AVAILABLE
            if self.min_rule_support_count is not None:
                min_rule = int(self.min_rule_support_count)
                consequents = [
                    q for q in consequents
                    if q_windows.get(q) is not None and q_windows[q]["count"] >= min_rule
                ]
            sup_P_map = {m: len(occurrences_map.get(m, [])) for m in antecedents}

            q_list = []
            q_windows_list = []
            q_counts_list = []
            q_prior_list = []
            for q in consequents:
                q_win = q_windows.get(q)
                if q_win is None:
                    continue
                q_list.append(q)
                if use_bitarray_windows:
                    q_windows_list.append(q_win["bitset"])
                else:
                    q_windows_list.append(q_win["window"])
                q_counts_list.append(q_win["count"])
                q_prior_list.append(q_win["prior"])

            Rn = len(self.regimes) - 1
            p_motifs = []
            p_ends = []
            p_wp = []
            p_r = []
            p_reg_counts = []
            p_sup_decay = []
            p_bitsets_list = [] if use_bitarray_windows else None
            for P in antecedents:
                occs = occurrences_map.get(P, [])
                ends = np.fromiter((_occ_end(occ) for occ in occs), dtype=int, count=len(occs))
                if ends.size > 1 and np.any(ends[1:] < ends[:-1]):
                    ends = np.sort(ends)
                p_motifs.append(P)
                p_ends.append(ends)
                if use_bitarray_windows:
                    bs = bitarray(self.T)
                    bs.setall(False)
                    for t in ends:
                        if 0 <= t < self.T:
                            bs[t] = True
                    p_bitsets_list.append(bs)
                if ends.size > 0:
                    wP = self._decay_weights[ends]
                    r = self._regime_of_t[ends]
                    if Rn > 0:
                        mask = r >= 0
                        if mask.any():
                            reg_counts = np.bincount(r[mask], minlength=Rn)
                        else:
                            reg_counts = np.zeros(Rn, dtype=int)
                    else:
                        reg_counts = np.zeros(Rn, dtype=int)
                else:
                    wP = np.array([], dtype=float)
                    r = np.array([], dtype=int)
                    reg_counts = np.zeros(Rn, dtype=int)
                p_wp.append(wP)
                p_r.append(r)
                p_reg_counts.append(reg_counts)
                p_sup_decay.append(float(wP.sum()))

            use_parallel = self.n_jobs > 1 and antecedents and consequents
            if use_parallel:
                try:
                    n_jobs = self.n_jobs
                    chunksize = max(1, len(antecedents) // (n_jobs * 4))

                    if q_list and p_motifs:
                        indices = list(range(len(p_motifs)))
                        q_packed = self.use_packed_windows and not use_bitarray_windows
                        if _can_use_process_pool():
                            from multiprocessing import Pool, cpu_count
                            n_jobs = min(n_jobs, cpu_count())
                            if n_jobs > 1:
                                shm = None
                                q_windows_payload = q_windows_list
                                q_shm_name = None
                                q_shm_shape = None
                                q_shm_dtype = None
                                if not use_bitarray_windows:
                                    try:
                                        from multiprocessing import shared_memory
                                        q_windows_matrix = np.stack(q_windows_list)
                                        if q_packed:
                                            q_windows_matrix = np.packbits(q_windows_matrix.astype(np.uint8, copy=False), axis=1)
                                        else:
                                            q_windows_matrix = q_windows_matrix.astype(np.uint8, copy=False)
                                        shm = shared_memory.SharedMemory(create=True, size=q_windows_matrix.nbytes)
                                        shm_arr = np.ndarray(
                                            q_windows_matrix.shape,
                                            dtype=q_windows_matrix.dtype,
                                            buffer=shm.buf
                                        )
                                        shm_arr[:] = q_windows_matrix
                                        q_windows_payload = None
                                        q_shm_name = shm.name
                                        q_shm_shape = q_windows_matrix.shape
                                        q_shm_dtype = q_windows_matrix.dtype.str
                                    except Exception:
                                        if shm is not None:
                                            try:
                                                shm.close()
                                                shm.unlink()
                                            except Exception:
                                                pass
                                        shm = None
                                pool = Pool(
                                    processes=n_jobs,
                                    initializer=_init_rule_light_worker,
                                    initargs=(
                                        p_motifs,
                                        p_ends,
                                        p_wp,
                                        p_r,
                                        p_reg_counts,
                                        p_sup_decay,
                                        p_bitsets_list,
                                        q_list,
                                        q_windows_payload,
                                        q_counts_list,
                                        q_prior_list,
                                        self.delta_min,
                                        self.delta_max,
                                        self.lambda_decay,
                                        self.regimes,
                                        self._regime_of_t,
                                        self.T,
                                        self.min_conf_per_regime,
                                        self.min_rule_support_count,
                                        top_k,
                                        q_packed,
                                        q_shm_name,
                                        q_shm_shape,
                                        q_shm_dtype,
                                    ),
                                )
                                try:
                                    results_iter = pool.imap_unordered(
                                        _rule_light_worker,
                                        indices,
                                        chunksize=chunksize
                                    )
                                    for part in results_iter:
                                        if t_start is not None and (time.perf_counter() - t_start) > time_budget_s:
                                            self._rule_timeout = True
                                            break
                                        if not part:
                                            continue
                                        if top_k is None:
                                            rules.extend([r for _, r in part])
                                        else:
                                            for sup_rule, rule in part:
                                                heapq.heappush(rule_heap, (sup_rule, heap_counter, rule))
                                                heap_counter += 1
                                                if len(rule_heap) > top_k:
                                                    heapq.heappop(rule_heap)
                                finally:
                                    if self._rule_timeout:
                                        pool.terminate()
                                    else:
                                        pool.close()
                                    pool.join()
                                    if shm is not None:
                                        try:
                                            shm.close()
                                            shm.unlink()
                                        except Exception:
                                            pass
                                if top_k is not None:
                                    rules = [r for _, _, r in sorted(rule_heap, key=lambda x: x[0], reverse=True)]
                                print(f"  生成{len(rules)}条规则")
                                return rules
                        else:
                            from concurrent.futures import ThreadPoolExecutor, as_completed
                            if not use_bitarray_windows and q_packed:
                                q_windows_matrix = np.packbits(np.stack(q_windows_list).astype(np.uint8, copy=False), axis=1)
                                q_windows_list = q_windows_matrix
                            _init_rule_light_worker(
                                p_motifs,
                                p_ends,
                                p_wp,
                                p_r,
                                p_reg_counts,
                                p_sup_decay,
                                p_bitsets_list,
                                q_list,
                                q_windows_list,
                                q_counts_list,
                                q_prior_list,
                                self.delta_min,
                                self.delta_max,
                                self.lambda_decay,
                                self.regimes,
                                self._regime_of_t,
                                self.T,
                                self.min_conf_per_regime,
                                self.min_rule_support_count,
                                top_k,
                                q_packed,
                                None,
                                None,
                                None,
                            )
                            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                                futures = [executor.submit(_rule_light_worker, i) for i in indices]
                                for future in as_completed(futures):
                                    if t_start is not None and (time.perf_counter() - t_start) > time_budget_s:
                                        self._rule_timeout = True
                                        break
                                    part = future.result()
                                    if not part:
                                        continue
                                    if top_k is None:
                                        rules.extend([r for _, r in part])
                                    else:
                                        for sup_rule, rule in part:
                                            heapq.heappush(rule_heap, (sup_rule, heap_counter, rule))
                                            heap_counter += 1
                                            if len(rule_heap) > top_k:
                                                heapq.heappop(rule_heap)
                            if top_k is not None:
                                rules = [r for _, _, r in sorted(rule_heap, key=lambda x: x[0], reverse=True)]
                            print(f"  生成{len(rules)}条规则")
                            return rules
                except Exception:
                    pass

            if q_list and p_motifs and self.batch_windows:
                q_windows_matrix = None
                try:
                    if use_bitarray_windows:
                        q_windows_matrix = np.stack([
                            np.fromiter(bs, dtype=np.uint8, count=len(bs)) for bs in q_windows_list
                        ])
                    else:
                        q_windows_matrix = np.stack(q_windows_list).astype(np.uint8, copy=False)
                except Exception:
                    q_windows_matrix = None

                if q_windows_matrix is not None:
                    indices = list(range(len(p_motifs)))
                    Qn = len(q_list)
                    for p_idx in indices:
                        if t_start is not None and (time.perf_counter() - t_start) > time_budget_s:
                            self._rule_timeout = True
                            break
                        ends = p_ends[p_idx]
                        if ends.size == 0:
                            continue
                        mask_matrix = q_windows_matrix[:, ends]
                        sup_rule_vec = mask_matrix.sum(axis=1)
                        if self.min_rule_support_count is not None:
                            sup_rule_mask = sup_rule_vec >= self.min_rule_support_count
                        else:
                            sup_rule_mask = sup_rule_vec > 0
                        if not sup_rule_mask.any():
                            continue

                        wP = p_wp[p_idx]
                        supP_decay = p_sup_decay[p_idx]
                        if supP_decay > 0:
                            matches_decay_vec = mask_matrix @ wP
                        else:
                            matches_decay_vec = None

                        if Rn > 0:
                            r = p_r[p_idx]
                            regime_matches_matrix = np.zeros((Qn, Rn), dtype=int)
                            for ri in range(Rn):
                                idxs = np.where(r == ri)[0]
                                if idxs.size > 0:
                                    regime_matches_matrix[:, ri] = mask_matrix[:, idxs].sum(axis=1)
                        else:
                            regime_matches_matrix = None

                        sup_P = int(ends.size)
                        reg_counts = p_reg_counts[p_idx]
                        P = p_motifs[p_idx]

                        if top_k is not None and len(rule_heap) >= top_k:
                            min_keep = rule_heap[0][0]
                        else:
                            min_keep = None

                        for q_idx in range(Qn):
                            if P == q_list[q_idx]:
                                continue
                            sup_rule = int(sup_rule_vec[q_idx])
                            if sup_rule <= 0:
                                continue
                            if self.min_rule_support_count is not None and sup_rule < self.min_rule_support_count:
                                continue
                            if top_k is not None and len(rule_heap) >= top_k and sup_rule <= min_keep:
                                continue

                            rule = Rule(P, q_list[q_idx], self.delta_min, self.delta_max)
                            rule.sup_P = sup_P
                            rule.sup_rule = sup_rule

                            prior_win = q_prior_list[q_idx]
                            rule.conf = sup_rule / sup_P if sup_P > 0 else 0.0
                            rule.lift = (rule.conf / prior_win) if prior_win > 0 else 0.0

                            rule.sup_P_lambda = supP_decay
                            matches_decay = float(matches_decay_vec[q_idx]) if matches_decay_vec is not None else 0.0
                            rule.sup_rule_lambda = matches_decay
                            rule.conf_lambda = matches_decay / supP_decay if supP_decay > 0 else 0.0
                            rule.lift_lambda = (rule.conf_lambda / prior_win) if prior_win > 0 else 0.0

                            rule.conf_per_regime = []
                            stable = 0
                            if Rn > 0:
                                regime_matches = regime_matches_matrix[q_idx]
                            else:
                                regime_matches = None
                            for ri in range(Rn):
                                denom = reg_counts[ri] if ri < len(reg_counts) else 0
                                if denom > 0:
                                    cr = regime_matches[ri] / denom
                                    rule.conf_per_regime.append(cr)
                                    if cr >= self.min_conf_per_regime:
                                        stable += 1
                                else:
                                    rule.conf_per_regime.append(0.0)
                            rule.stability = stable / Rn if Rn > 0 else 0.0

                            if top_k is None:
                                rules.append(rule)
                            else:
                                heapq.heappush(rule_heap, (sup_rule, heap_counter, rule))
                                heap_counter += 1
                                if len(rule_heap) > top_k:
                                    heapq.heappop(rule_heap)
                                min_keep = rule_heap[0][0]

                    if top_k is not None:
                        rules = [r for _, _, r in sorted(rule_heap, key=lambda x: x[0], reverse=True)]
                    print(f"  生成{len(rules)}条规则")
                    return rules

            if q_list and p_motifs:
                indices = list(range(len(p_motifs)))
                q_packed = self.use_packed_windows and not use_bitarray_windows
                if not use_bitarray_windows and q_packed:
                    q_windows_matrix = np.packbits(np.stack(q_windows_list).astype(np.uint8, copy=False), axis=1)
                    q_windows_list = q_windows_matrix
                _init_rule_light_worker(
                    p_motifs,
                    p_ends,
                    p_wp,
                    p_r,
                    p_reg_counts,
                    p_sup_decay,
                    p_bitsets_list,
                    q_list,
                    q_windows_list,
                    q_counts_list,
                    q_prior_list,
                    self.delta_min,
                    self.delta_max,
                    self.lambda_decay,
                    self.regimes,
                    self._regime_of_t,
                    self.T,
                    self.min_conf_per_regime,
                    self.min_rule_support_count,
                    top_k,
                    q_packed,
                    None,
                    None,
                    None,
                )
                for idx in indices:
                    if t_start is not None and (time.perf_counter() - t_start) > time_budget_s:
                        self._rule_timeout = True
                        break
                    part = _rule_light_worker(idx)
                    if not part:
                        continue
                    if top_k is None:
                        rules.extend([r for _, r in part])
                    else:
                        for sup_rule, rule in part:
                            heapq.heappush(rule_heap, (sup_rule, heap_counter, rule))
                            heap_counter += 1
                            if len(rule_heap) > top_k:
                                heapq.heappop(rule_heap)
                if top_k is not None:
                    rules = [r for _, _, r in sorted(rule_heap, key=lambda x: x[0], reverse=True)]
                print(f"  生成{len(rules)}条规则")
                return rules

            for P in antecedents:
                sup_P = sup_P_map.get(P, 0)
                for Q in consequents:
                    if t_start is not None and (time.perf_counter() - t_start) > time_budget_s:
                        self._rule_timeout = True
                        break
                    if P == Q:
                        continue
                    p_bs = p_bitsets_map.get(P)
                    q_win = q_windows.get(Q)
                    if p_bs is None or q_win is None:
                        continue
                    upper = sup_P if sup_P < q_win["count"] else q_win["count"]
                    if self.min_rule_support_count is not None and upper < self.min_rule_support_count:
                        continue
                    if top_k is not None and len(rule_heap) >= top_k and upper <= rule_heap[0][0]:
                        continue
                    sup_rule = self._bitset_count(p_bs & q_win["bitset"])
                    if sup_rule <= 0:
                        continue
                    if self.min_rule_support_count is not None and sup_rule < self.min_rule_support_count:
                        continue
                    if top_k is not None and len(rule_heap) >= top_k and sup_rule <= rule_heap[0][0]:
                        continue
                    rule = self._compute_rule_stats_window(P, Q, occurrences_map, q_win, sup_rule)
                    if rule.sup_rule > 0:
                        if top_k is None:
                            rules.append(rule)
                        else:
                            heapq.heappush(rule_heap, (rule.sup_rule, heap_counter, rule))
                            heap_counter += 1
                            if len(rule_heap) > top_k:
                                heapq.heappop(rule_heap)
                if self._rule_timeout:
                    break
        else:
            for P in antecedents:
                for Q in consequents:
                    if t_start is not None and (time.perf_counter() - t_start) > time_budget_s:
                        self._rule_timeout = True
                        break
                    if P == Q:
                        continue

                    rule = self._compute_rule_stats(P, Q, occurrences_map)
                    if rule.sup_rule > 0:
                        if top_k is None:
                            rules.append(rule)
                        else:
                            heapq.heappush(rule_heap, (rule.sup_rule, heap_counter, rule))
                            heap_counter += 1
                            if len(rule_heap) > top_k:
                                heapq.heappop(rule_heap)
                if self._rule_timeout:
                    break
        if top_k is not None:
            rules = [r for _, _, r in sorted(rule_heap, key=lambda x: x[0], reverse=True)]

        print(f"  生成{len(rules)}条规则")
        return rules


    def _build_single_step_occurrences(self, motif: Motif, index: IndexBuilder) -> List[Occurrence]:
        if len(motif.itemsets) != 1:
            return []
        positions = index.get_positions(motif.itemsets[0])
        return [Occurrence((pos,)) for pos in positions]

    def _build_consequents_from_index(self,
                                      index: IndexBuilder,
                                      top_q: int = 500,
                                      max_itemset_size: int = 2,
                                      single_ratio: float = 0.7) -> List[Motif]:
        ev = [(e, index.event_support(e)) for e in index.bitsets]
        ev.sort(key=lambda x: x[1], reverse=True)
        if not ev or top_q <= 0:
            return []

        if max_itemset_size < 2:
            return [Motif(((e,),)) for e, _ in ev[:top_q]]

        n_single = min(len(ev), max(1, int(top_q * single_ratio)), top_q)
        n_pair = max(0, top_q - n_single)

        singles = [Motif(((e,),)) for e, _ in ev[:n_single]]
        if n_pair == 0:
            return singles

        N = min(80, len(ev))
        top_events = [e for e, _ in ev[:N]]
        pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                e1, e2 = top_events[i], top_events[j]
                sup = index.bitset_and_count(e1, e2)
                if sup > 0:
                    itemset = (e1, e2) if e1 < e2 else (e2, e1)
                    pairs.append((Motif((itemset,)), sup))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return singles + [m for m, _ in pairs[:n_pair]]

    def _compute_rule_stats(self, P: Motif, Q: Motif, motif_occurrences: Dict) -> Rule:
        P_occs = motif_occurrences[P]
        Q_occs = motif_occurrences[Q]

        Q_starts = sorted(_occ_start(occ) for occ in Q_occs)
        P_ends = sorted(_occ_end(occ) for occ in P_occs)

        rule = Rule(P, Q, self.delta_min, self.delta_max)
        rule.sup_P = len(P_ends)

        Rn = len(self.regimes) - 1
        regime_matches = [0] * Rn
        regime_P_counts = [0] * Rn
        regime_of_t = self._regime_of_t

        j = 0
        matches = 0
        matches_decay = 0.0
        supP_decay = 0.0

        for tP in P_ends:
            if 0 <= tP < self.T:
                wP = self._decay_weights[tP]
            else:
                wP = np.exp(-self.lambda_decay * (self.T - tP))
            supP_decay += wP

            r = regime_of_t[tP] if 0 <= tP < self.T else -1
            if r >= 0:
                regime_P_counts[r] += 1

            lo = tP + self.delta_min
            hi = tP + self.delta_max

            while j < len(Q_starts) and Q_starts[j] < lo:
                j += 1
            if j < len(Q_starts) and Q_starts[j] <= hi:
                matches += 1
                matches_decay += wP
                if r >= 0:
                    regime_matches[r] += 1

        rule.sup_rule = matches
        rule.conf = matches / rule.sup_P if rule.sup_P > 0 else 0.0

        prior_win = self._window_prior(Q_starts)
        rule.lift = (rule.conf / prior_win) if prior_win > 0 else 0.0

        rule.sup_P_lambda = supP_decay
        rule.sup_rule_lambda = matches_decay
        rule.conf_lambda = matches_decay / supP_decay if supP_decay > 0 else 0.0
        rule.lift_lambda = (rule.conf_lambda / prior_win) if prior_win > 0 else 0.0

        rule.conf_per_regime = []
        stable = 0
        for r in range(Rn):
            if regime_P_counts[r] > 0:
                cr = regime_matches[r] / regime_P_counts[r]
                rule.conf_per_regime.append(cr)
                if cr >= self.min_conf_per_regime:
                    stable += 1
            else:
                rule.conf_per_regime.append(0.0)

        rule.stability = stable / Rn if Rn > 0 else 0.0
        return rule

    def _build_occurrence_bitsets(self,
                                  motifs: List[Motif],
                                  motif_occurrences: Dict[Motif, List[Occurrence]]):
        out = {}
        for motif in motifs:
            occs = motif_occurrences.get(motif, [])
            if not occs:
                continue
            if BITARRAY_AVAILABLE:
                bs = bitarray(self.T)
                bs.setall(False)
            else:
                bs = np.zeros(self.T, dtype=bool)
            for occ in occs:
                t = _occ_end(occ)
                if 0 <= t < self.T:
                    bs[t] = True
            out[motif] = bs
        return out

    def _build_window_bitsets(self,
                              motifs: List[Motif],
                              motif_occurrences: Dict[Motif, List[Occurrence]]):
        out = {}
        for motif in motifs:
            occs = motif_occurrences.get(motif, [])
            if not occs:
                continue
            diff = np.zeros(self.T + 1, dtype=int)
            starts = []
            for occ in occs:
                q = _occ_start(occ)
                starts.append(q)
                a = q - self.delta_max
                b = q - self.delta_min
                if b < 0 or a > self.T - 1:
                    continue
                if a < 0:
                    a = 0
                if b > self.T - 1:
                    b = self.T - 1
                diff[a] += 1
                diff[b + 1] -= 1
            window = np.cumsum(diff[:-1]) > 0
            if BITARRAY_AVAILABLE:
                bs = bitarray(self.T)
                bs.setall(False)
                for i, v in enumerate(window):
                    if v:
                        bs[i] = True
            else:
                bs = window
            if self.use_bitarray_windows and BITARRAY_AVAILABLE:
                count = bs.count()
                window_store = None
            else:
                count = int(window.sum())
                window_store = window
            out[motif] = {
                "bitset": bs,
                "window": window_store,
                "starts": starts,
                "count": count,
                "prior": float(count) / float(self.T) if self.T > 0 else 0.0,
            }
        return out

    def _bitset_count(self, bs) -> int:
        if BITARRAY_AVAILABLE:
            return bs.count()
        return int(bs.sum())

    def _compute_rule_stats_window(self,
                                   P: Motif,
                                   Q: Motif,
                                   motif_occurrences: Dict,
                                   q_window: Dict,
                                   sup_rule: int) -> Rule:
        P_occs = motif_occurrences[P]
        rule = Rule(P, Q, self.delta_min, self.delta_max)
        rule.sup_P = len(P_occs)
        rule.sup_rule = sup_rule

        prior_win = q_window["prior"]
        rule.conf = sup_rule / rule.sup_P if rule.sup_P > 0 else 0.0
        rule.lift = (rule.conf / prior_win) if prior_win > 0 else 0.0

        Rn = len(self.regimes) - 1
        regime_matches = [0] * Rn
        regime_P_counts = [0] * Rn
        regime_of_t = self._regime_of_t
        window = q_window.get("window")
        window_bs = q_window["bitset"] if window is None else None

        matches_decay = 0.0
        supP_decay = 0.0

        P_ends = sorted(_occ_end(occ) for occ in P_occs)
        for tP in P_ends:
            if tP < 0 or tP >= self.T:
                continue
            wP = self._decay_weights[tP]
            supP_decay += wP

            r = regime_of_t[tP]
            if r >= 0:
                regime_P_counts[r] += 1

            if window is not None:
                if window[tP]:
                    matches_decay += wP
                    if r >= 0:
                        regime_matches[r] += 1
            else:
                if window_bs[tP]:
                    matches_decay += wP
                    if r >= 0:
                        regime_matches[r] += 1

        rule.sup_P_lambda = supP_decay
        rule.sup_rule_lambda = matches_decay
        rule.conf_lambda = matches_decay / supP_decay if supP_decay > 0 else 0.0
        rule.lift_lambda = (rule.conf_lambda / prior_win) if prior_win > 0 else 0.0

        rule.conf_per_regime = []
        stable = 0
        for r in range(Rn):
            if regime_P_counts[r] > 0:
                cr = regime_matches[r] / regime_P_counts[r]
                rule.conf_per_regime.append(cr)
                if cr >= self.min_conf_per_regime:
                    stable += 1
            else:
                rule.conf_per_regime.append(0.0)
        rule.stability = stable / Rn if Rn > 0 else 0.0
        return rule

    def _window_prior(self, Q_starts: List[int]) -> float:
        if not Q_starts or self.T <= 0:
            return 0.0
        intervals = []
        for q in Q_starts:
            a = q - self.delta_max
            b = q - self.delta_min
            if b < 0 or a > self.T - 1:
                continue
            if a < 0:
                a = 0
            if b > self.T - 1:
                b = self.T - 1
            if a <= b:
                intervals.append((a, b))
        if not intervals:
            return 0.0
        intervals.sort()
        union = 0
        cur_a, cur_b = intervals[0]
        for a, b in intervals[1:]:
            if a <= cur_b + 1:
                cur_b = max(cur_b, b)
            else:
                union += (cur_b - cur_a + 1)
                cur_a, cur_b = a, b
        union += (cur_b - cur_a + 1)
        return union / float(self.T)

    def _build_regime_map(self) -> np.ndarray:
        regime_of_t = np.full(self.T, -1, dtype=int)
        for r in range(len(self.regimes) - 1):
            a, b = self.regimes[r], self.regimes[r + 1]
            a = max(a, 0)
            b = min(b, self.T)
            if a < b:
                regime_of_t[a:b] = r
        return regime_of_t

    def _get_regime_index(self, t: int) -> Optional[int]:
        for r in range(len(self.regimes) - 1):
            if self.regimes[r] <= t < self.regimes[r + 1]:
                return r
        return None

# ==================== 6. Reporter: 输出报告 ====================

class Reporter:
    """输出双榜单与案例"""

    def rank_and_report(self,
                        rules: List[Rule],
                        top_K: int = 200,
                        event_sequence: List[Tuple[int, ...]] = None,
                        print_top_n: int = 10,
                        compute_ranking: bool = True) -> Dict:
        if not compute_ranking:
            total_rules = len(rules)
            avg_conf = float(np.mean([r.conf for r in rules])) if total_rules > 0 else 0.0
            avg_conf_lambda = float(np.mean([r.conf_lambda for r in rules])) if total_rules > 0 else 0.0
            avg_stability = float(np.mean([r.stability for r in rules if r.conf_per_regime])) if total_rules > 0 else 0.0
            return {
                'current_rules': [],
                'stable_rules': [],
                'summary': {
                    'total_rules': total_rules,
                    'avg_conf': avg_conf,
                    'avg_conf_lambda': avg_conf_lambda,
                    'avg_stability': avg_stability,
                }
            }

        print("\n==================== 规则排名 ====================")

        current_scores = []
        for rule in rules:
            score = (rule.sup_rule_lambda *
                     rule.conf_lambda *
                     np.log(1 + rule.lift_lambda))
            current_scores.append((rule, score))

        current_scores.sort(key=lambda x: x[1], reverse=True)
        top_current = [rule for rule, _ in current_scores[:top_K]]

        stable_scores = []
        for rule in rules:
            if rule.conf_per_regime:
                conf_quantile = np.percentile(rule.conf_per_regime, 20) if rule.conf_per_regime else 0.0
                score = rule.stability * conf_quantile
                stable_scores.append((rule, score))

        stable_scores.sort(key=lambda x: x[1], reverse=True)
        top_stable = [rule for rule, _ in stable_scores[:top_K]]

        if print_top_n and print_top_n > 0:
            print(f"\n【Top-{min(print_top_n, len(top_current))} Current Rules】")
            for i, rule in enumerate(top_current[:print_top_n]):
                print(f"{i + 1}. {rule}")
                print(f"   conf_λ={rule.conf_lambda:.3f}, lift_λ={rule.lift_lambda:.2f}, sup_λ={rule.sup_rule_lambda:.1f}")

            print(f"\n【Top-{min(print_top_n, len(top_stable))} Stable Rules】")
            for i, rule in enumerate(top_stable[:print_top_n]):
                print(f"{i + 1}. {rule}")
                print(f"   stability={rule.stability:.3f}, conf_per_regime={[f'{c:.2f}' for c in rule.conf_per_regime]}")

        total_rules = len(rules)
        avg_conf = float(np.mean([r.conf for r in rules])) if total_rules > 0 else 0.0
        avg_conf_lambda = float(np.mean([r.conf_lambda for r in rules])) if total_rules > 0 else 0.0
        avg_stability = float(np.mean([r.stability for r in rules if r.conf_per_regime])) if total_rules > 0 else 0.0

        return {
            'current_rules': top_current,
            'stable_rules': top_stable,
            'summary': {
                'total_rules': total_rules,
                'avg_conf': avg_conf,
                'avg_conf_lambda': avg_conf_lambda,
                'avg_stability': avg_stability,
            }
        }


# ==================== 主流程封装 ====================

class NonStationaryRuleMiner:
    """非平稳多变量序列规则挖掘系统"""

    def __init__(self, **kwargs):
        self.config = {
            'eventizer_kind': kwargs.get('eventizer_kind', 'default'),
            'n_bins': kwargs.get('n_bins', 4),
            'eventizer_window': kwargs.get('eventizer_window', 200),
            'quantile_stride': kwargs.get('quantile_stride', 10),
            'change_threshold': kwargs.get('change_threshold', 0.1),
            'event_support_decay_lambda': kwargs.get('event_support_decay_lambda', None),
            'event_support_min': kwargs.get('event_support_min', 0.01),
            'event_support_max': kwargs.get('event_support_max', 0.95),
            'use_incremental_eventizer': kwargs.get('use_incremental_eventizer', False),
            'use_bitarray': kwargs.get('use_bitarray', True),
            'index_cache_maxsize': kwargs.get('index_cache_maxsize', 10000),

            'seg_window': kwargs.get('seg_window', 1000),
            'seg_threshold': kwargs.get('seg_threshold', 0.1),

            'max_length': kwargs.get('max_length', 6),
            'max_itemset_size': kwargs.get('max_itemset_size', 1),
            'gap_min': kwargs.get('gap_min', 0),
            'gap_max': kwargs.get('gap_max', kwargs.get('gap', 3)),
            'top_M0': kwargs.get('top_M0', 200),
            'top_candidates': kwargs.get('top_candidates', 2000),
            'motif_n_jobs': kwargs.get('motif_n_jobs', 1),
            'motif_min_support_ratio': kwargs.get('motif_min_support_ratio', None),
            'motif_min_support_count': kwargs.get('motif_min_support_count', None),
            'motif_time_budget_s': kwargs.get('motif_time_budget_s', None),
            'motif_beam_width': kwargs.get('motif_beam_width', None),
            'motif_use_incremental_support': kwargs.get('motif_use_incremental_support', True),
            'motif_verify_top_k': kwargs.get('motif_verify_top_k', None),
            'motif_max_occurrences': kwargs.get('motif_max_occurrences', None),
            'motif_bitarray_gap_threshold': kwargs.get('motif_bitarray_gap_threshold', 8),
            'motif_delay_occurrence_materialization': kwargs.get('motif_delay_occurrence_materialization', False),
            'motif_next_pos_top_n': kwargs.get('motif_next_pos_top_n', 200),
            'motif_use_shm_positions': kwargs.get('motif_use_shm_positions', False),

            'delta_min': kwargs.get('delta_min', 1),
            'delta_max': kwargs.get('delta_max', 10),
            'top_M1': kwargs.get('top_M1', 200),
            'top_M2': kwargs.get('top_M2', 500),
            'max_consequent_itemset_size': kwargs.get('max_consequent_itemset_size', 2),
            'rule_min_support_count': kwargs.get('rule_min_support_count', None),
            'rule_fast_eval': kwargs.get('rule_fast_eval', True),
            'rule_top_k': kwargs.get('rule_top_k', None),
            'rule_n_jobs': kwargs.get('rule_n_jobs', 1),
            'rule_use_bitarray_windows': kwargs.get('rule_use_bitarray_windows', None),
            'rule_use_packed_windows': kwargs.get('rule_use_packed_windows', True),
            'rule_batch_windows': kwargs.get('rule_batch_windows', True),

            'top_K': kwargs.get('top_K', 200),
            'report_print_top_n': kwargs.get('report_print_top_n', 10),
            'report_compute_ranking': kwargs.get('report_compute_ranking', True),
            'print_stage_timing': kwargs.get('print_stage_timing', True),
        }

    def _normalize_fit_input(self, X):
        if isinstance(X, tuple):
            if len(X) != 2:
                raise ValueError("fit expects X or (X, feature_meta)")
            matrix, feature_meta = X
        else:
            matrix, feature_meta = X, None
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"expected 2D matrix, got shape {matrix.shape}")
        if feature_meta is None:
            feature_meta = {}
        return matrix, feature_meta

    def _build_eventizer(self, feature_meta: Optional[Dict] = None):
        if self.config['eventizer_kind'] == 'icu':
            from src.icu.icu_eventizer import ICUEventizer

            feature_names = []
            if feature_meta is not None:
                feature_names = list(feature_meta.get('feature_names', []))

            return ICUEventizer(
                feature_names=feature_names,
                n_bins=self.config['n_bins'],
                window_size=self.config['eventizer_window'],
                quantile_stride=self.config['quantile_stride'],
                change_threshold=self.config['change_threshold'],
                support_decay_lambda=self.config['event_support_decay_lambda'],
                support_min=self.config['event_support_min'],
                support_max=self.config['event_support_max'],
            )

        eventizer_cls = IncrementalEventizer if self.config['use_incremental_eventizer'] else Eventizer
        return eventizer_cls(
            n_bins=self.config['n_bins'],
            window_size=self.config['eventizer_window'],
            quantile_stride=self.config['quantile_stride'],
            change_threshold=self.config['change_threshold'],
            support_decay_lambda=self.config['event_support_decay_lambda'],
            support_min=self.config['event_support_min'],
            support_max=self.config['event_support_max']
        )

    def fit(self, X: np.ndarray | Tuple[np.ndarray, Dict]) -> Dict:
        X_matrix, feature_meta = self._normalize_fit_input(X)
        T, d = X_matrix.shape
        print(f"输入数据: T={T}, d={d}")
        timings = {}

        print("\n[1/6] 事件化...")
        t0 = time.perf_counter()
        eventizer = self._build_eventizer(feature_meta)
        if self.config['eventizer_kind'] == 'icu':
            event_sequence = eventizer.transform(X_matrix, feature_meta=feature_meta)
        else:
            event_sequence = eventizer.transform(X_matrix)
        _set_custom_event_labels(getattr(eventizer, 'event_labels', None))
        timings['Eventizer'] = time.perf_counter() - t0
        mean_events = float(np.mean([len(E) for E in event_sequence])) if event_sequence else 0.0
        print(f"  生成事件序列，平均每步{mean_events:.1f}个事件 (t={timings['Eventizer']:.3f}s)")

        print("\n[2/6] 分段检测...")
        t0 = time.perf_counter()
        segmenter = Segmenter(
            window_size=self.config['seg_window'],
            threshold=self.config['seg_threshold']
        )
        regimes = segmenter.detect_changepoints(event_sequence)
        timings['Segmenter'] = time.perf_counter() - t0
        print(f"  检测到{len(regimes) - 1}个阶段: {regimes} (t={timings['Segmenter']:.3f}s)")

        print("\n[3/6] 构建索引...")
        t0 = time.perf_counter()
        index = IndexBuilder(
            event_sequence,
            use_bitarray=self.config['use_bitarray'],
            cache_maxsize=self.config['index_cache_maxsize']
        )
        timings['Index'] = time.perf_counter() - t0
        print(f"  索引{len(index.bitsets)}个事件 (t={timings['Index']:.3f}s)")

        print("\n[4/6] Motif挖掘...")
        motif_miner = MotifMiner(
            max_length=self.config['max_length'],
            max_itemset_size=self.config['max_itemset_size'],
            gap_min=self.config['gap_min'],
            gap_max=self.config['gap_max'],
            top_M0=self.config['top_M0'],
            n_jobs=self.config['motif_n_jobs'],
            top_candidates=self.config['top_candidates'],
            min_support_ratio=self.config['motif_min_support_ratio'],
            min_support_count=self.config['motif_min_support_count'],
            time_budget_s=self.config['motif_time_budget_s'],
            beam_width=self.config['motif_beam_width'],
            use_incremental_support=self.config['motif_use_incremental_support'],
            verify_top_k=self.config['motif_verify_top_k'],
            max_occurrences=self.config['motif_max_occurrences'],
            bitarray_gap_threshold=self.config['motif_bitarray_gap_threshold'],
            delay_occurrence_materialization=self.config['motif_delay_occurrence_materialization'],
            next_pos_top_n=self.config['motif_next_pos_top_n'],
            use_shm_positions=self.config['motif_use_shm_positions']
        )
        motifs, motif_occurrences = motif_miner.mine(index)
        timings['MotifA'] = motif_miner.last_phase_times.get('MotifA', 0.0)
        timings['MotifB'] = motif_miner.last_phase_times.get('MotifB', 0.0)
        print(f"  MotifA: {timings['MotifA']:.3f}s, MotifB: {timings['MotifB']:.3f}s")

        print("\n[5/6] 规则挖掘...")
        t0 = time.perf_counter()
        rule_miner = RuleMiner(
            delta_min=self.config['delta_min'],
            delta_max=self.config['delta_max'],
            regimes=regimes,
            T=T,
            min_rule_support_count=self.config['rule_min_support_count'],
            fast_rule_eval=self.config['rule_fast_eval'],
            rule_top_k=self.config['rule_top_k'],
            n_jobs=self.config['rule_n_jobs'],
            use_bitarray_windows=self.config['rule_use_bitarray_windows'],
            use_packed_windows=self.config['rule_use_packed_windows'],
            batch_windows=self.config['rule_batch_windows']
        )
        rules = rule_miner.mine_rules(
            motifs, motif_occurrences,
            top_M1=self.config['top_M1'],
            top_M2=self.config['top_M2'],
            index=index,
            max_consequent_itemset_size=self.config['max_consequent_itemset_size']
        )
        timings['RuleMiner'] = time.perf_counter() - t0
        print(f"  RuleMiner: {timings['RuleMiner']:.3f}s")

        print("\n[6/6] 生成报告...")
        t0 = time.perf_counter()
        reporter = Reporter()
        result = reporter.rank_and_report(
            rules,
            top_K=self.config['top_K'],
            event_sequence=event_sequence,
            print_top_n=self.config['report_print_top_n'],
            compute_ranking=self.config['report_compute_ranking']
        )
        timings['Report'] = time.perf_counter() - t0
        print(f"  Report: {timings['Report']:.3f}s")

        result['motifs'] = motifs
        result['motif_occurrences'] = motif_occurrences
        result['all_rules'] = rules
        result['regimes'] = regimes
        result['event_sequence'] = event_sequence
        result['event_labels'] = dict(getattr(eventizer, 'event_labels', {}))
        result['eventizer_name'] = eventizer.__class__.__name__
        result['feature_meta'] = feature_meta
        result['timings'] = timings

        return result


# ==================== 示例使用 ====================

if __name__ == "__main__":
    np.random.seed(42)

    T = 3000
    d = 5

    X = np.zeros((T, d))

    t1 = np.arange(1000)
    for i in range(d):
        X[:1000, i] = np.sin(2 * np.pi * t1 / 100 + i) + 0.1 * np.random.randn(1000)

    t2 = np.arange(1000)
    for i in range(d):
        X[1000:2000, i] = np.sin(2 * np.pi * t2 / 20 + i) + t2 / 500 + 0.2 * np.random.randn(1000)

    for i in range(d):
        X[2000:, i] = np.cumsum(0.1 * np.random.randn(1000))

    for t in range(0, 1000, 50):
        if t + 5 < 1000:
            X[t, 0] += 0.5
            X[t + 4, 1] += 0.5

    for t in range(2000, 3000, 50):
        if t + 5 < 3000:
            X[t, 0] += 0.5
            X[t + 4, 1] += 0.5

    print("=" * 60)
    print("非平稳多变量流式序列规则挖掘系统 - 示例运行")
    print("=" * 60)

    miner = NonStationaryRuleMiner(
        n_bins=4,
        seg_window=800,
        max_length=4,
        gap=5,
        delta_min=3,
        delta_max=5,
        top_K=50,
        report_print_top_n=0,
        report_compute_ranking=False
    )

    result = miner.fit(X)

    print("\n" + "=" * 60)
    print("挖掘完成！")
    print("=" * 60)
    print(f"总体摘要: {result['summary']}")
