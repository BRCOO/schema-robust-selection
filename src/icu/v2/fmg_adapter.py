"""FMG-Bridge program discovery on the TRAIN split only (v2 protocol, audit item P0-2).

Why this module exists
----------------------
The ICDE v1 run (``artifacts/results/mimic_iv/fmgbridge_contract_internal_v1``) scored and
selected FMG programs on all 119 discovery episodes. 57 of them are internal TEST episodes
(``docs/experiments/icde_comparator_split_manifest.json``), while the baselines mined on the
62 TRAIN episodes only (``scripts/run_comparator_discovery_pack.py:239-249``). Every locked
program reports ``positive_total=60, control_total=59`` (= all 119; see
``artifacts/results/mimic_iv/contract_quality_complete_v1/contract_quality_summary.json``), and
per the audit ``program_005`` (Lactate -> DBP) fires on no training episode. This module reruns
the *same* v1 contrastive stage on the training episodes only, converts the result to
:class:`ProgramRecord`, and guards future runs against support counts that include non-train
episodes.

v1 call chain (line numbers: working tree of 2026-10-18 = HEAD 4ad4c48 plus uncommitted,
performance-only edits to both v1 files; the CLI records their sha256)
------------------------------------------------------------------------------------------
``rme`` = ``scripts/run_mimic_experiment.py``, ``scm`` = ``src/icu/shock_contrastive_miner.py``.

A. Per-episode rule mining (label-free; produces each episode's ``rules`` list)

1. ``rme:3794-3806``: ``miner_config = _default_miner_config()`` (``rme:1389-1410``: ICU eventizer,
   n_bins=4, eventizer_window=6, max_length=3, gap_max=2, top_K=20, ...) plus ``--miner-*``
   overrides. Episode specs come from ``shock_episode_index.json`` (``rme:66-81``), are filtered
   (``rme:1375``) and limited (``_limit_specs_with_type_balance``, ``rme:84-162``: about half
   controls and half positives, shuffled first when ``--episode-sampling-seed`` is given).
2. ``rme:3880-3927``: each episode's ``.npy`` and missing mask are loaded; too short
   (``min_episode_timesteps``) or low-information episodes (``_is_low_information_episode``,
   ``rme:1463``; called at ``rme:3914``) are skipped. This is the 120 -> 119 drop.
3. ``rme:3928-3929``: ``NonStationaryRuleMiner(**miner_config).fit((X, feature_meta))``
   (``nonstationary_miner.py`` execs ``Non-stationary time series.py``). The eventizer
   (``src/icu/icu_eventizer.py:130-150``) reads only ``feature_names``, ``missing_mask`` and
   the feature-set keys; it never sees ``episode_type``, other episodes, or the split.
4. ``rme:3960-3975``: ``contrastive_inputs.append({"episode_id", "episode_type",
   "anchor_segments": _episode_anchor_segments(X) (rme:1544), "rules": extract_rule_records(...)})``.
   ``scm:387-429`` builds one record per mined rule. ``rule_id`` (``scm:415``) is
   ``_motif_to_label(antecedent) + " => " + _motif_to_label(consequent)``, where itemsets are
   joined by ``" -> "`` (``scm:383-384``) and the tokens of one itemset are joined by ``", "``
   inside braces in event-id order (``scm:378-380``; labels from ``scm:363-375`` /
   ``icu_eventizer``: ``level_k, up, down, missing, sparse_measurement, onset, offset,
   deterioration, recovery``). ``lead_time_steps`` is the mean distance from consequent
   occurrences to the window end (``scm:91-107``).
5. ``rme:3976-3980``: the list is cached as ``contrastive_inputs.json``. ``--reuse-contrastive-inputs``
   (``rme:3851-3865``) reloads it through ``_load_cached_contrastive_inputs`` (``rme:2509-2525``,
   which keeps only ``episode_id``, ``episode_type`` and ``rules``).

B. Contrastive scoring and program selection (uses labels; v1 ran it on ALL 119 episodes)

6. ``rme:5801-5808`` (``main``): ``--profile-json`` keys are copied onto the argparse namespace
   by ``_apply_profile_overrides`` (``rme:252-265``; unknown keys are silently ignored);
   ``exclude_sparse_measurement_dominant`` is derived from two flags. ``rme:4621-4633`` resolves
   modes and clamps calibration / support-confidence knobs; ``rme:4645-4652`` sets the
   effective top-k care-process cap (locked gate profile -> ``algorithm_ready_top5_care_process_max``).
   ``rme:4657-4834`` only records these knobs as ``summary["contrastive_filters"]``.
7. ``rme:4881-4917``: ``scm.score_contrastive_rule_records(contrastive_inputs, **rule_scoring_kwargs)``
   (``scm:432-749``). Totals are counted over every episode passed in (``scm:466-469``),
   per-rule positive / control episode sets over the same episodes (``scm:493-532``), then
   min-count filters (``scm:537-546``), enrichment and score (``scm:548-631``), sort and
   candidate pool (``scm:737-749``). Locked keys: ``min_rule_positive_count=2``,
   ``min_rule_control_count=1``, ``sparse_measurement_penalty=0.55`` (exponent 2),
   ``exclude_missing_only=True``, significance off (weight 0, q-max 1.0).
8. ``rme:4923-4941``: ``_recover_rule_candidates_for_low_mechanism_coverage`` (``rme:2911-3027``;
   rescoring on the same inputs; disabled in the locked profile).
9. ``rme:4943-4976``: consensus boost from cross-run stability caches (none in the locked run).
10. ``rme:4978-5007``: ``_apply_rule_anchor_diversity_rerank`` (``rme:3030-3125``; locked
    weight 0.65, top_k 5, epsilon 0.15, quality floor 0.9).
11. ``rme:5009-5104``: mechanism-diversity rerank (``rme:3128``, gate ``rme:3361``); locked
    weight 0, so v1 only records metadata (the call works on copies).
12. ``rme:5106-5168``: shadow predictive rule rerank (weight 0 -> off). ``rme:5170-5185``:
    statistical evidence and null benchmark (diagnostics only).
13. ``rme:5186-5209``: ``scm.consolidate_contrastive_programs`` (``scm:1171-1302``): groups rules
    by (antecedent feature set, consequent feature set) (``_program_key``, ``scm:752-755``);
    representative = best member (``scm:1224``); ``member_rule_ids`` = top-5 members
    (``scm:1225-1228``); ``representative_score_support`` snapshot (``scm:1239-1247``);
    coarse ``mechanism_groups`` (``scm:1255``); greedy diversity selection (``scm:993-1168``,
    candidate pool ``scm:907-990``) with locked anchor/mechanism knobs; ids ``program_NNN``
    by selection order (``scm:1300-1302``); at most 20 programs (default ``max_programs``).
14. ``rme:5210-5254``: shadow program rerank (off). ``rme:5256-5260``:
    ``_apply_program_mechanism_group_mode`` (``rme:1597-1632``; locked ``coarse_plus_feature``,
    floor 0.85, adds ``"<group>::<feature>"`` tags). ``rme:5261-5270``:
    ``_apply_program_score_calibration`` (``rme:1650``; locked ``none``). ``rme:5271-5278``:
    ``_apply_program_support_confidence_adjustment`` (``rme:1772``; locked weight 0).
    ``rme:5280-5287``: ``_enforce_program_mechanism_coverage_lower_bound`` (``rme:1165``;
    window 20, min groups 7, floor 0.8).
15. ``rme:5289-5340``: ``_select_topk_with_constraints`` (``rme:551``; returns the full list
    with the constrained top window first) and a gate snapshot: ``_summarize_contrastive_qc``
    (``rme:1851``, again over the same inputs) and ``_evaluate_algorithm_ready_gates``
    (``rme:2367``). ``rme:5342-5540``: inline autotune loop (locked: 2 steps) scored with the
    transfer target ``_evaluate_transfer_target_against_baseline`` (``rme:1256``; active only
    when ``transfer_baseline_*`` are set). ``rme:5541``: ``summary["contrastive_programs"]``.

C. Locked profile: ``docs/experiments/strict_profile_icde_locked.json`` merged with
   ``FORMAL_LOCKED_OVERRIDES`` / ``TRANSFER_TOLERANCES`` by ``build_reference_profile``
   (``scripts/run_component_ablation_upgrade.py:25-142``). The paper's full selector adds a
   transfer ("schema-survival") target whose baselines come from a first pass without it
   (``run_component_ablation_upgrade.py:145-160``, ``run_full_ablation_bridge_refit.py:139-162``);
   see :func:`transfer_target_config`.

D. Downstream firing (``scripts/run_contract_quality_complete.py``): ``contrastive_programs[:top_k]``
   (``:517``); a program fires on an episode iff the episode's ``rules`` ids intersect
   ``member_rule_ids`` plus ``representative_rule_id`` (``_fmg_program_rule_sets`` /
   ``_fmg_firings_for_inputs``, ``:257-293``; the rule flags are not consulted).

What is reused and what is re-implemented
-----------------------------------------
``scripts/run_mimic_experiment.py`` is import-safe: at module level it only inserts the repo
root into ``sys.path`` (``rme:13-16``), defines constants and functions, and guards ``main()``
with ``__name__ == "__main__"`` (``rme:6005``). Its imports are side-effect free except one:
``nonstationary_miner`` execs ``Non-stationary time series.py``, which calls
``warnings.filterwarnings('ignore')`` process-wide (line 22; optional ``bitarray`` / ``numba``
imports, neither installed here; its demo is under a ``__main__`` guard). It is therefore
imported lazily inside ``warnings.catch_warnings()``, which restores the filters. No file I/O,
subprocess, or GPU work happens at import (measured: 0.2 s).

All stage functions above are imported and called unchanged, in v1 order, with v1 argument
casts. Only glue that is inline in ``run_mimic_experiment`` is re-implemented: the parameter
plumbing (steps 6-15) and the autotune loop (``rme:5342-5541``). Metadata-only work (step 11
when inactive, step 12 diagnostics, program models, actionability) is skipped because it
cannot change the selected programs. Options that would read anything other than the given
train inputs are rejected (``limit``, stability caches, shadow predictive reranks, input
directories). :func:`mine_fmg_programs_via_v1_entrypoint` runs v1's own ``main`` on the
same train-only inputs and is used to check that the two paths return identical programs.

Re-running per-episode mining
-----------------------------
Step A is per-episode and label-free, so for the *v1 windows* the cached per-episode rules
can be reused and only step B must be restricted to TRAIN (this module). The v2 task changes
the windows (gap before the anchor, anchor-linked features removed, patient-level split of
the full index), so the episodes themselves change and step A must be re-run before this
module is applied to the new task.
"""

from __future__ import annotations

import codecs
import importlib
import json
import math
import warnings
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from src.icu.shock_contrastive_miner import consolidate_contrastive_programs, score_contrastive_rule_records
from src.icu.v2.types import (
    Itemset,
    ProgramRecord,
    feature_of,
    features_of_itemsets,
    make_symbolic_program,
)

METHOD = "fmg_bridge"
RULE_SEPARATOR = " => "
ITEMSET_SEPARATOR = " -> "
FIRING_SEMANTICS = "v1: fires iff episode rule ids intersect member_rule_ids + representative_rule_id"

_V1_EXPERIMENT_MODULE = "scripts.run_mimic_experiment"
_V1_PROFILE_MODULE = "scripts.run_component_ablation_upgrade"
_UNUSED_INPUT = "<train-only contrastive inputs supplied in memory>"
_JSON_WHITESPACE = " \t\n\r"

TRANSFER_BASELINE_KEYS = (
    "transfer_baseline_top5_unique_groups",
    "transfer_baseline_top10_unique_groups",
    "transfer_baseline_top5_score_min",
)

# v1 keys that only act before the contrastive stage (per-episode mining and episode
# sampling, rme:3794-3975). Accepted, but reported as having no effect here.
UPSTREAM_ONLY_KEYS = frozenset(
    {
        "min_core_observed_channels",
        "min_core_observed_points",
        "min_episode_timesteps",
        "miner_top_candidates",
        "miner_top_m1",
        "miner_top_m2",
        "motif_verify_top_k",
        "motif_min_support_ratio",
        "rule_top_k",
        "filter_anchor_kind",
        "filter_window_hours",
        "episode_sampling_seed",
        "episode_sampling_unique_stays",
        "disable_episode_type_balance",
    }
)

# v1 keys whose non-default values would make the contrastive stage read data other than
# the given train inputs, or re-select episodes.
_FORBIDDEN_UNLESS_DEFAULT = {
    "input": "episodes come from the given train inputs, not from an input directory",
    "out_dir": "this adapter does not write v1 summaries",
    "profile_json": "pass the profile mapping itself",
    "reuse_contrastive_inputs": "episodes come from the given train inputs",
    "no_cache_contrastive_inputs": "per-episode mining is not run here",
    "stop_after_contrastive_cache": "per-episode mining is not run here",
    "limit": "episode subsampling (rme:3854-3863) must come from the split manifest",
    "stability_cache_glob": "cross-run caches carry rule rankings from runs on non-train episodes",
    "stability_auto_adaptive_history_glob": "cross-run caches carry rankings from non-train runs",
    "shadow_rule_blend_weight": "fits a model on raw episodes of an input directory (rme:5106-5119)",
    "shadow_predictive_blend_weight": "fits a model on raw episodes of an input directory (rme:5106-5119)",
}


class TrainSupportViolation(ValueError):
    """Raised when program supports were computed with episodes outside the training split."""


@dataclass(frozen=True)
class FmgMiningResult:
    """Output of the v1 contrastive stage on one set of (train) inputs.

    ``programs`` is v1's ``summary["contrastive_programs"]``: the full ranked list with the
    constrained top window first; callers take ``programs[:top_k]``.
    """

    programs: list[dict]
    rule_count: int
    qc: dict
    gates: dict
    selector: dict
    stages: dict
    config: dict


# --------------------------------------------------------------------------- splits / inputs


def load_split_manifest(path: str | Path) -> dict[str, str]:
    """Map episode_id -> split name.

    Accepts the v1 manifest (``{"split": {"train": [...], "val": [...], "test": [...]}}``, as in
    ``docs/experiments/icde_comparator_split_manifest.json``; non-list entries such as
    ``seed`` are skipped) or a JSON list of ``{"episode_id", "split"}`` rows (e.g. dumped
    ``EpisodeRecord`` objects). An episode listed under two different splits raises.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    assignment: dict[str, str] = {}

    def _assign(episode_id: object, split: object) -> None:
        key = str(episode_id).strip()
        name = str(split).strip()
        if not key or not name:
            raise ValueError(f"empty episode id or split name in {path}")
        previous = assignment.setdefault(key, name)
        if previous != name:
            raise ValueError(f"episode {key!r} is assigned to both {previous!r} and {name!r}")

    if isinstance(payload, dict) and isinstance(payload.get("split"), dict):
        splits = payload["split"]
        if not isinstance(splits.get("train"), list):
            raise ValueError(f"split manifest {path} has no 'split.train' list")
        for name, episode_ids in splits.items():
            if isinstance(episode_ids, list):
                for episode_id in episode_ids:
                    _assign(episode_id, name)
    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, Mapping) or "episode_id" not in row or "split" not in row:
                raise ValueError(f"split rows in {path} need 'episode_id' and 'split'")
            _assign(row["episode_id"], row["split"])
    else:
        raise ValueError(f"unrecognized split manifest format: {path}")
    return assignment


def filter_contrastive_inputs(inputs: Sequence[dict], allowed_episode_ids: Collection[str]) -> list[dict]:
    """Keep the inputs whose ``episode_id`` is allowed, in input order (no copies).

    Raises on a duplicated kept episode id, because v1 counts list entries for its totals
    but episode-id sets for its supports (``scm:466-532``).
    """
    if isinstance(allowed_episode_ids, (str, bytes)):
        raise TypeError("allowed_episode_ids must be a collection of ids, not a single string")
    allowed = {str(episode_id) for episode_id in allowed_episode_ids}
    kept: list[dict] = []
    seen: set[str] = set()
    for episode in inputs:
        episode_id = str(episode.get("episode_id", "") or "")
        if episode_id not in allowed:
            continue
        if episode_id in seen:
            raise ValueError(f"duplicate episode_id {episode_id!r} in contrastive inputs")
        seen.add(episode_id)
        kept.append(episode)
    return kept


def iter_json_array(path: str | Path, *, chunk_size: int = 1 << 20, hasher: Any = None) -> Iterator[Any]:
    """Yield the elements of a top-level JSON array without loading the whole file.

    Used for the 280 MB ``contrastive_inputs.json`` so that only the kept (train) episodes
    are held in memory. ``hasher`` (e.g. ``hashlib.sha256()``) receives every raw byte read;
    it covers the whole file once the iterator is exhausted. A UTF-8 BOM is accepted.
    """
    decoder = json.JSONDecoder()
    text_decoder = codecs.getincrementaldecoder("utf-8-sig")()
    with open(path, "rb") as handle:
        buffer = ""
        pos = 0
        eof = False

        def _fill() -> None:
            nonlocal buffer, pos, eof
            raw = handle.read(max(1, int(chunk_size)))
            if raw and hasher is not None:
                hasher.update(raw)
            eof = not raw
            buffer = buffer[pos:] + text_decoder.decode(raw, final=eof)
            pos = 0

        def _skip_whitespace() -> bool:
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos] in _JSON_WHITESPACE:
                    pos += 1
                if pos < len(buffer):
                    return True
                if eof:
                    return False
                _fill()

        if not _skip_whitespace() or buffer[pos] != "[":
            raise ValueError(f"{path} does not contain a top-level JSON array")
        pos += 1
        first = True
        while True:
            if not _skip_whitespace():
                raise ValueError(f"unterminated JSON array in {path}")
            if buffer[pos] == "]":
                pos += 1
                if _skip_whitespace():
                    raise ValueError(f"trailing data after the JSON array in {path}")
                return
            if not first:
                if buffer[pos] != ",":
                    raise ValueError(f"expected ',' between array elements in {path}")
                pos += 1
                if not _skip_whitespace():
                    raise ValueError(f"unterminated JSON array in {path}")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    _fill()
                    continue
                if end >= len(buffer) and not eof:
                    _fill()  # a number or literal may continue in the next chunk
                    continue
                break
            pos = end
            first = False
            yield value


# --------------------------------------------------------------------------- v1 plumbing


def _v1_experiment() -> ModuleType:
    """Import ``scripts/run_mimic_experiment.py`` lazily, restoring global warning filters."""
    with warnings.catch_warnings():
        return importlib.import_module(_V1_EXPERIMENT_MODULE)


def _differs(value: object, default: object) -> bool:
    if isinstance(value, float) and isinstance(default, float) and math.isnan(value) and math.isnan(default):
        return False
    return value != default


def _clip01(value: object) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_profile(miner_config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(miner_config, Mapping):
        raise TypeError("miner_config must be a mapping of v1 profile keys")
    profile = {str(key): value for key, value in miner_config.items()}
    # Some v1 drivers serialise the NaN 'disabled' sentinel as null.
    if "transfer_baseline_top5_score_min" in profile and profile["transfer_baseline_top5_score_min"] is None:
        profile["transfer_baseline_top5_score_min"] = float("nan")
    return profile


def resolve_v1_profile(miner_config: Mapping[str, object]) -> dict[str, Any]:
    """Resolve a profile exactly as ``run_mimic_experiment.py --profile-json`` does.

    Keys are applied with v1's ``_apply_profile_overrides`` onto v1's argparse defaults and
    ``exclude_sparse_measurement_dominant`` is derived like ``main`` (``rme:5801-5808``).
    Unlike v1, unknown keys raise instead of being ignored, and keys that would pull in
    non-train data raise. Returns every v1 parameter value used downstream.
    """
    profile = _normalized_profile(miner_config)
    rme = _v1_experiment()
    parser = rme.build_parser()
    defaults = vars(parser.parse_args(["--input", _UNUSED_INPUT]))
    args = parser.parse_args(["--input", _UNUSED_INPUT])
    report = rme._apply_profile_overrides(args, profile)
    if report["ignored"]:
        raise ValueError(f"keys not accepted by v1 --profile-json: {report['ignored']}")
    resolved = dict(vars(args))
    for key, reason in _FORBIDDEN_UNLESS_DEFAULT.items():
        if _differs(resolved.get(key), defaults.get(key)):
            raise ValueError(f"{key!r} is not supported in train-only mining: {reason}")
    exclude_sparse = bool(resolved["hard_exclude_sparse_measurement_dominant"])
    if bool(resolved["include_sparse_measurement_dominant"]):
        exclude_sparse = False
    resolved["exclude_sparse_measurement_dominant"] = exclude_sparse
    resolved["upstream_only_keys_without_effect"] = sorted(UPSTREAM_ONLY_KEYS.intersection(profile))
    return resolved


def _mechanism_confidence_thresholds(rme: ModuleType, cfg: Mapping[str, Any], domain: str) -> dict:
    """v1 rme:3839-3849 (the domain is inferred from the input directory name there)."""
    threshold_map = rme._load_rule_mechanism_confidence_domain_thresholds(
        cfg["rule_mechanism_diversity_confidence_domain_thresholds_path"]
    )
    return rme._resolve_rule_mechanism_confidence_thresholds(
        domain=rme._normalize_adaptive_domain(domain),
        domain_thresholds=threshold_map,
        default_min_support_ratio=float(cfg["rule_mechanism_diversity_confidence_min_support_ratio"]),
        default_min_reliable_rules=int(cfg["rule_mechanism_diversity_confidence_min_reliable_rules"]),
        default_min_expected_gain=int(cfg["rule_mechanism_diversity_confidence_min_expected_gain"]),
    )


def _rule_scoring_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """v1 rme:4881-4913 (values passed through uncast, as v1 does)."""
    return {
        "exclude_missing_only": True,
        "exclude_sparse_measurement_dominant": cfg["exclude_sparse_measurement_dominant"],
        "sparse_measurement_penalty": cfg["sparse_measurement_penalty"],
        "sparse_measurement_penalty_exponent": cfg["sparse_measurement_penalty_exponent"],
        "rule_anchor_saturation_penalty_weight": cfg["rule_anchor_saturation_penalty_weight"],
        "rule_anchor_saturation_penalty_exponent": cfg["rule_anchor_saturation_penalty_exponent"],
        "rule_mechanism_saturation_penalty_weight": cfg["rule_mechanism_saturation_penalty_weight"],
        "rule_mechanism_saturation_penalty_exponent": cfg["rule_mechanism_saturation_penalty_exponent"],
        "lead_time_quality_weight": cfg["rule_leadtime_quality_weight"],
        "rule_support_reliability_weight": cfg["rule_support_reliability_weight"],
        "rule_support_reliability_exponent": cfg["rule_support_reliability_exponent"],
        "min_positive_count": cfg["min_rule_positive_count"],
        "min_control_count": cfg["min_rule_control_count"],
        "allow_zero_control_candidates": cfg["allow_zero_control_candidates"],
        "enrichment_rate_prior_count": cfg["enrichment_rate_prior_count"],
        "zero_control_rule_penalty": cfg["zero_control_rule_penalty"],
        "min_contrastive_score": cfg["min_rule_contrastive_score"],
        "rule_anchor_candidate_top_n": cfg["rule_anchor_candidate_top_n"],
        "rule_mechanism_candidate_top_n": cfg["rule_mechanism_candidate_top_n"],
        "rule_candidate_quality_floor_ratio": cfg["rule_candidate_quality_floor_ratio"],
        "rule_care_process_max_ratio": cfg["rule_care_process_max_ratio"],
        "rule_care_process_penalty": cfg["rule_care_process_penalty"],
        "rule_anchor_min_unique": cfg["rule_anchor_min_unique"],
        "rule_anchor_max_share": cfg["rule_anchor_max_share"],
        "rule_mechanism_min_unique": cfg["rule_mechanism_min_unique"],
        "significance_weight": cfg["rule_significance_weight"],
        "significance_prior_count": cfg["rule_significance_prior_count"],
        "significance_bootstrap_samples": cfg["rule_significance_bootstrap_samples"],
        "significance_random_state": cfg["rule_significance_random_state"],
        "significance_max_q_value": cfg["rule_significance_max_q_value"],
        "return_candidate_pool_diagnostics": True,
    }


def run_fmg_contrastive_stage(
    train_inputs: Sequence[dict],
    miner_config: Mapping[str, object],
    *,
    domain: str = "mimic_iv",
) -> FmgMiningResult:
    """Run v1 steps 6-15 (module docstring) on ``train_inputs`` only.

    ``domain`` replaces v1's inference from the input directory name (``.../mimic-iv`` ->
    ``mimic_iv``); it matters only with a mechanism-confidence thresholds file.
    """
    rme = _v1_experiment()
    cfg = resolve_v1_profile(miner_config)
    inputs = list(train_inputs)
    if not inputs:
        raise ValueError("no training inputs: v1 skips the contrastive stage for an empty input list")

    # rme:4621-4633
    group_mode = rme._resolve_program_mechanism_group_mode(cfg["program_mechanism_group_mode"])
    feature_floor = _clip01(cfg["program_mechanism_feature_score_floor_ratio"])
    calibration_mode = rme._resolve_program_score_calibration_mode(cfg["program_score_calibration_mode"])
    calibration_q_low = _clip01(cfg["program_score_calibration_quantile_low"])
    calibration_q_high = _clip01(cfg["program_score_calibration_quantile_high"])
    calibration_blend = _clip01(cfg["program_score_calibration_blend_weight"])
    support_weight = _clip01(cfg["program_support_confidence_weight"])
    support_z = max(0.0, float(cfg["program_support_confidence_z_score"]))
    support_floor = _clip01(cfg["program_support_confidence_floor"])
    support_exponent = max(1e-6, float(cfg["program_support_confidence_exponent"]))
    # rme:4645-4652
    care_max = int(cfg["program_topk_care_process_max_count"])
    if care_max < 0 and str(cfg["algorithm_ready_gate_profile"] or "").strip().lower() == "locked":
        care_max = int(cfg["algorithm_ready_top5_care_process_max"])
    care_window = max(1, int(cfg["program_topk_care_process_window"]))

    # rme:4881-4941
    scoring_kwargs = _rule_scoring_kwargs(cfg)
    scored = score_contrastive_rule_records(inputs, **scoring_kwargs)
    rules, rule_pool = scored if isinstance(scored, tuple) else (list(scored), {})
    rules, rule_pool, recovery_meta = rme._recover_rule_candidates_for_low_mechanism_coverage(
        contrastive_inputs=inputs,
        ranked_rules=rules,
        candidate_pool_diagnostics=rule_pool,
        scoring_kwargs=scoring_kwargs,
        recovery_enabled=bool(cfg["rule_low_coverage_recovery_enabled"]),
        trigger_top_k=int(cfg["rule_low_coverage_recovery_trigger_top_k"]),
        min_unique_groups=int(cfg["rule_low_coverage_recovery_min_unique_groups"]),
        recovery_min_control_count=int(cfg["rule_low_coverage_recovery_min_control_count"]),
        recovery_allow_zero_control_candidates=bool(cfg["rule_low_coverage_recovery_allow_zero_control_candidates"]),
        recovery_zero_control_penalty=float(cfg["rule_low_coverage_recovery_zero_control_penalty"]),
        recovery_support_reliability_weight=float(cfg["rule_low_coverage_recovery_support_reliability_weight"]),
        recovery_support_reliability_exponent=float(cfg["rule_low_coverage_recovery_support_reliability_exponent"]),
        recovery_significance_max_q_value=float(cfg["rule_low_coverage_recovery_significance_max_q_value"]),
    )
    # rme:4943-4976 (consensus) needs stability caches, which resolve_v1_profile rejects.

    # rme:4978-5007
    anchor_weight = max(0.0, float(cfg["rule_anchor_diversity_weight"]))
    if anchor_weight > 0.0:
        rules = rme._apply_rule_anchor_diversity_rerank(
            rules,
            diversity_weight=anchor_weight,
            top_k=max(1, int(cfg["rule_anchor_diversity_top_k"])),
            rerank_epsilon=max(0.0, float(cfg["rule_anchor_diversity_rerank_epsilon"])),
            quality_floor_ratio=_clip01(cfg["rule_anchor_diversity_quality_floor_ratio"]),
        )

    # rme:5009-5104 (when inactive v1 only records metadata, computed on copies)
    mechanism_weight = max(0.0, float(cfg["rule_mechanism_diversity_weight"]))
    mechanism_top_k = max(1, int(cfg["rule_mechanism_diversity_top_k"]))
    mechanism_epsilon = max(0.0, float(cfg["rule_mechanism_diversity_rerank_epsilon"]))
    mechanism_floor = _clip01(cfg["rule_mechanism_diversity_quality_floor_ratio"])
    mechanism_target = max(0, int(cfg["rule_mechanism_diversity_target"]))
    mechanism_meta: dict = {"enabled": False}
    if mechanism_weight > 0.0:
        allowed = True
        if bool(cfg["rule_mechanism_diversity_confidence_gate_enabled"]):
            thresholds = _mechanism_confidence_thresholds(rme, cfg, domain)
            gate = rme._evaluate_rule_mechanism_diversity_confidence_gate(
                rules,
                top_k=mechanism_top_k,
                rerank_epsilon=mechanism_epsilon,
                quality_floor_ratio=mechanism_floor,
                coverage_target=mechanism_target,
                min_reliable_support_ratio=float(
                    thresholds.get("min_support_ratio", cfg["rule_mechanism_diversity_confidence_min_support_ratio"])
                ),
                min_reliable_rules=int(
                    thresholds.get("min_reliable_rules", cfg["rule_mechanism_diversity_confidence_min_reliable_rules"])
                ),
                min_expected_coverage_gain=int(
                    thresholds.get("min_expected_gain", cfg["rule_mechanism_diversity_confidence_min_expected_gain"])
                ),
            )
            allowed = bool(gate.get("enabled", False))
            mechanism_meta = {"enabled": False, "confidence_gate": dict(gate)}
        if allowed:
            rules, rerank_meta = rme._apply_rule_mechanism_diversity_rerank(
                rules,
                diversity_weight=mechanism_weight,
                top_k=mechanism_top_k,
                rerank_epsilon=mechanism_epsilon,
                quality_floor_ratio=mechanism_floor,
                coverage_target=mechanism_target,
                two_stage_swap_budget=cfg["rule_mechanism_diversity_two_stage_swap_budget"],
                stage1_min_support_ratio=cfg["rule_mechanism_diversity_stage1_min_support_ratio"],
                return_metadata=True,
            )
            mechanism_meta = {"enabled": True, **dict(rerank_meta)}
    # rme:5106-5185: shadow reranks are rejected by resolve_v1_profile; the rest is diagnostics.

    # rme:5186-5209
    programs = consolidate_contrastive_programs(
        rules,
        lead_time_diversity_weight=cfg["program_leadtime_diversity_weight"],
        lead_time_diversity_gap_hours=cfg["program_leadtime_diversity_gap_hours"],
        program_mechanism_diversity_weight=cfg["program_mechanism_diversity_weight"],
        program_mechanism_max_same_count=cfg["program_mechanism_max_same_count"],
        program_care_process_penalty=cfg["program_care_process_penalty"],
        program_redundancy_penalty_weight=cfg["program_redundancy_penalty_weight"],
        program_coverage_target=cfg["program_coverage_target"],
        program_coverage_target_weight=cfg["program_coverage_target_weight"],
        program_anchor_diversity_weight=cfg["program_anchor_diversity_weight"],
        program_anchor_candidate_top_n=cfg["program_anchor_candidate_top_n"],
        program_anchor_candidate_quality_ratio=cfg["program_anchor_candidate_quality_ratio"],
        program_mechanism_candidate_top_n=cfg["program_mechanism_candidate_top_n"],
        program_mechanism_min_unique=cfg["program_mechanism_min_unique"],
        program_anchor_max_same_count=cfg["program_anchor_max_same_count"],
        program_anchor_novelty_target=cfg["program_anchor_novelty_target"],
        program_anchor_novelty_weight=cfg["program_anchor_novelty_weight"],
        program_anchor_novelty_rerank_epsilon=cfg["program_anchor_novelty_rerank_epsilon"],
        program_diversity_quality_floor_ratio=cfg["program_diversity_quality_floor_ratio"],
        program_topk_care_process_max_count=care_max,
        program_topk_care_process_window=care_window,
        min_program_representative_score=cfg["min_program_representative_score"],
    )
    # rme:5256-5287
    programs = rme._apply_program_mechanism_group_mode(
        programs, mode=str(group_mode), feature_score_floor_ratio=float(feature_floor)
    )
    programs, calibration_meta = rme._apply_program_score_calibration(
        programs,
        mode=str(calibration_mode),
        quantile_low=float(calibration_q_low),
        quantile_high=float(calibration_q_high),
        target_low=float(cfg["program_score_calibration_target_low"]),
        target_high=float(cfg["program_score_calibration_target_high"]),
        blend_weight=float(calibration_blend),
    )
    programs, support_meta = rme._apply_program_support_confidence_adjustment(
        programs,
        weight=float(support_weight),
        z_score=float(support_z),
        floor=float(support_floor),
        exponent=float(support_exponent),
    )
    programs, preselect_meta = rme._enforce_program_mechanism_coverage_lower_bound(
        programs,
        window_size=int(cfg["program_preselect_coverage_window"]),
        min_unique_groups=int(cfg["program_preselect_coverage_min_unique_groups"]),
        quality_floor_ratio=float(cfg["program_preselect_quality_floor_ratio"]),
    )

    def _gate_snapshot(ranked: list[dict]) -> tuple[dict, dict]:  # rme:5289-5316
        qc = rme._summarize_contrastive_qc(inputs, ranked, top_k=5)
        gates = rme._evaluate_algorithm_ready_gates(
            {"contrastive_rules": rules, "contrastive_programs": ranked, "contrastive_qc": qc},
            gate_profile=cfg["algorithm_ready_gate_profile"],
            baseline_rules_count=cfg["algorithm_ready_baseline_rules_count"],
            baseline_programs_count=cfg["algorithm_ready_baseline_programs_count"],
            rule_count_min_ratio=cfg["algorithm_ready_rule_count_min_ratio"],
            program_count_min_ratio=cfg["algorithm_ready_program_count_min_ratio"],
            rule_count_min_abs=cfg["algorithm_ready_rule_count_min_abs"],
            program_count_min_abs=cfg["algorithm_ready_program_count_min_abs"],
            top5_care_process_max=cfg["algorithm_ready_top5_care_process_max"],
            top5_max_same_anchor_max=cfg["algorithm_ready_top5_max_same_anchor_max"],
            top5_unique_anchor_min=cfg["algorithm_ready_top5_unique_anchor_min"],
            top5_unique_groups_min=cfg["algorithm_ready_top5_unique_groups_min"],
            top10_unique_groups_min=cfg["algorithm_ready_top10_unique_groups_min"],
            top5_score_min_threshold=cfg["algorithm_ready_top5_score_min_threshold"],
        )
        return qc, gates

    selector_kwargs = {  # rme:5318-5330
        "top_window_size": int(cfg["program_top_window_size"]),
        "top_window_unique_groups_target": int(cfg["program_top_window_unique_groups_target"]),
        "top_window_fixed_prefix": int(cfg["program_top_window_fixed_prefix"]),
        "top5_unique_groups_target": int(cfg["program_top5_unique_groups_target"]),
        "top5_fixed_prefix": int(cfg["program_top5_fixed_prefix"]),
        "topk_care_process_window": int(care_window),
        "topk_care_process_max_count": int(care_max),
        "top5_max_same_anchor_count": int(cfg["algorithm_ready_top5_max_same_anchor_max"]),
        "candidate_pool_size": int(cfg["program_constrained_candidate_pool_size"]),
        "quality_floor_ratio": float(cfg["program_constrained_quality_floor_ratio"]),
        "two_stage_swap_budget": int(cfg["program_constrained_two_stage_swap_budget"]),
    }
    best_programs, best_meta = rme._select_topk_with_constraints(programs, **selector_kwargs)
    best_qc, best_gates = _gate_snapshot(best_programs)
    best_programs, best_meta, best_qc, best_gates = _autotune_constrained_selection(
        rme, cfg, programs, selector_kwargs, _gate_snapshot, best_programs, best_meta, best_qc, best_gates
    )
    return FmgMiningResult(
        programs=list(best_programs),
        rule_count=len(rules),
        qc=best_qc,
        gates=best_gates,
        selector=best_meta,
        stages={
            "rule_candidate_pool": rule_pool,
            "rule_low_coverage_recovery": recovery_meta,
            "rule_mechanism_diversity_rerank": mechanism_meta,
            "program_score_calibration": calibration_meta,
            "program_support_confidence_adjustment": support_meta,
            "program_preselect_coverage": preselect_meta,
        },
        config=cfg,
    )


def _autotune_constrained_selection(
    rme: ModuleType,
    cfg: Mapping[str, Any],
    programs: list[dict],
    selector_kwargs: Mapping[str, Any],
    gate_snapshot: Callable[[list[dict]], tuple[dict, dict]],
    best_programs: list[dict],
    best_meta: dict,
    best_qc: dict,
    best_gates: dict,
) -> tuple[list[dict], dict, dict, dict]:
    """Port of the inline autotune loop of ``run_mimic_experiment`` (rme:5342-5541)."""
    transfer_kwargs = {
        "baseline_top5_unique_groups": int(cfg["transfer_baseline_top5_unique_groups"]),
        "baseline_top10_unique_groups": int(cfg["transfer_baseline_top10_unique_groups"]),
        "baseline_top5_score_min": float(cfg["transfer_baseline_top5_score_min"]),
        "max_drop_top5_unique_groups": int(cfg["transfer_max_drop_top5_unique_groups"]),
        "max_drop_top10_unique_groups": int(cfg["transfer_max_drop_top10_unique_groups"]),
        "max_drop_top5_score_min": float(cfg["transfer_max_drop_top5_score_min"]),
    }

    def _transfer(qc: dict) -> dict:
        diversity = dict(qc.get("mechanism_diversity", {}))
        return rme._evaluate_transfer_target_against_baseline(diversity, **transfer_kwargs)

    def _candidate_score(qc: dict, gates: dict, selector: dict) -> tuple:  # rme:5342-5371
        mechanism = dict(qc.get("mechanism_diversity", {}))
        checks = dict(dict(gates.get("quality", {})).get("checks", {}))
        target = _transfer(qc)
        return (
            int(bool(target.get("pass", False))) if bool(target.get("enabled", False)) else -1,
            int(bool(gates.get("overall_pass", False))),
            int(bool(selector.get("all_constraints_pass", False))),
            int(sum(1 for check in checks.values() if bool(check.get("pass", False)))),
            int(target.get("delta_top10_unique_groups", 0)),
            int(target.get("delta_top5_unique_groups", 0)),
            float(target.get("delta_top5_score_min", 0.0)),
            int(mechanism.get("top5_unique_groups", 0)),
            int(mechanism.get("top10_unique_groups", 0)),
            -int(mechanism.get("top5_care_process_count", 0)),
            -int(mechanism.get("top5_max_same_anchor_count", 0)),
            float(mechanism.get("top5_score_min", 0.0)),
        )

    steps = max(0, int(cfg["program_constraint_autotune_steps"]))
    group_step = max(1, int(cfg["program_constraint_autotune_group_step"]))
    pool_step = max(1, int(cfg["program_constraint_autotune_pool_step"]))
    initial_target = _transfer(best_qc)
    trace: list[dict] = []
    should_autotune = steps > 0 and (
        (not bool(best_gates.get("overall_pass", False)))
        or (bool(initial_target.get("enabled", False)) and not bool(initial_target.get("pass", False)))
    )
    if should_autotune:
        base_kwargs = dict(selector_kwargs)
        for tune_round in range(steps):
            failure = dict(best_gates.get("failure_attribution", {}))
            failed = {
                str(name)
                for key in ("quality_failures", "reference_count_failures")
                for name in list(failure.get(key, []))
            }
            current = _transfer(best_qc)
            if bool(current.get("enabled", False)) and not bool(current.get("pass", False)):
                if int(current.get("delta_top10_unique_groups", 0)) < 0:
                    failed.add("transfer_top10_drop")
                if int(current.get("delta_top5_unique_groups", 0)) < 0:
                    failed.add("transfer_top5_drop")
                if float(current.get("delta_top5_score_min", 0.0)) < 0.0:
                    failed.add("transfer_top5_score_drop")
            trials: list[tuple[str, dict]] = []
            if "top5_unique_groups_ge_min" in failed or "transfer_top5_drop" in failed:
                trial = dict(base_kwargs)
                trial["top5_unique_groups_target"] = int(trial["top5_unique_groups_target"]) + group_step
                trials.append(("raise_top5_groups_target", trial))
            if (
                "top10_unique_groups_ge_min" in failed
                or "top5_unique_groups_ge_min" in failed
                or "transfer_top10_drop" in failed
            ):
                trial = dict(base_kwargs)
                trial["top_window_unique_groups_target"] = int(trial["top_window_unique_groups_target"]) + group_step
                trial["candidate_pool_size"] = max(
                    int(trial.get("candidate_pool_size", 0)),
                    int(base_kwargs.get("top_window_size", 10)) + pool_step,
                )
                trials.append(("raise_top_window_groups_target", trial))
            if "transfer_top10_drop" in failed:
                trial = dict(base_kwargs)
                trial["quality_floor_ratio"] = max(0.0, float(trial.get("quality_floor_ratio", 0.0)) - 0.15)
                trial["candidate_pool_size"] = int(trial.get("candidate_pool_size", 0)) + (pool_step * 4)
                trial["two_stage_swap_budget"] = int(trial.get("two_stage_swap_budget", 0)) + 40
                trials.append(("relax_quality_floor_for_top10", trial))
            if "transfer_top5_score_drop" in failed:
                trial = dict(base_kwargs)
                trial["quality_floor_ratio"] = min(1.0, float(trial.get("quality_floor_ratio", 0.0)) + 0.05)
                trials.append(("raise_quality_floor_ratio", trial))
            trial = dict(base_kwargs)
            trial["candidate_pool_size"] = int(trial.get("candidate_pool_size", 0)) + pool_step
            trials.append(("expand_candidate_pool", trial))
            trial = dict(base_kwargs)
            trial["two_stage_swap_budget"] = int(trial.get("two_stage_swap_budget", 0)) + 10
            trials.append(("increase_swap_budget", trial))

            best_trial = None
            best_trial_score = None
            for trial_name, trial_kwargs in trials:
                trial_programs, trial_meta = rme._select_topk_with_constraints(programs, **trial_kwargs)
                trial_qc, trial_gates = gate_snapshot(trial_programs)
                trial_score = _candidate_score(trial_qc, trial_gates, trial_meta)
                trace.append(
                    {
                        "round": int(tune_round + 1),
                        "trial": trial_name,
                        "params": dict(trial_kwargs),
                        "score": list(trial_score),
                        "overall_pass": bool(trial_gates.get("overall_pass", False)),
                        "selector_all_constraints_pass": bool(trial_meta.get("all_constraints_pass", False)),
                        "transfer_target": _transfer(trial_qc),
                    }
                )
                if best_trial_score is None or trial_score > best_trial_score:
                    best_trial_score = trial_score
                    best_trial = (trial_programs, trial_meta, trial_qc, trial_gates, trial_kwargs)
            if best_trial is None:
                break
            if best_trial_score is not None and best_trial_score > _candidate_score(best_qc, best_gates, best_meta):
                best_programs, best_meta, best_qc, best_gates, accepted = best_trial
                base_kwargs = dict(accepted)
            else:
                break
            latest = _transfer(best_qc)
            if bool(best_gates.get("overall_pass", False)) and (
                (not bool(latest.get("enabled", False))) or bool(latest.get("pass", False))
            ):
                break

    best_meta["autotune"] = {  # rme:5502-5539
        "enabled": True if trace else bool(steps > 0),
        "steps_requested": int(steps),
        "group_step": int(group_step),
        "pool_step": int(pool_step),
        "attempts": trace,
        "final_overall_pass": bool(best_gates.get("overall_pass", False)),
        "final_transfer_target": _transfer(best_qc),
    }
    return best_programs, best_meta, best_qc, best_gates


def mine_fmg_programs(train_inputs: Sequence[dict], miner_config: Mapping[str, object]) -> list[dict]:
    """Return v1's ranked ``contrastive_programs`` computed on ``train_inputs`` only.

    ``miner_config`` holds v1 ``--profile-json`` keys (e.g. the locked profile); see
    :func:`resolve_v1_profile`. Callers keep ``programs[:top_k]``.
    """
    return run_fmg_contrastive_stage(train_inputs, miner_config).programs


def without_transfer_target(miner_config: Mapping[str, object]) -> dict[str, object]:
    """First-pass profile: disable the transfer target the way ``build_reference_profile`` does."""
    config = _normalized_profile(miner_config)
    config.update(dict(zip(TRANSFER_BASELINE_KEYS, (-1, -1, float("nan")))))
    return config


def v1_transfer_tolerances() -> dict[str, object]:
    """v1 ``TRANSFER_TOLERANCES`` (``scripts/run_component_ablation_upgrade.py:107-111``)."""
    module = importlib.import_module(_V1_PROFILE_MODULE)
    return dict(module.TRANSFER_TOLERANCES)


def v1_locked_reference_profile() -> dict[str, object]:
    """v1 ``build_reference_profile()``: locked JSON + FORMAL_LOCKED_OVERRIDES + tolerances."""
    module = importlib.import_module(_V1_PROFILE_MODULE)
    return dict(module.build_reference_profile())


def transfer_target_config(
    miner_config: Mapping[str, object],
    first_pass: FmgMiningResult,
    tolerances: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Second-pass profile of v1's transfer-aware selector.

    Mirrors ``run_component_ablation_upgrade.py:145-160``: the baselines are the first pass's
    final ``contrastive_qc.mechanism_diversity`` top-5 / top-10 unique groups and top-5 score
    floor; tolerances default to v1 ``TRANSFER_TOLERANCES``.
    """
    diversity = dict(first_pass.qc.get("mechanism_diversity", {}) or {})
    config = _normalized_profile(miner_config)
    config.update(
        {
            "transfer_baseline_top5_unique_groups": int(diversity.get("top5_unique_groups", 0) or 0),
            "transfer_baseline_top10_unique_groups": int(diversity.get("top10_unique_groups", 0) or 0),
            "transfer_baseline_top5_score_min": float(diversity.get("top5_score_min", 0.0) or 0.0),
        }
    )
    config.update(dict(v1_transfer_tolerances() if tolerances is None else tolerances))
    return config


def mine_fmg_programs_via_v1_entrypoint(
    train_inputs: Sequence[dict],
    miner_config: Mapping[str, object],
    work_dir: str | Path,
) -> list[dict]:
    """Reference path: v1's own CLI ``main`` with ``--profile-json`` and ``--reuse-contrastive-inputs``.

    Writes a train-only cache, the profile and v1's summary under ``work_dir``; v1 also runs
    ``git rev-parse HEAD`` and fingerprints its (empty) input directory. Used to check that
    :func:`mine_fmg_programs` reproduces v1 exactly.
    """
    resolve_v1_profile(miner_config)
    rme = _v1_experiment()
    work = Path(work_dir)
    input_dir = work / "input" / "mimic-iv"  # v1 infers the domain "mimic_iv" from this name
    input_dir.mkdir(parents=True, exist_ok=True)
    cache_path = work / "train_contrastive_inputs.json"
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "episode_id": episode.get("episode_id"),
                    "episode_type": episode.get("episode_type"),
                    "rules": list(episode.get("rules", []) or []),
                }
                for episode in train_inputs
            ],
            handle,
        )
    profile_path = work / "profile.json"
    profile_path.write_text(json.dumps(_normalized_profile(miner_config)), encoding="utf-8")
    summary = rme.main(
        [
            "--input",
            str(input_dir),
            "--out-dir",
            str(work / "v1_output"),
            "--profile-json",
            str(profile_path),
            "--reuse-contrastive-inputs",
            str(cache_path),
        ]
    )
    return list(summary.get("contrastive_programs", []) or [])


# --------------------------------------------------------------------------- records


def parse_fmg_rule_id(rule_id: str) -> tuple[tuple[Itemset, ...], tuple[Itemset, ...]]:
    """Parse ``"{A:x, B:y} -> {C:z} => {D:w}"`` into (antecedent, consequent) itemsets.

    Itemsets are ``{...}`` groups joined by ``" -> "``; one ``" => "`` separates antecedent
    from consequent; tokens inside a group are comma separated (``scm:378-415``).
    """
    text = str(rule_id)
    sides: list[list[Itemset]] = [[]]
    pos = 0
    while True:
        if not text.startswith("{", pos):
            raise ValueError(f"malformed FMG rule id {rule_id!r}: expected '{{' at offset {pos}")
        close = text.find("}", pos + 1)
        body = text[pos + 1 : close] if close >= 0 else ""
        if close < 0 or "{" in body:
            raise ValueError(f"malformed FMG rule id {rule_id!r}: unbalanced braces")
        tokens = [token.strip() for token in body.split(",")]
        if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
            raise ValueError(f"malformed FMG rule id {rule_id!r}: empty or repeated token in {{{body}}}")
        sides[-1].append(frozenset(tokens))
        pos = close + 1
        if pos == len(text):
            break
        if text.startswith(ITEMSET_SEPARATOR, pos):
            pos += len(ITEMSET_SEPARATOR)
        elif text.startswith(RULE_SEPARATOR, pos) and len(sides) == 1:
            sides.append([])
            pos += len(RULE_SEPARATOR)
        else:
            raise ValueError(f"malformed FMG rule id {rule_id!r}: unexpected text at offset {pos}")
    if len(sides) != 2 or not sides[1]:
        raise ValueError(f"malformed FMG rule id {rule_id!r}: missing '{RULE_SEPARATOR.strip()}' consequent")
    return tuple(sides[0]), tuple(sides[1])


def format_fmg_rule_id(antecedent: Sequence[Itemset], consequent: Sequence[Itemset]) -> str:
    """Inverse of :func:`parse_fmg_rule_id` with tokens sorted inside each itemset.

    v1 orders tokens by event id, so ``format(parse(s)) == s`` only when each group of ``s``
    is sorted (always true for one-token groups); ``parse(format(x)) == x`` always holds.
    """
    if not antecedent or not consequent or any(not itemset for itemset in (*antecedent, *consequent)):
        raise ValueError("a rule needs non-empty antecedent and consequent itemsets")

    def _side(itemsets: Sequence[Itemset]) -> str:
        return ITEMSET_SEPARATOR.join("{" + ", ".join(sorted(itemset)) + "}" for itemset in itemsets)

    return _side(antecedent) + RULE_SEPARATOR + _side(consequent)


def _rule_feature_signature(rule_id: str) -> tuple[frozenset[str], frozenset[str]]:
    antecedent, consequent = parse_fmg_rule_id(rule_id)
    return features_of_itemsets(antecedent), features_of_itemsets(consequent)


def _program_id(program: Mapping[str, Any], rank: int) -> str:
    # Same fallback as run_contract_quality_complete._fmg_program_rule_sets (:259-260).
    return str(program.get("program_id") or f"fmg_program_{rank:03d}")


def fmg_programs_to_records(programs: Sequence[Mapping[str, Any]], method: str = METHOD) -> list[ProgramRecord]:
    """Convert v1 program dicts into ``ordered_rule`` records (order and v1 ids kept).

    ``elements`` / ``consequent`` come from ``representative_rule_id``. Every member rule must
    have the representative's feature signature (v1's ``_program_key``) and the v1
    ``antecedent_features`` / ``consequent_features``, otherwise a ValueError is raised.
    ``params`` keeps the v1 ids, supports, lead times and ``mechanism_groups``.
    """
    records: list[ProgramRecord] = []
    seen: set[str] = set()
    for rank, program in enumerate(programs, start=1):
        program_id = _program_id(program, rank)
        if program_id in seen:
            raise ValueError(f"duplicate program_id {program_id!r}")
        seen.add(program_id)
        representative = str(program.get("representative_rule_id", "") or "")
        if not representative:
            raise ValueError(f"program {program_id!r} has no representative_rule_id")
        elements, consequent = parse_fmg_rule_id(representative)
        signature = (features_of_itemsets(elements), features_of_itemsets(consequent))
        v1_signature = (
            frozenset(feature_of(str(f)) for f in program.get("antecedent_features", ()) or ()),
            frozenset(feature_of(str(f)) for f in program.get("consequent_features", ()) or ()),
        )
        if ("antecedent_features" in program or "consequent_features" in program) and v1_signature != signature:
            raise ValueError(f"program {program_id!r}: representative rule features differ from v1 feature lists")
        members = tuple(str(rule_id) for rule_id in program.get("member_rule_ids", ()) or () if str(rule_id))
        for member in members:
            if _rule_feature_signature(member) != signature:
                raise ValueError(f"program {program_id!r}: member {member!r} has a different feature signature")
        support = dict(program.get("representative_score_support", {}) or {})
        params = {
            "source": "fmg_bridge_v1_contrastive_programs",
            "rank": rank,
            "program_label": program.get("program_label"),
            "representative_rule_id": representative,
            "member_rule_ids": members,
            "firing_semantics": FIRING_SEMANTICS,
            "supports": support,
            "lead_time_hours_mean": float(program.get("lead_time_hours_mean", 0.0) or 0.0),
            "lead_time_steps_mean": float(program.get("lead_time_steps_mean", 0.0) or 0.0),
            "v1_mechanism_groups": tuple(str(g) for g in program.get("mechanism_groups", ()) or ()),
            "v1_antecedent_features": tuple(str(f) for f in program.get("antecedent_features", ()) or ()),
            "v1_consequent_features": tuple(str(f) for f in program.get("consequent_features", ()) or ()),
            "rule_count": int(program.get("rule_count", 0) or 0),
        }
        records.append(
            make_symbolic_program(
                program_id,
                method,
                "ordered_rule",
                elements,
                consequent,
                score=float(program.get("representative_score", 0.0) or 0.0),
                params=params,
            )
        )
    return records


def program_record_to_json(record: ProgramRecord) -> dict[str, Any]:
    """JSON-ready view of a record (itemsets and sets as sorted lists, tuples as lists)."""

    def _plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): _plain(item) for key, item in value.items()}
        if isinstance(value, (frozenset, set)):
            return sorted(_plain(item) for item in value)
        if isinstance(value, (list, tuple)):
            return [_plain(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return {
        "program_id": record.program_id,
        "method": record.method,
        "kind": record.kind,
        "elements": [sorted(itemset) for itemset in record.elements],
        "consequent": [sorted(itemset) for itemset in record.consequent],
        "features": sorted(record.features),
        "anchor_signature": sorted(record.anchor_signature),
        "score": _plain(float(record.score)),
        "params": _plain(record.params),
    }


# --------------------------------------------------------------------------- firing / guard


def _v1_firing_rule_ids(record: ProgramRecord) -> frozenset[str]:
    members = {str(rule_id) for rule_id in record.params.get("member_rule_ids", ()) or () if str(rule_id)}
    representative = str(record.params.get("representative_rule_id", "") or "")
    if representative:
        members.add(representative)
    if not members:
        raise ValueError(f"record {record.program_id!r} has no v1 rule ids; is it an FMG record?")
    return frozenset(members)


def fmg_firing_sets(
    records: Sequence[ProgramRecord],
    inputs_by_episode: Mapping[str, dict],
) -> dict[str, frozenset[str]]:
    """Map program_id -> ids of the episodes where it fires, with v1 semantics.

    A program fires on an episode iff the ids in the episode's ``rules`` list intersect
    ``member_rule_ids`` plus ``representative_rule_id`` (``run_contract_quality_complete.py:257-293``;
    rule flags such as ``is_missing_only`` are not consulted). Keys of ``inputs_by_episode``
    are the episode ids. Same orientation as ``patterns.firing_sets``.
    """
    programs_by_rule: dict[str, list[str]] = {}
    fired: dict[str, set[str]] = {}
    for record in records:
        if record.program_id in fired:
            raise ValueError(f"duplicate program_id {record.program_id!r}")
        fired[record.program_id] = set()
        for rule_id in _v1_firing_rule_ids(record):
            programs_by_rule.setdefault(rule_id, []).append(record.program_id)
    for episode_id, episode in inputs_by_episode.items():
        for rule in episode.get("rules", []) or []:
            for program_id in programs_by_rule.get(str(rule.get("rule_id", "") or ""), ()):
                fired[program_id].add(str(episode_id))
    return {program_id: frozenset(episode_ids) for program_id, episode_ids in fired.items()}


def _support_view(program: Any, rank: int) -> tuple[str, str, Any]:
    if isinstance(program, ProgramRecord):
        representative = str(program.params.get("representative_rule_id", "") or "")
        return program.program_id, representative, program.params.get("supports")
    return (
        _program_id(program, rank),
        str(program.get("representative_rule_id", "") or ""),
        program.get("representative_score_support"),
    )


def assert_train_only_supports(
    programs: Sequence[Mapping[str, Any] | ProgramRecord],
    train_episode_ids: Collection[str],
    inputs: Sequence[dict],
) -> None:
    """Raise :class:`TrainSupportViolation` unless every support was computable from train episodes.

    Train episodes are the ``inputs`` whose id is in ``train_episode_ids``. For each program
    (v1 dict or FMG record) the v1 totals must not exceed the train totals
    (``max(1, n)``, as in ``scm:468-469``), and ``positive_count`` / ``control_count`` must not
    exceed the number of train positives / controls whose rules contain the representative
    rule. A program without support information fails closed.
    """
    if isinstance(train_episode_ids, (str, bytes)):
        raise TypeError("train_episode_ids must be a collection of ids, not a single string")
    train_inputs = filter_contrastive_inputs(inputs, train_episode_ids)
    n_positive = sum(1 for episode in train_inputs if episode.get("episode_type") == "positive")
    n_control = sum(1 for episode in train_inputs if episode.get("episode_type") == "control")
    views = [_support_view(program, rank) for rank, program in enumerate(programs, start=1)]
    needed = {representative for _, representative, _ in views if representative}
    holders: dict[str, dict[str, set[str]]] = {rule_id: {"positive": set(), "control": set()} for rule_id in needed}
    for episode in train_inputs:
        kind = episode.get("episode_type")
        if kind not in ("positive", "control"):
            continue
        for rule in episode.get("rules", []) or []:
            rule_id = str(rule.get("rule_id", "") or "")
            if rule_id in holders:
                holders[rule_id][kind].add(str(episode.get("episode_id")))

    problems: list[str] = []
    for program_id, representative, support in views:
        if not representative or not isinstance(support, Mapping):
            problems.append(f"{program_id}: no representative rule or support counts to verify")
            continue
        try:
            positive_total = int(support["positive_total"])
            control_total = int(support["control_total"])
            positive_count = int(support["positive_count"])
            control_count = int(support["control_count"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{program_id}: incomplete support counts {dict(support)}")
            continue
        if positive_total > max(1, n_positive) or control_total > max(1, n_control):
            problems.append(
                f"{program_id}: totals {positive_total}+/{control_total}- exceed train totals "
                f"{n_positive}+/{n_control}-"
            )
        train_positive = len(holders[representative]["positive"])
        train_control = len(holders[representative]["control"])
        if positive_count > train_positive or control_count > train_control:
            problems.append(
                f"{program_id}: support {positive_count}+/{control_count}- exceeds the train support "
                f"{train_positive}+/{train_control}- of {representative!r}"
            )
    if problems:
        raise TrainSupportViolation("supports use non-train episodes:\n  " + "\n  ".join(problems))
