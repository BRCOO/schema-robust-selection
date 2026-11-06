"""Generator-agnostic schema-survivable diversified selection: paper Eq. (1) + Algorithm 2.

:func:`select` applies ONE selection procedure to any pool of ``ProgramRecord`` candidates
(FMG-Bridge, PrefixSpan, RuleGrowth, CM-SPADE, motif, shapelet, ...). It replaces two v1 code
paths that the paper presented as one "FMG selector":

* the selector that produced the locked FMG registry, a multi-stage heuristic spread over
  ``src/icu/shock_contrastive_miner.py`` and ``scripts/run_mimic_experiment.py`` (Table B), and
* the one-line stand-in that the fair-selector table (Table 5) applied to classical pools:
  ``utility + 0.85*novelty + 0.35*schema - 1.20*duplicate``
  (``scripts/build_fair_selector_baselines.py:181-182``).

Objective (a set function, so the per-step marginal gains telescope to F(P))
----------------------------------------------------------------------------
::

    F(P) =  w_utility    * U(P)          U(P)        = sum_p u(p)
          + w_family     * C_fam(P)      C_fam(P)    = |union_p g(p)|
          + w_anchor     * C_anchor(P)   C_anchor(P) = |union_p sigma_a(p)|
          + w_schema     * S_schema(P)   S_schema(P) = sum_p s(p)
          + w_lead       * L_lead(P)     L_lead(P)   = number of distinct lead-time bins
          - w_redundancy * R(P)          R(P)        = sum over unordered pairs of r(p, q)
          - w_care       * D_care(P)     D_care(P)   = sum_p d(p)

* u(p): ``ProgramRecord.score`` min-max normalized to [0, 1] over the bounded pool B (every
  unit gets 1.0 when all pool scores are equal). Any generator's native score works (FMG
  contrastive score, support, confidence, ...); only its order and spacing matter.
* g(p), the program family: ``{coarse_group(f) for f in sigma_f(p)}``, the coarse mechanism
  groups of ALL features. sigma_a(p) is ``ProgramRecord.anchor_signature`` (coarse groups of
  the terminal element: the consequent for rules, the last itemset for sequences).
* s(p) = mean over target schemas F_t of 1[sigma_f(p) subset of F_t]; 0 when no target is
  given. An empty sigma_f survives vacuously (same convention as ``program_metrics``).
* lead time l(p) = ``params["lead_time_hours_mean"]`` (else ``params["lead_time_hours"]``);
  bin = floor(l(p) / lead_gap_hours). Programs without a lead time add no bin.
* r(p, q) = ``redundancy_fn(p, q)`` (a float, or a mapping with key ``"value"``). Default:
  ``src.icu.v2.program_metrics.pairwise_redundancy(p, q, train_firings)["value"]`` (paper
  Definition 3), imported lazily. r is evaluated only when ``w_redundancy > 0`` or a
  redundancy gate is declared.
* d(p) = |sigma_f(p) & care_process_features| / |sigma_f(p)| (0 for an empty sigma_f).

The marginal gain of unit u given P is F(P + {u}) - F(P): u(u); the NUMBER of new family
groups; the number of new anchor groups; s(u); 1 if u opens a new lead-time bin; minus
sum_{q in P} r(u, q); minus d(u).

Algorithm
---------
1. :func:`consolidate`: one unit per (sorted sigma_f, sorted sigma_a, method), keeping the
   highest score (ties: smaller program_id, then canonical key).
2. Bounded pool B = top ``pool_size`` units by score (quality view), plus the top
   ``pool_per_anchor_group`` units for each coarse group occurring in some sigma_a (anchor
   view), plus the top ``pool_per_family`` units for each coarse group occurring in some
   g(p) (family view), plus, for each target schema F_t, the top ``pool_per_schema`` units
   executable on it, sigma_f(p) within F_t (schema view; only when w_schema > 0, so a
   configuration without the schema term keeps the same pool). Without the schema view the schema
   term can only act on executable units that already score high: on the MIMIC-IV pools
   with labs removed from the target schema, 15 of 200 PrefixSpan and 49 of 200 ERMiner
   candidates were executable but only 2 of each reached the score-based pool, so the
   selection did not change for any w_schema up to 10 (a structural check that needs no
   labels; see scripts/v2/run_schema_aware_selection.py).
3. Greedy: while |P| < k, add the cap-feasible unit of B with the largest marginal gain.
   Tie-break (deterministic and independent of input order): gain rounded to 1e-12, then the
   larger raw score, then the smaller program_id, then method. Hard caps: at most
   ``max_per_anchor_group`` selected units with an identical sigma_a, at most
   ``max_per_family`` with an identical g(p), and at most ``care_process_cap`` units whose
   terminal element contains a care-process feature among the first
   ``care_process_cap_window`` ranks. Caps are never relaxed: when every remaining unit is
   blocked, fewer than k units are returned (see ``stop_reason``).
4. Repair: while a declared gate is violated (anchor coverage < ``min_anchor_coverage``,
   family coverage < ``min_family_coverage``, Definition-3 mean redundancy >
   ``max_redundancy``) and fewer than ``max_repair_swaps`` swaps were made, apply the best
   cap-feasible single swap (a selected unit for an unselected unit of B) that worsens no
   gate and improves at least one. Best = smallest remaining coverage shortfall, then
   smallest redundancy excess, then largest F, then the later rank is swapped out, then the
   unit tie-break. The incoming unit takes the outgoing unit's rank.
5. The registry is the selected units in rank order. Program ids are not renamed (firing
   sets are keyed by them); the rank is the list position.

Trace (second return value of :func:`select`): one ``"setup"`` record (consolidation, pool
views, utility normalization, config), one ``"greedy"`` record per pick (raw ``components``,
signed ``weighted`` components, ``gain`` = sum of ``weighted``, ``objective_before`` and
``objective_after``, and the caps that blocked better candidates), one ``"repair"`` record per
swap, and one ``"final"`` record (F(P) by term, gates, stop reasons). The greedy gains plus
the repair deltas sum to the final objective.

Table A. Paper (``paper/ieee_conference_template/main.tex``) -> this module
--------------------------------------------------------------------------
File lines: Eq. (1) l.148-155 (terms explained at l.160), Definition 2 l.127-129, Definition 3
l.131-135, Algorithm 2 l.214-246 (cited below by its own step numbers), Proposition 1
l.285-287 (remark on the deployed selector at l.290).

* lambda1 U(P), "enrichment or frozen-bridge validation score" -> ``w_utility`` * sum of the
  pool-normalized generator score. Modular, as Proposition 1 assumes. No bridge validation
  score is used: selection never sees validation data.
* lambda2 C_fam(P), "covers program families" -> ``w_family`` * |union g(p)|. The paper never
  defines g(p); v2 fixes it to the coarse groups of all features.
* lambda3 C_anchor(P), "covers anchor contexts" -> ``w_anchor`` * |union sigma_a(p)|; sigma_a
  matches the registry example (Cr -> RR carries the respiratory anchor of RR).
* lambda4 S_schema(P); Definition 2 calls it the FRACTION of selected programs that survive
  -> ``w_schema`` * sum s(p). The sum keeps the term modular; for a fixed |P| = k it orders
  sets exactly like the fraction. Several targets are averaged per program.
* lambda5 L_lead(P), "optionally rewards distinct early lead-time regimes" -> ``w_lead`` *
  number of distinct lead-time bins; off by default, as in the locked profile (l.160).
* lambda6 R(P); Definition 3 defines set redundancy as the AVERAGE pairwise r ->
  ``w_redundancy`` * SUM over pairs (= C(|P|, 2) * average). The marginal of the average
  depends on the current mean and can reward adding a redundant program to an already
  redundant set, so it cannot serve as a greedy penalty. The final trace record reports the
  Definition-3 average as ``redundancy_mean``.
* lambda7 D_care(P), "penalizes care-process-dominated programs" -> ``w_care`` * sum of the
  care-feature share of sigma_f.
* Omega_C(K): budget, hard caps, projection rules, fixed tie-breaks -> ``k``, the three caps,
  and the tie-break above. "Projection rules" are NOT a feasibility constraint here: a
  non-survivable program may be selected (it earns a lower s(p)), and its coordinate is
  zeroed at refiring time (``src.icu.v2.bridge.refire_external``).
* Alg. 2 steps 1-2, group by feature signature, anchor signature and miner family, keep the
  highest-quality representative -> :func:`consolidate` ("miner family" = ``method``).
* Alg. 2 steps 3-5, compute U(u), S(u), D(u), s(u) -> precomputed once per unit.
* Alg. 2 step 6, bounded pool from quality-floor, anchor-wise and family-wise views -> step 2
  above. The quality view is rank based (top ``pool_size``), not a score-ratio floor. v2 adds
  the schema-wise view (``pool_per_schema``) so that lambda4 can reach executable programs.
* Alg. 2 steps 8-19, argmax of the marginal gain with deterministic ties, then the cap check
  (an infeasible argmax is marked infeasible) -> caps are checked before the argmax. This is
  equivalent for the set-level caps (counts only grow, so a blocked unit stays blocked); the
  positional care cap blocks a unit only inside its window, as in v1.
* Alg. 2 steps 20-22, swaps while "a declared coverage or redundancy gate is violated" ->
  step 4 above. The paper declares no gate values; the defaults are in :class:`SelectorConfig`.
* Alg. 2 step 23, stable registry ids -> generator ids are kept; rank = list position.
* Proposition 1, (1 - 1/e) special case -> holds for this F when ``w_redundancy = w_care = 0``
  and there are no caps and no repair: U and S_schema are modular and non-negative, C_fam,
  C_anchor and L_lead are coverage counts.

Table B. v1 selection behind the locked FMG registry -> this module
------------------------------------------------------------------
Locked profile = ``docs/experiments/strict_profile_icde_locked.json`` + ``FORMAL_LOCKED_OVERRIDES``
(``scripts/run_component_ablation_upgrade.py:25-104``), the profile of the paper's full row
(``scripts/run_full_ablation_bridge_refit.py:139-162``). SCM = ``src/icu/shock_contrastive_miner.py``,
RME = ``scripts/run_mimic_experiment.py``. Steps in execution order:

a. Rule score, SCM ``score_contrastive_rule_records`` (432-749), the multiplicative contrastive
   score of paper Eq. (3) -> not here: every generator supplies ``ProgramRecord.score``.
b. Rule-level anchor-diversity rerank, RME ``_apply_rule_anchor_diversity_rerank`` (3030;
   weight 0.65, top 5, eps 0.15, floor 0.9) -> no equivalent. It only reorders rules, and
   consolidation re-sorts programs by score, so it only changes the order of exact ties.
c. Consolidation, SCM ``consolidate_contrastive_programs`` (1171-1302), key ``_program_key``
   (752) = (antecedent feature set, consequent feature set), representative = max score ->
   :func:`consolidate` with key (sigma_f, sigma_a, method).
d. Pool, SCM ``_prepare_program_candidate_pool`` (907-990): top 25 per antecedent tuple, every
   unit >= 0.8 * top score, top 3 per consequent coarse group, >= 7 groups, filled to 20 ->
   pool views (step 2).
e. Greedy, SCM ``_select_programs_with_lead_time_diversity`` (993-1168), picks 20 programs by
   score + 0.3 * consequent-group novelty + 0.8 * antecedent-feature novelty - 0.1 * care
   share (+/-0.6 while the picks cover fewer than 3 distinct antecedent features, and a
   rerank toward a new antecedent feature within 0.15); every adjustment is 0 for units below
   0.9 * top score. Caps: <= 2 per antecedent tuple, no care-process program in picks 1-5.
   The picks are renamed ``program_001``... in this order (SCM 1300-1301) -> greedy (step 3).
f. ``coarse_plus_feature`` tags, RME ``_apply_program_mechanism_group_mode`` (1597): units
   >= 0.85 * top score get "group::feature" labels that every later target counts -> none;
   v2 counts coarse groups only.
g. Score calibration (RME 1650) and support confidence (RME 1772): mode "none" / weight 0 in
   the locked profile, i.e. no-ops -> none.
h. Preselect coverage, RME ``_enforce_program_mechanism_coverage_lower_bound`` (1165; window
   20, >= 7 groups): a no-op, because step e returns <= 20 programs -> none.
i. Constrained top 10, RME ``_select_topk_with_constraints`` (551-1162): lexicographic and
   diversity first (new-group priorities outrank score) inside the units >= 0.85 * top score
   (pool <= 20); caps top-5 same antecedent <= 2 and top-5 care <= 0, relaxed (levels 1-3)
   when nothing fits; then <= 25 swaps toward top-5 >= 5 and top-10 >= 7 groups
   (coarse+feature labels), then top-5 and coverage "rescue" swaps that may admit units below
   0.75 * floor -> greedy caps + one repair loop (steps 3-4).
j. Autotune, RME 5373-5540 (2 rounds): re-runs step i with perturbed targets, pool, floor and
   swap budget and keeps the best gate tuple, including a "transfer target" that compares
   top-5/top-10 group counts with a reference run (the paper's "w/o schema survivability"
   ablation removes it) -> none.
k. Registry = the first 10 of the step-i ranking (RME 5541;
   ``scripts/run_contract_quality_complete.py:517``), still carrying the step-e names -> the
   registry is the greedy/repair order and keeps the generator ids.

Name crossover. v1 "anchor" (SCM 825, RME 1575) is the sorted ANTECEDENT feature tuple;
v1 "mechanism_groups" are the coarse groups of the CONSEQUENT features, which is v2 sigma_a.
The v1 ablation labels the mechanism-group terms "family coverage" and the antecedent terms
"anchor diversity" (``run_full_ablation_bridge_refit.py:174-215``). v2 follows the
``types.py`` definitions instead, so defaults map by meaning, not by name: ``w_anchor`` takes
v1's consequent-group weight (0.3) and ``w_family`` takes v1's antecedent weight (0.8).

Differences from v1 (all of them, as far as the code shows)
----------------------------------------------------------
1. Data. v1 computed rule statistics on a discovery slice that contained the internal test
   episodes (audit P0-2). This module only sees the scores and ``train_firings`` it is given;
   callers must build both from TRAIN episodes only. ``"final"`` lists selected programs that
   never fire on train.
2. Utility scale. v1 used the raw representative score in log2-enrichment units (the locked
   gate requires the 5th-best score to be >= 0.82; the strict mid-60 baseline records a top-5
   minimum of 2.0 in ``docs/experiments/strict_channel_baseline_mid60.json``). v2 uses [0, 1]
   within B, so the same weights give the coverage terms more relative weight than in v1.
   Sweep the weights (audit P1-7) instead of reading them as v1-equivalent.
3. Quality floors. v1 gave diversity bonuses and care/redundancy penalties only to units
   >= 0.9 * top score and drew its final top 10 from units >= 0.85 * top score. v2 applies
   every term to every unit of B; only the pool bound limits how far down diversity reaches.
4. Bonus form. v1 bonuses are per-candidate ratios in [0, 1] (share of the candidate's labels
   not yet covered, else 1 - max Jaccard) and are 0 for the first pick. v2 uses set-function
   marginals (counts of new groups), so the first pick also earns coverage and the gains
   telescope to F(P).
5. Group units. v1 coverage counted coarse groups plus "group::feature" labels; v2 counts
   coarse groups only (at most 9 anchor groups).
6. Ordering. v1's final top 10 was lexicographic (coverage targets dominate score); v2
   ranks by one weighted sum.
7. Schema survival. The v1 selection code has no schema term at all (no "schema" or "surviv"
   in RME or SCM); what the paper ablates as schema survivability is the transfer target of
   step j. v2 adds S_schema; its default weight comes from the Table-5 stand-in, not from the
   v1 selector.
8. Redundancy. v1 locked ``program_redundancy_penalty_weight = 0.0``, and its penalty
   (SCM ``_program_redundancy_penalty``, 833) was a max Jaccard over antecedent/mechanism
   tokens, although main.tex l.290 says the deployed selector "keeps pairwise redundancy
   penalties". The v2 default mirrors v1 (0.0); ``w_redundancy > 0`` uses Definition 3.
9. Caps. v1 relaxed its caps when nothing was feasible (step i) and fell back to the top
   unit in step e; v2 never relaxes and may return fewer than k units. v1 caps keyed on the
   antecedent tuple, v2 caps on sigma_a and g(p).
10. Lead time. v1 bonus = min(1, min_q |l(u) - l(q)| / gap), 0 for the first pick; v2 counts
    newly opened bins of width ``lead_gap_hours``.
11. Care process. v1 penalty = share of the consequent coarse groups equal to care_process
    (only above the 0.9 floor); v2 = share of ALL features in ``care_process_features``. The
    positional cap uses the terminal element in both.
12. Consolidation key. v2 (sigma_f, sigma_a, method) is coarser than v1 (antecedent set,
    consequent set) for rules: {Lactate, MAP} -> MAP merges with Lactate -> MAP. For a
    single-feature numeric method (motif, shapelet) it keeps one template per feature.
13. Not reproduced: steps b, f, g, h, j. The Table-5 stand-in's exact-signature penalty
    (-1.20 * duplicate) is replaced by consolidation plus the Definition-3 redundancy.
14. Tie-breaks. v1 used (combined score, representative score, mean score, rule count), then
    input order; v2 uses (rounded gain, score, program_id, method), independent of input order.
15. Repair. v1 ran three swap stages (25-swap stage 2 plus two rescue loops) on coarse+feature
    targets; v2 runs one loop (at most ``max_repair_swaps``) on coarse coverage gates and only
    swaps in units of B.
16. Registry ids. v1 renamed its 20 step-e picks ``program_001``... before step i reordered
    them, so v1 ids do not follow the final rank; v2 keeps the generator ids and the rank is
    the list position.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from itertools import combinations

from src.icu.v2.types import ProgramRecord, coarse_group, features_of_itemsets

FiringSets = Mapping[str, AbstractSet[str]]
RedundancyFn = Callable[[ProgramRecord, ProgramRecord], object]

CARE_PROCESS_FEATURES = ("vasopressor_on", "ventilation_on", "sedation_on")
LEAD_TIME_PARAM_KEYS = ("lead_time_hours_mean", "lead_time_hours")
GAIN_DECIMALS = 12
TERMS = (
    "utility",
    "family_coverage",
    "anchor_coverage",
    "schema_survival",
    "lead_time",
    "redundancy",
    "care_process",
)
WEIGHT_FIELDS = ("w_utility", "w_family", "w_anchor", "w_schema", "w_lead", "w_redundancy", "w_care")
OBJECTIVE_TEXT = (
    "F(P) = w_utility*sum u + w_family*|union g| + w_anchor*|union sigma_a| + w_schema*sum s "
    "+ w_lead*#lead bins - w_redundancy*sum_pairs r - w_care*sum d"
)

_TERM_WEIGHT = dict(zip(TERMS, WEIGHT_FIELDS))
_PENALTY_TERMS = frozenset({"redundancy", "care_process"})
_NUMERIC_KINDS = frozenset({"motif", "shapelet"})
_GATE_TOLERANCE = 1e-12


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class SelectorConfig:
    """Budget, Eq. (1) weights, hard caps, pool views, and repair gates.

    Defaults mirror the v1 locked profile where a counterpart exists. LP = the locked profile
    (``docs/experiments/strict_profile_icde_locked.json`` + ``FORMAL_LOCKED_OVERRIDES`` in
    ``scripts/run_component_ablation_upgrade.py``); SCM = ``src/icu/shock_contrastive_miner.py``.

    ======================== ======= =================================================================
    field                    default source
    ======================== ======= =================================================================
    k                        10      ``top_k`` in ``docs/experiments/icde_comparator_execution_manifest.json``
                                     and ``icde_mining_contract.json``; LP ``program_top_window_size``
    w_utility (lambda1)      1.0     SCM greedy adds the representative score with weight 1 (raw units)
    w_family (lambda2)       0.8     LP ``program_anchor_diversity_weight`` (v1 antecedent-feature
                                     novelty, the v1 term closest to an all-feature family)
    w_anchor (lambda3)       0.3     LP ``program_mechanism_diversity_weight`` (v1 consequent-group
                                     novelty; consequent coarse groups = v2 sigma_a)
    w_schema (lambda4)       0.35    no v1 selector equivalent; schema weight of the Table-5 stand-in
                                     (``scripts/build_fair_selector_baselines.py:182``)
    w_lead (lambda5)         0.0     LP ``program_leadtime_diversity_weight``
    w_redundancy (lambda6)   0.0     LP ``program_redundancy_penalty_weight`` (main.tex l.290 says the
                                     deployed selector keeps redundancy penalties; v1 did not)
    w_care (lambda7)         0.1     LP ``program_care_process_penalty``
    max_per_anchor_group     2       LP ``program_anchor_max_same_count`` and
                                     ``algorithm_ready_top5_max_same_anchor_max`` (v1 applied them to
                                     the antecedent tuple, not to sigma_a)
    max_per_family           None    no v1 equivalent (v1 ``program_mechanism_max_same_count`` = 0, off)
    care_process_cap         0       LP ``program_topk_care_process_max_count``
                                     (= ``algorithm_ready_top5_care_process_max``)
    care_process_cap_window  5       LP ``program_topk_care_process_window``
    pool_size                20      LP ``program_constrained_candidate_pool_size``; SCM
                                     ``consolidate_contrastive_programs(max_programs=20)``
    pool_per_anchor_group    3       LP ``program_mechanism_candidate_top_n`` (top 3 per consequent
                                     coarse group)
    pool_per_family          3       no v1 family view (v1 antecedent view, top 25 per tuple, was
                                     effectively unbounded); same depth as the anchor view
    pool_per_schema          20      no v1 equivalent (v1 had no schema term); per target schema,
                                     as deep as the quality view; None or 0 disables the view
    min_anchor_coverage      6       LP ``algorithm_ready_top10_unique_groups_min``; v1 counted
                                     coarse+feature labels, so 6 coarse groups is stricter (v1 selector
                                     targets were 5 for the top 5 and 7 for the top 10, same labels)
    min_family_coverage      None    no v1 equivalent
    max_redundancy           None    no v1 selector gate (the paper's 0.55 structural gate was added
                                     after the main runs; audit P1-2)
    max_repair_swaps         25      LP ``program_constrained_two_stage_swap_budget``
    lead_gap_hours           1.0     LP ``program_leadtime_diversity_gap_hours``
    care_process_features    3 names vasopressor_on, ventilation_on, sedation_on: the features that
                                     ``MECHANISM_GROUP_BY_FEATURE`` maps to ``care_process``
    ======================== ======= =================================================================

    Caps use ``None`` for "no cap"; weights must be finite and non-negative (penalties are
    subtracted by the objective).
    """

    k: int = 10
    w_utility: float = 1.0
    w_family: float = 0.8
    w_anchor: float = 0.3
    w_schema: float = 0.35
    w_lead: float = 0.0
    w_redundancy: float = 0.0
    w_care: float = 0.1
    max_per_anchor_group: int | None = 2
    max_per_family: int | None = None
    care_process_cap: int | None = 0
    care_process_cap_window: int = 5
    pool_size: int | None = 20
    pool_per_anchor_group: int = 3
    pool_per_family: int = 3
    pool_per_schema: int | None = 20
    min_anchor_coverage: int | None = 6
    min_family_coverage: int | None = None
    max_redundancy: float | None = None
    max_repair_swaps: int = 25
    lead_gap_hours: float = 1.0
    care_process_features: tuple[str, ...] = CARE_PROCESS_FEATURES

    def __post_init__(self) -> None:
        if not _is_int(self.k) or self.k < 1:
            raise ValueError(f"k must be an integer >= 1, got {self.k!r}")
        for name in WEIGHT_FIELDS:
            value = getattr(self, name)
            if not _is_number(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        for name in ("max_per_anchor_group", "max_per_family", "pool_size"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 1):
                raise ValueError(f"{name} must be None (no limit) or an integer >= 1, got {value!r}")
        for name in ("care_process_cap", "min_anchor_coverage", "min_family_coverage"):
            value = getattr(self, name)
            if value is not None and (not _is_int(value) or value < 0):
                raise ValueError(f"{name} must be None or an integer >= 0, got {value!r}")
        if self.pool_per_schema is not None and (not _is_int(self.pool_per_schema) or self.pool_per_schema < 0):
            raise ValueError(f"pool_per_schema must be None or an integer >= 0, got {self.pool_per_schema!r}")
        for name in ("pool_per_anchor_group", "pool_per_family", "max_repair_swaps"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}")
        if not _is_int(self.care_process_cap_window) or self.care_process_cap_window < 1:
            raise ValueError(f"care_process_cap_window must be an integer >= 1, got {self.care_process_cap_window!r}")
        if self.max_redundancy is not None and (not _is_number(self.max_redundancy) or self.max_redundancy < 0):
            raise ValueError(f"max_redundancy must be None or a finite number >= 0, got {self.max_redundancy!r}")
        if not _is_number(self.lead_gap_hours) or self.lead_gap_hours <= 0:
            raise ValueError(f"lead_gap_hours must be a finite number > 0, got {self.lead_gap_hours!r}")
        if isinstance(self.care_process_features, str):
            raise ValueError("care_process_features must be a sequence of feature names, not one string")
        object.__setattr__(self, "care_process_features", tuple(str(f) for f in self.care_process_features))


def program_family(program: ProgramRecord) -> frozenset[str]:
    """g(p): coarse mechanism groups of all features the program needs."""
    return frozenset(coarse_group(feature) for feature in program.features)


def terminal_features(program: ProgramRecord) -> frozenset[str]:
    """Base features of the terminal element (consequent for rules, last itemset for sequences).

    Numeric kinds, and symbolic records without itemsets, fall back to ``features``.
    """
    if program.kind not in _NUMERIC_KINDS:
        terminal = program.consequent if program.consequent else program.elements[-1:]
        found = features_of_itemsets(tuple(terminal))
        if found:
            return found
    return frozenset(program.features)


def consolidation_key(program: ProgramRecord) -> tuple:
    """Group key of :func:`consolidate`: (sorted sigma_f, sorted sigma_a, method)."""
    return (tuple(sorted(program.features)), tuple(sorted(program.anchor_signature)), program.method)


def _representative_key(program: ProgramRecord) -> tuple:
    return (-float(program.score), program.program_id, repr(program.canonical_key()))


def _rank_key(program: ProgramRecord) -> tuple:
    return (-float(program.score), program.program_id, program.method)


def _consolidate_groups(candidates: Sequence[ProgramRecord]) -> tuple[list[ProgramRecord], dict[str, list[str]]]:
    groups: dict[tuple, list[ProgramRecord]] = {}
    identity: dict[str, tuple] = {}
    for program in candidates:
        if not isinstance(program, ProgramRecord):
            raise TypeError(f"candidates must be ProgramRecord objects, got {type(program).__name__}")
        if not math.isfinite(float(program.score)):
            raise ValueError(f"program {program.program_id!r} has a non-finite score")
        ident = (program.method, program.canonical_key(), frozenset(program.features), frozenset(program.anchor_signature))
        if identity.setdefault(program.program_id, ident) != ident:
            raise ValueError(
                f"program_id {program.program_id!r} names two different programs; "
                "firing sets keyed by program_id would be ambiguous"
            )
        groups.setdefault(consolidation_key(program), []).append(program)
    representatives: list[ProgramRecord] = []
    merged: dict[str, list[str]] = {}
    for members in groups.values():
        representative = min(members, key=_representative_key)
        representatives.append(representative)
        others = sorted({member.program_id for member in members} - {representative.program_id})
        if others:
            merged[representative.program_id] = others
    representatives.sort(key=_rank_key)
    return representatives, merged


def consolidate(candidates: Sequence[ProgramRecord]) -> list[ProgramRecord]:
    """Keep one program unit per (sorted sigma_f, sorted sigma_a, method) group.

    The representative is the member with the highest ``score``; ties go to the smaller
    ``program_id``, then to the smaller ``repr(canonical_key())``. The result is sorted by
    score (descending), then program_id, then method, and does not depend on input order.
    Raises ``ValueError`` for a non-finite score or for one program_id naming two different
    programs. Idempotent.
    """
    return _consolidate_groups(candidates)[0]


def _group_view(units: Sequence[ProgramRecord], groups_of: Callable[[ProgramRecord], AbstractSet[str]], per_group: int) -> set[str]:
    """Ids of the top ``per_group`` units (in rank order) for every coarse group they carry."""
    kept: set[str] = set()
    if per_group <= 0:
        return kept
    counts: dict[str, int] = {}
    for program in units:
        for group in sorted(groups_of(program)):
            if counts.get(group, 0) < per_group:
                counts[group] = counts.get(group, 0) + 1
                kept.add(program.program_id)
    return kept


def _bounded_pool(
    units: list[ProgramRecord], config: SelectorConfig, targets: Mapping[str, frozenset[str]] | None = None
) -> tuple[list[ProgramRecord], dict[str, list[str]]]:
    quality = units if config.pool_size is None else units[: config.pool_size]
    anchor_view = _group_view(units, lambda program: program.anchor_signature, config.pool_per_anchor_group)
    family_view = _group_view(units, program_family, config.pool_per_family)
    schema_view: set[str] = set()
    if config.pool_per_schema and targets and config.w_schema > 0:  # the view only serves the schema term
        for features in targets.values():
            executable = [program for program in units if frozenset(program.features) <= features]
            schema_view.update(program.program_id for program in executable[: config.pool_per_schema])
    keep = {program.program_id for program in quality} | anchor_view | family_view | schema_view
    pool = [program for program in units if program.program_id in keep]
    views = {
        "quality": [program.program_id for program in quality],
        "anchor": [program.program_id for program in units if program.program_id in anchor_view],
        "family": [program.program_id for program in units if program.program_id in family_view],
        "schema": [program.program_id for program in units if program.program_id in schema_view],
    }
    return pool, views


def _lead_bin(program: ProgramRecord, gap_hours: float) -> int | None:
    for key in LEAD_TIME_PARAM_KEYS:
        if key in program.params and program.params[key] is not None:
            try:
                hours = float(program.params[key])
            except (TypeError, ValueError):
                return None
            return int(math.floor(hours / gap_hours)) if math.isfinite(hours) else None
    return None


@dataclass(frozen=True, eq=False)
class _Unit:
    program: ProgramRecord
    utility: float
    family: frozenset[str]
    anchor: frozenset[str]
    schema: float
    care: float
    care_terminal: bool
    lead_bin: int | None

    @property
    def tie_key(self) -> tuple:
        return _rank_key(self.program)


def _make_unit(
    program: ProgramRecord,
    utility: float,
    targets: Mapping[str, frozenset[str]],
    care_features: frozenset[str],
    gap_hours: float,
) -> _Unit:
    features = frozenset(program.features)
    schema = sum(1 for target in targets.values() if features <= target) / len(targets) if targets else 0.0
    return _Unit(
        program=program,
        utility=utility,
        family=program_family(program),
        anchor=frozenset(program.anchor_signature),
        schema=schema,
        care=len(features & care_features) / len(features) if features else 0.0,
        care_terminal=bool(terminal_features(program) & care_features),
        lead_bin=_lead_bin(program, gap_hours),
    )


def _default_redundancy_fn(train_firings: FiringSets) -> RedundancyFn:
    from src.icu.v2.program_metrics import pairwise_redundancy  # lazy: written in parallel, only needed here

    def redundancy(p: ProgramRecord, q: ProgramRecord) -> float:
        return float(pairwise_redundancy(p, q, train_firings)["value"])

    return redundancy


class _PairRedundancy:
    """Symmetric cache of r(p, q) keyed by sorted program ids."""

    def __init__(self, fn: RedundancyFn, units: Mapping[str, _Unit]) -> None:
        self._fn = fn
        self._units = units
        self._cache: dict[tuple[str, str], float] = {}

    def __call__(self, a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        value = self._cache.get(key)
        if value is None:
            raw = self._fn(self._units[key[0]].program, self._units[key[1]].program)
            if isinstance(raw, Mapping):
                raw = raw["value"]
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"redundancy_fn returned a non-finite value for {key}")
            self._cache[key] = value
        return value


class _Objective:
    """Eq. (1) as a set function over unit ids, with marginal gains and gate checks."""

    def __init__(self, units: Mapping[str, _Unit], config: SelectorConfig, pair: _PairRedundancy | None) -> None:
        self.units = units
        self.config = config
        self.pair = pair

    def terms(self, ids: Sequence[str]) -> dict[str, float | None]:
        rows = [self.units[uid] for uid in ids]
        return {
            "utility": math.fsum(row.utility for row in rows),
            "family_coverage": float(len(frozenset().union(*(row.family for row in rows)))),
            "anchor_coverage": float(len(frozenset().union(*(row.anchor for row in rows)))),
            "schema_survival": math.fsum(row.schema for row in rows),
            "lead_time": float(len({row.lead_bin for row in rows if row.lead_bin is not None})),
            "redundancy": None if self.pair is None else math.fsum(self.pair(a, b) for a, b in combinations(ids, 2)),
            "care_process": math.fsum(row.care for row in rows),
        }

    def weighted(self, raw: Mapping[str, float | None]) -> dict[str, float]:
        """Signed contributions: weight * value, negated for the two penalties."""
        out: dict[str, float] = {}
        for term in TERMS:
            value = raw[term] or 0.0
            contribution = getattr(self.config, _TERM_WEIGHT[term]) * value
            out[term] = 0.0 - contribution if term in _PENALTY_TERMS else contribution
        return out

    def value(self, ids: Sequence[str]) -> float:
        return math.fsum(self.weighted(self.terms(ids)).values())

    def marginal(
        self,
        unit: _Unit,
        selected: Sequence[str],
        covered_family: AbstractSet[str],
        covered_anchor: AbstractSet[str],
        covered_bins: AbstractSet[int],
    ) -> tuple[dict[str, float | None], frozenset[str], frozenset[str]]:
        new_family = unit.family - covered_family
        new_anchor = unit.anchor - covered_anchor
        opens_bin = unit.lead_bin is not None and unit.lead_bin not in covered_bins
        uid = unit.program.program_id
        raw = {
            "utility": unit.utility,
            "family_coverage": float(len(new_family)),
            "anchor_coverage": float(len(new_anchor)),
            "schema_survival": unit.schema,
            "lead_time": 1.0 if opens_bin else 0.0,
            "redundancy": None if self.pair is None else math.fsum(self.pair(uid, q) for q in selected),
            "care_process": unit.care,
        }
        return raw, new_family, new_anchor

    def redundancy_mean(self, ids: Sequence[str]) -> float | None:
        """Definition-3 set redundancy (average over pairs); 0.0 below two programs."""
        if self.pair is None:
            return None
        pairs = list(combinations(ids, 2))
        return math.fsum(self.pair(a, b) for a, b in pairs) / len(pairs) if pairs else 0.0

    def gates(self, ids: Sequence[str]) -> dict[str, dict]:
        config = self.config
        rows = [self.units[uid] for uid in ids]
        report: dict[str, dict] = {}
        coverage = (
            ("anchor_coverage", config.min_anchor_coverage, lambda row: row.anchor),
            ("family_coverage", config.min_family_coverage, lambda row: row.family),
        )
        for name, required, groups_of in coverage:
            if required is not None:
                observed = len(frozenset().union(*(groups_of(row) for row in rows)))
                report[name] = {
                    "required_min": required,
                    "observed": observed,
                    "shortfall": max(0, required - observed),
                    "pass": observed >= required,
                }
        if config.max_redundancy is not None:
            observed = self.redundancy_mean(ids) or 0.0
            excess = max(0.0, observed - config.max_redundancy)
            report["redundancy_mean"] = {
                "required_max": config.max_redundancy,
                "observed": observed,
                "excess": excess,
                "pass": excess <= _GATE_TOLERANCE,
            }
        return report


def _blocking_caps(
    unit: _Unit,
    position: int,
    anchor_counts: Mapping[frozenset[str], int],
    family_counts: Mapping[frozenset[str], int],
    care_count: int,
    config: SelectorConfig,
) -> list[str]:
    caps = []
    if config.max_per_anchor_group is not None and anchor_counts.get(unit.anchor, 0) >= config.max_per_anchor_group:
        caps.append("anchor_group")
    if config.max_per_family is not None and family_counts.get(unit.family, 0) >= config.max_per_family:
        caps.append("family")
    if (
        config.care_process_cap is not None
        and unit.care_terminal
        and position < config.care_process_cap_window
        and care_count >= config.care_process_cap
    ):
        caps.append("care_process_window")
    return caps


def _caps_ok(ids: Sequence[str], units: Mapping[str, _Unit], config: SelectorConfig) -> bool:
    anchor_counts: dict[frozenset[str], int] = {}
    family_counts: dict[frozenset[str], int] = {}
    care_count = 0
    for position, uid in enumerate(ids):
        unit = units[uid]
        anchor_counts[unit.anchor] = anchor_counts.get(unit.anchor, 0) + 1
        family_counts[unit.family] = family_counts.get(unit.family, 0) + 1
        if unit.care_terminal and position < config.care_process_cap_window:
            care_count += 1
    if config.max_per_anchor_group is not None and max(anchor_counts.values(), default=0) > config.max_per_anchor_group:
        return False
    if config.max_per_family is not None and max(family_counts.values(), default=0) > config.max_per_family:
        return False
    return config.care_process_cap is None or care_count <= config.care_process_cap


def _greedy(
    order: Sequence[str],
    units: Mapping[str, _Unit],
    objective: _Objective,
    config: SelectorConfig,
    train_firings: FiringSets,
) -> tuple[list[str], list[dict], str]:
    selected: list[str] = []
    chosen: set[str] = set()
    covered_family: set[str] = set()
    covered_anchor: set[str] = set()
    covered_bins: set[int] = set()
    anchor_counts: dict[frozenset[str], int] = {}
    family_counts: dict[frozenset[str], int] = {}
    care_count = 0
    records: list[dict] = []
    objective_before = 0.0
    if not order:
        return selected, records, "empty_pool"
    stop_reason = "k_reached"
    while len(selected) < config.k:
        position = len(selected)
        feasible: list[tuple[tuple, dict]] = []
        blocked: list[tuple[tuple, dict, list[str]]] = []
        for uid in order:
            if uid in chosen:
                continue
            unit = units[uid]
            raw, new_family, new_anchor = objective.marginal(unit, selected, covered_family, covered_anchor, covered_bins)
            weighted = objective.weighted(raw)
            gain = math.fsum(weighted.values())
            key = (-round(gain, GAIN_DECIMALS), *unit.tie_key)
            row = {"uid": uid, "raw": raw, "weighted": weighted, "gain": gain, "new_family": new_family, "new_anchor": new_anchor}
            caps = _blocking_caps(unit, position, anchor_counts, family_counts, care_count, config)
            if caps:
                blocked.append((key, row, caps))
            else:
                feasible.append((key, row))
        if not feasible:
            stop_reason = "hard_caps_block_remaining" if blocked else "pool_exhausted"
            break
        feasible.sort(key=lambda item: item[0])
        best_key, best = feasible[0]
        unit = units[best["uid"]]
        selected.append(best["uid"])
        chosen.add(best["uid"])
        covered_family |= unit.family
        covered_anchor |= unit.anchor
        if unit.lead_bin is not None:
            covered_bins.add(unit.lead_bin)
        anchor_counts[unit.anchor] = anchor_counts.get(unit.anchor, 0) + 1
        family_counts[unit.family] = family_counts.get(unit.family, 0) + 1
        if unit.care_terminal and position < config.care_process_cap_window:
            care_count += 1
        objective_after = objective.value(selected)
        better_blocked = sorted((item for item in blocked if item[0] < best_key), key=lambda item: item[0])
        blocked_by_cap: dict[str, int] = {}
        for _, _, caps in blocked:
            for cap in caps:
                blocked_by_cap[cap] = blocked_by_cap.get(cap, 0) + 1
        records.append(
            {
                "phase": "greedy",
                "rank": position + 1,
                "program_id": best["uid"],
                "method": unit.program.method,
                "kind": unit.program.kind,
                "score": float(unit.program.score),
                "components": best["raw"],
                "weighted": best["weighted"],
                "gain": best["gain"],
                "objective_before": objective_before,
                "objective_after": objective_after,
                "new_family_groups": sorted(best["new_family"]),
                "new_anchor_groups": sorted(best["new_anchor"]),
                "train_firing_count": len(train_firings.get(best["uid"], ())),
                "n_feasible": len(feasible),
                "n_blocked": len(blocked),
                "blocked_by_cap": blocked_by_cap,
                "active_caps": sorted({cap for _, _, caps in better_blocked for cap in caps}),
                "blocked_better": [
                    {"program_id": row["uid"], "gain": row["gain"], "caps": caps} for _, row, caps in better_blocked[:3]
                ],
                "runner_up": (
                    {"program_id": feasible[1][1]["uid"], "gain": feasible[1][1]["gain"]} if len(feasible) > 1 else None
                ),
            }
        )
        objective_before = objective_after
    return selected, records, stop_reason


def _gate_amounts(report: Mapping[str, dict]) -> tuple[int, float]:
    shortfall = sum(int(gate.get("shortfall", 0)) for gate in report.values())
    excess = math.fsum(float(gate.get("excess", 0.0)) for gate in report.values())
    return shortfall, excess


def _improves(before: Mapping[str, dict], after: Mapping[str, dict]) -> bool:
    """True when no gate gets worse and at least one gets better."""
    better = False
    for name, gate in before.items():
        old = float(gate.get("shortfall", gate.get("excess", 0.0)))
        new = float(after[name].get("shortfall", after[name].get("excess", 0.0)))
        if new > old + _GATE_TOLERANCE:
            return False
        if new < old - _GATE_TOLERANCE:
            better = True
    return better


def _repair(
    selected: Sequence[str],
    order: Sequence[str],
    units: Mapping[str, _Unit],
    objective: _Objective,
    config: SelectorConfig,
) -> tuple[list[str], list[dict], str]:
    current = list(selected)
    records: list[dict] = []
    while True:
        report = objective.gates(current)
        if all(gate["pass"] for gate in report.values()):
            return current, records, "gates_satisfied" if report else "no_gates_declared"
        if config.max_repair_swaps == 0:
            return current, records, "repair_disabled"
        if len(records) >= config.max_repair_swaps:
            return current, records, "max_repair_swaps_reached"
        before_value = objective.value(current)
        in_set = set(current)
        best: tuple | None = None
        for out_position in range(len(current)):
            for in_id in order:
                if in_id in in_set:
                    continue
                trial = list(current)
                trial[out_position] = in_id
                if not _caps_ok(trial, units, config):
                    continue
                trial_report = objective.gates(trial)
                if not _improves(report, trial_report):
                    continue
                value = objective.value(trial)
                shortfall, excess = _gate_amounts(trial_report)
                key = (
                    shortfall,
                    round(excess, GAIN_DECIMALS),
                    -round(value, GAIN_DECIMALS),
                    -out_position,
                    *units[in_id].tie_key,
                )
                if best is None or key < best[0]:
                    best = (key, out_position, in_id, trial_report, value)
        if best is None:
            return current, records, "no_valid_improving_swap"
        _, out_position, in_id, trial_report, value = best
        swap_out = current[out_position]
        current[out_position] = in_id
        records.append(
            {
                "phase": "repair",
                "swap": len(records) + 1,
                "rank": out_position + 1,
                "swap_out": swap_out,
                "swap_in": in_id,
                "objective_before": before_value,
                "objective_after": value,
                "delta": value - before_value,
                "gates_before": report,
                "gates_after": trial_report,
            }
        )


def select(
    candidates: Sequence[ProgramRecord],
    config: SelectorConfig,
    train_firings: FiringSets,
    target_schemas: Mapping[str, AbstractSet[str]],
    redundancy_fn: RedundancyFn | None = None,
) -> tuple[list[ProgramRecord], list[dict]]:
    """Select a ranked registry of at most ``config.k`` programs and return ``(registry, trace)``.

    ``train_firings`` maps program_id to the TRAIN episode ids where the program fires; it
    feeds the default redundancy (paper Definition 3) and the trace. ``target_schemas`` maps
    a target name to its available base features (an empty mapping disables the schema
    term). ``redundancy_fn(p, q)`` returns r(p, q) as a float or as a mapping with key
    ``"value"``; when it is ``None`` the default is imported lazily, and only if
    ``config.w_redundancy > 0`` or ``config.max_redundancy`` is set. See the module
    docstring for the objective, the algorithm, and the trace records.
    """
    if not isinstance(config, SelectorConfig):
        raise TypeError("config must be a SelectorConfig")
    units_list, merged = _consolidate_groups(candidates)
    targets = {str(name): frozenset(features) for name, features in sorted(target_schemas.items())}
    pool, views = _bounded_pool(units_list, config, targets)
    scores = [float(program.score) for program in pool]
    low, high = (min(scores), max(scores)) if scores else (None, None)
    care_features = frozenset(config.care_process_features)
    units = {
        program.program_id: _make_unit(
            program,
            (float(program.score) - low) / (high - low) if high > low else 1.0,
            targets,
            care_features,
            config.lead_gap_hours,
        )
        for program in pool
    }
    order = [program.program_id for program in pool]

    needs_redundancy = config.w_redundancy > 0 or config.max_redundancy is not None
    if not needs_redundancy:
        pair = None
        redundancy_source = "not evaluated (w_redundancy == 0 and no redundancy gate)"
    elif redundancy_fn is not None:
        pair = _PairRedundancy(redundancy_fn, units)
        redundancy_source = "caller-supplied redundancy_fn"
    else:
        pair = _PairRedundancy(_default_redundancy_fn(train_firings), units)
        redundancy_source = "program_metrics.pairwise_redundancy on train_firings (paper Definition 3)"
    objective = _Objective(units, config, pair)

    trace: list[dict] = [
        {
            "phase": "setup",
            "n_candidates": len(candidates),
            "n_units": len(units_list),
            "merged_members": merged,
            "pool_ids": order,
            "pool_views": views,
            "utility_normalization": {
                "rule": "min-max of ProgramRecord.score over the bounded pool; all equal -> 1.0",
                "min": low,
                "max": high,
            },
            "n_units_without_anchor_signature": sum(1 for uid in order if not units[uid].anchor),
            "target_schemas": sorted(targets),
            "redundancy": redundancy_source,
            "objective": OBJECTIVE_TEXT,
            "config": asdict(config),
        }
    ]
    selected, greedy_records, stop_reason = _greedy(order, units, objective, config, train_firings)
    trace.extend(greedy_records)
    selected, repair_records, repair_reason = _repair(selected, order, units, objective, config)
    trace.extend(repair_records)

    terms = objective.terms(selected)
    weighted = objective.weighted(terms)
    trace.append(
        {
            "phase": "final",
            "selected_ids": list(selected),
            "n_selected": len(selected),
            "k": config.k,
            "stop_reason": stop_reason,
            "repair_stop_reason": repair_reason,
            "n_repair_swaps": len(repair_records),
            "objective": math.fsum(weighted.values()),
            "terms": terms,
            "weighted_terms": weighted,
            "sum_of_greedy_gains": math.fsum(record["gain"] for record in greedy_records),
            "sum_of_repair_deltas": math.fsum(record["delta"] for record in repair_records),
            "redundancy_mean": objective.redundancy_mean(selected),
            "gates": objective.gates(selected),
            "anchor_groups": sorted(frozenset().union(*(units[uid].anchor for uid in selected))),
            "family_groups": sorted(frozenset().union(*(units[uid].family for uid in selected))),
            "selected_without_train_firing": [uid for uid in selected if not train_firings.get(uid)],
            "nonsurvivable_by_target": {
                name: [uid for uid in selected if not frozenset(units[uid].program.features) <= features]
                for name, features in targets.items()
            },
        }
    )
    return [units[uid].program for uid in selected], trace
