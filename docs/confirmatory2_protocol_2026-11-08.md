# Second confirmatory study: the eICU-source design on pools with window-statistic predicates

Protocol written on 2026-11-08, before any selection, firing, model fitting or evaluation of this study. It is committed to
the code repository before the study's first run; results will be appended in a separate section without changing
Sections 1-7.

## 1. Purpose and what was known when this was written

The paper's candidate pools are being extended by a sixth generator, window-statistic predicates (a window statistic of one
feature compared with a TRAIN quantile). An exploratory look at the extended pools on the main study's four K = 10 settings
(MIMIC-IV source) showed that every selector gained several AUPRC points at the real targets, that most selected programs were
predicates, that the lead of SRS+T over NMIFS shrank to between -0.02 and +2.6 points per setting, and that SRS+T registries
scored about as well as logistic models on the K window statistics chosen by an L1 path. The first confirmatory study
(protocol of 2026-11-07, `docs/confirmatory_protocol_2026-11-07.md` in the code release; same design, the four non-FMG
generators) had found H1-H3 holding and H4 (mRMR) and H5 failing. No eICU-source result with the extended pools has been
seen.

This study tests, on a second source with real site schemas, (i) whether the frozen method still leads the strongest filter
when the pools contain the predicates, (ii) whether the predicates improve the registries at the sites, and (iii) whether a
registry of K programs is non-inferior to a model on K window statistics.

## 2. Method, frozen

As in the first study (its Section 2): SRS+T and SRS with Newton screening of all candidates, shortlist m = 10 refitted
exactly, AUPRC decision among the shortlist, likelihood minimal gain 1e-4 bits; V = 3 folds, eta_0 = 1; surrogate chains with
M = 5; scenarios: source schema, the known target schemas (SRS+T only) and D = 20 random coarse-group dropout schemas, dropout
rate omega in {0, 0.25} chosen on VAL AUPRC; shipped probe class-balanced L2 logistic regression, penalty from
{0.01, 0.1, 1, 10} on VAL. CV seeds 0-4 (Section 5). No parameter, rule or implementation detail will be changed after this
protocol.

## 3. Data and design

* Cohorts, hospital schemas, source hospitals, patient-level split, site groups, MIMIC-IV and HiRID targets and controlled
  targets: exactly those of the first study (its Section 3; the files written by its build phase are reused unchanged):
  vasopressor task 53 source hospitals / 4,671 episodes and 6 site groups (21 hospitals, 1,814 episodes); AKI task
  61 source hospitals / 11,326 episodes and 12 site groups (43 hospitals, 6,267 episodes).
* Pools: the first study's pools (four non-FMG generators, mined on source TRAIN) plus the window-statistic predicates, mined
  on the same source TRAIN by `scripts/v2/add_statistic_predicates.py`: for every source feature the statistics minimum,
  maximum, mean, standard deviation, last value, least-squares slope and share of missing bins (status features: share of
  bins on, number of transitions, last value), each compared with its TRAIN quantiles (>= at 0.5, 0.75, 0.9; <= at 0.1, 0.25,
  0.5); predicates firing on fewer than 2% or more than 98% of TRAIN windows are dropped, predicates with identical TRAIN
  firings are kept once, and the 200 with the highest TRAIN information gain enter the pool (the cap of the SPMF generators).
* Budgets: K in {5, 10, 20}; 6 runs (2 tasks x 3 budgets).
* Comparators: the twelve established selectors of the paper (NMIFS, mRMR, generator-stratified mRMR, JMI, CMIM, MMRFS, MMR,
  MMS, DPP, L1, L1 with schema dropout, Filter+mRMR), tuned on VAL by the paper's drivers; SRS+T with the likelihood decision;
  SRS+T selected from the first study's pools (H6); and three statistic models trained on the eICU source TRAIN with every
  choice on VAL (`run_stat_baselines.py`): all window statistics with L2 logistic regression and with gradient-boosted trees,
  and the K statistics first chosen on an L1 path, refitted with L2 logistic regression ("K-statistics", K = the run's
  budget). A statistic model sees a target's features outside the target's schema as missing in every bin (TRAIN medians).
* Deployment: every registry ships with surrogate chains and is deployed at each target by resolving its chains against the
  target's schema; SRS+T, NMIFS and mRMR are also deployed by zeroing non-executable programs (H4).

## 4. Endpoints

* Primary: AUPRC (average precision) over the pooled episodes of the real site targets, each episode scored under its
  hospital group's schema.
* Secondary: mean AUPRC at MIMIC-IV and HiRID; mean AUPRC over the four controlled targets; AUPRC on source TEST.

## 5. Statistics

* CV seeds: SRS+T, SRS, SRS+T with the likelihood decision and SRS+T on the first study's pools are selected with CV seeds
  0-4 (the seed moves the patient-grouped CV folds; omega is chosen on VAL for each seed); a run's value of such a selector is
  the mean AUPRC of its five registries. The other selectors and the statistic models are deterministic (one value).
* Per comparison, the difference of the 6-run means of an endpoint, with the 95% interval point +- 1.96 sqrt(v_boot +
  v_seed): v_boot is the variance of the difference over a joint patient bootstrap (1,000 replicates; each replicate draws
  eICU, MIMIC-IV and HiRID patients with replacement once and applies the draw to every run, registry, seed and dataset);
  v_seed is the between-seed variance of each seed mean (variance over the five seeds / 5, summed over runs and both
  selectors, divided by 36).
* Per run, the same interval with the run's own variances; runs are counted as better / not different / worse.

## 6. Hypotheses and decision rule

A hypothesis holds when the lower end of the 95% interval of the difference of the 6-run means exceeds its margin: zero for
H1-H6, -0.02 AUPRC for the non-inferiority hypothesis H7.

* H1 (primary): SRS+T has higher primary-endpoint AUPRC than NMIFS.
* H2: SRS+T has higher primary-endpoint AUPRC than mRMR.
* H3: SRS+T has higher mean AUPRC than NMIFS at MIMIC-IV and HiRID.
* H4: deploying with surrogate chains gives higher primary-endpoint AUPRC than zeroing, for each of SRS+T, NMIFS and mRMR.
* H5: SRS, which knows no target schema, has higher primary-endpoint AUPRC than NMIFS.
* H6: SRS+T selected from the extended pools has higher primary-endpoint AUPRC than SRS+T selected from the first study's
  pools.
* H7: SRS+T is non-inferior to the K-statistics model on the primary endpoint (margin 0.02 AUPRC).

Every other comparison is descriptive. All results are reported whatever their direction, and a hypothesis that fails is
reported as failed. Technical failures (for example a crashed job) are fixed without changing any analysis choice, and any
such fix is listed with the results.

## 7. Code

`scripts/v2/run_eicu_source_study.py --variant stat`, phases run (pools, selection, CV seeds), stats (statistic models), dump,
boot and summarize, at the commit that adds this protocol. Outputs: `artifacts/v2/eicu_source2/` and the roots
`union_esrc2_k5`, `union_esrc2_k10`, `union_esrc2_k20`.

## 8. Results (appended on 2026-11-09 after the runs; Sections 1-7 are unchanged)

All six runs (vasopressor 1 h and AKI 6 h at K = 5, 10, 20) completed, with CV seeds 1-4 for SRS+T, SRS, SRS+T with the
likelihood decision and SRS+T on the first study's pools. There was no deviation from the protocol. The predicate generator
added 200 predicates to each task's pool. Technical notes: the summarize phase was started by hand once its inputs existed
(same command); nothing else was run outside the job queue.

| Hypothesis | Difference of 6-run means, AUPRC x 100 [95% interval] | Runs better / n.s. / worse | Verdict |
|---|---|---|---|
| H1 SRS+T - NMIFS, sites | +0.41 [-0.07, +0.88] | 2 / 4 / 0 | fails |
| H2 SRS+T - mRMR, sites | +1.24 [+0.68, +1.80] | 5 / 1 / 0 | holds |
| H3 SRS+T - NMIFS, MIMIC-IV and HiRID | +0.59 [+0.26, +0.92] | 3 / 3 / 0 | holds |
| H4 surrogates - zeroing, sites: SRS+T | +0.44 [+0.02, +0.86] | 1 / 5 / 0 | |
| H4 surrogates - zeroing, sites: NMIFS | +0.03 [-0.29, +0.35] | 0 / 4 / 2 | |
| H4 surrogates - zeroing, sites: mRMR | +0.05 [-0.29, +0.39] | 0 / 6 / 0 | H4 fails |
| H5 SRS - NMIFS, sites | +0.21 [-0.25, +0.66] | 1 / 5 / 0 | fails |
| H6 SRS+T, extended - first study's pools, sites | +4.70 [+3.80, +5.60] | 6 / 0 / 0 | holds |
| H7 SRS+T - K-statistics, sites (margin -2.0) | +1.04 [+0.36, +1.71] | 3 / 0 / 3 | holds |

Per run, H1 was negative in the three vasopressor runs (-0.23, -0.11, -0.27) and positive in the three AKI runs (+1.58,
+0.74, +0.73); H7 was negative in the vasopressor runs (-1.49, -2.29, -1.80, each interval below zero) and positive in the
AKI runs (+2.56, +5.38, +3.87). Intervals from the patient bootstrap alone (without the seed variance) lead to the same
verdicts.

Descriptive (not tested), at the sites: SRS+T 77.7, NMIFS 77.3, K-statistics 76.7, mRMR 76.4 and CMIM 76.4 AUPRC x 100;
SRS+T led every established selector, by 0.4 (NMIFS) to 5.7 points (stratified mRMR); SRS+T - SRS +0.20 [-0.03, +0.42];
SRS+T - SRS+T with the likelihood decision +0.72 [+0.23, +1.21]. Models on all window statistics stayed ahead: logistic
regression by 3.2 points [2.3, 4.1] and gradient-boosted trees by 5.5 [4.8, 6.2]. On the first study's pools SRS+T reached
73.0, below NMIFS on the extended pools by 4.3 points.
