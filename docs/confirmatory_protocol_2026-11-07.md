# Confirmatory study: a second source with real site schemas (eICU hospitals)

Protocol written on 2026-11-07, before any selection, firing or evaluation of this study. It is committed to the code
repository before the first run; results will be appended in a separate section without changing Sections 1-6.

## 1. Purpose

The paper's evidence comes from one source (MIMIC-IV) whose real targets lack at most one harmonized feature, and two
components of the method (the AUPRC decision and the random scenarios) were added after results had been seen. This study
tests the frozen method on a second source, with real targets that are sites with their own schemas: eICU hospitals that do
not record some features.

## 2. Method, frozen

SRS+T and SRS exactly as in the paper and the code at this commit: greedy forward selection with Newton screening of all
candidates, shortlist m = 10 refitted exactly, AUPRC decision among the shortlist, likelihood minimal gain 1e-4 bits;
V = 3 folds, fold-probe penalty eta_0 = 1; surrogate chains with M = 5; scenarios: source schema, the known target schemas
(SRS+T only) and D = 20 random coarse-group dropout schemas, dropout rate omega in {0, 0.25} chosen on VAL AUPRC; shipped
probe class-balanced L2 logistic regression, penalty from {0.01, 0.1, 1, 10} on VAL. CV seed 0 (the shipped registry). No
parameter, rule or implementation detail will be changed after this protocol.

## 3. Data and design

* Tasks and cohorts: the paper's eICU-CRD cohorts of the vasopressor task (1 h gap) and the AKI task (6 h gap), as
  extracted for the paper (no re-extraction).
* Hospital schemas, without labels (as in the paper): a measured feature is present at a hospital when it is observed in at
  least 5% of the hospital's windows; a status feature (sedation, vasopressor) when it is "on" in at least one window.
  Hospitals with fewer than 20 episodes in a cohort are excluded from both roles (their schema cannot be estimated).
* Source: hospitals whose schema contains every harmonized eICU feature (vasopressor task: 53 hospitals, 4,671 episodes;
  AKI: 61 hospitals, 11,326 episodes). Patient-level split into TRAIN / VAL / TEST (70/10/20), seed 20261107.
* Real site targets (primary): hospitals whose schema lacks at least one feature (vasopressor task: 21 hospitals, 1,816
  episodes in 6 schema groups; AKI: 43 hospitals, 6,280 episodes in 12 groups). Each hospital group is a target with its
  schema; SRS+T knows these schemas (catalog metadata), SRS does not. Target episodes of patients who also have source
  episodes are removed.
* Other databases: MIMIC-IV and HiRID, all episodes of the task's cohort (both record every source feature).
* Controlled targets on source TEST: no labs, no blood pressure, no urine output, vital signs only.
* Pools: the four non-FMG generators of the paper (PrefixSpan, ERMiner, matrix-profile motifs, shapelets) with the paper's
  settings, mined on source TRAIN (as in the paper's re-split runs). FMG is not run.
* Budgets: K in {5, 10, 20}; 6 runs (2 tasks x 3 budgets).
* Comparators: the twelve established selectors of the paper (NMIFS, mRMR, generator-stratified mRMR, JMI, CMIM, MMRFS,
  MMR, MMS, DPP, L1, L1 with schema dropout, Filter+mRMR), tuned on VAL by the paper's drivers; SRS+T with the likelihood
  decision for reference.
* Deployment: every registry ships with surrogate chains and is deployed at each target by resolving its chains against the
  target's schema (the paper's rule); also by zeroing non-executable programs (for H4).

## 4. Endpoints

* Primary: AUPRC (average precision) over the pooled episodes of the real site targets, each episode scored under its
  hospital's schema.
* Secondary: mean AUPRC at MIMIC-IV and HiRID; mean AUPRC over the four controlled targets; AUPRC on source TEST.

## 5. Statistics

* Per comparison, the difference of the 6-run means of an endpoint, with a 95% percentile interval from a joint patient
  bootstrap (1,000 replicates): each replicate draws eICU, MIMIC-IV and HiRID patients with replacement once and applies the
  draw to every run, registry and dataset.
* Per run, the same bootstrap restricted to the run gives a 95% interval; runs are counted as better / not different / worse.

## 6. Hypotheses and decision rule

A hypothesis holds when the 95% interval of the difference of the 6-run means lies entirely above zero.

* H1 (primary): SRS+T has higher primary-endpoint AUPRC than NMIFS.
* H2: SRS+T has higher primary-endpoint AUPRC than mRMR.
* H3: SRS+T has higher mean AUPRC than NMIFS at MIMIC-IV and HiRID.
* H4: deploying with surrogate chains gives higher primary-endpoint AUPRC than zeroing, for each of SRS+T, NMIFS and mRMR.
* H5: SRS, which knows no target schema, has higher primary-endpoint AUPRC than NMIFS.

Every other comparison is descriptive. All results are reported whatever their direction, and a hypothesis that fails is
reported as failed. Technical failures (for example a crashed job) are fixed without changing any analysis choice, and any
such fix is listed with the results.

## 7. Results (appended on 2026-11-07 after the runs; Sections 1-6 are unchanged)

All six runs (vasopressor 1 h and AKI 6 h at K = 5, 10, 20) completed; there was no deviation from the protocol. The
primary endpoint pools 1,814 vasopressor and 6,267 AKI episodes of the site targets (2 and 13 episodes of patients with
source episodes were removed). Code: `scripts/v2/run_eicu_source_study.py` (phases build, run, dump, boot, summarize).

| Hypothesis | Difference of 6-run means, AUPRC x 100 [95% interval] | Runs better / n.s. / worse | Verdict |
|---|---|---|---|
| H1 SRS+T - NMIFS, sites | +0.54 [+0.05, +1.00] | 2 / 4 / 0 | holds |
| H2 SRS+T - mRMR, sites | +1.96 [+1.40, +2.44] | 4 / 2 / 0 | holds |
| H3 SRS+T - NMIFS, MIMIC-IV and HiRID | +1.26 [+1.00, +1.50] | 3 / 3 / 0 | holds |
| H4 surrogates - zeroing, sites: SRS+T | +0.68 [+0.46, +0.90] | 2 / 4 / 0 | |
| H4 surrogates - zeroing, sites: NMIFS | +0.15 [+0.05, +0.25] | 3 / 3 / 0 | |
| H4 surrogates - zeroing, sites: mRMR | +0.07 [-0.04, +0.18] | 1 / 5 / 0 | H4 fails |
| H5 SRS - NMIFS, sites | -0.47 [-1.01, +0.05] | 0 / 5 / 1 | fails |

Descriptive (not tested): at the sites SRS+T led every established selector, from CMIM (+1.39 [+0.95, +1.81]) to JMI
(+5.07 [+4.41, +5.70]); SRS+T - SRS +1.01 [+0.54, +1.45]; SRS+T - SRS+T with the likelihood decision +0.90 [+0.54, +1.21].
