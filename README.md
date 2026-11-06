# SRS: Selecting Temporal Programs that Survive Schema Shift

Code for the paper *Select Once, Deploy Across Schemas: Choosing Temporal Programs from Heterogeneous Generators*
(PVLDB submission). SRS (schema-robust selection) chooses a registry of K temporal programs (sequential patterns, rules,
motifs, shapelets) from the union of several generators' candidate pools at a source database, by the estimated deployed
value of a frozen logistic probe under schema scenarios; every program carries a chain of surrogate programs, and the
registry compiles to SQL that runs inside each target database. SRS+T additionally knows the targets' schemas.

**No data are included.** The experiments use MIMIC-IV 3.1, eICU-CRD 2.0 and HiRID 1.1.1 from PhysioNet (credentialed
access under each database's data use agreement) and the public Backblaze hard-drive data (2024). Nothing derived from
PhysioNet records (episodes, firings, registries, identifiers) is in this repository; the scripts recreate everything
from the raw tables.

## Layout

| Path | Content |
|---|---|
| `src/icu/v2/` | the method and pipeline: extraction adapters (`extract/`), cohorts and splits, candidate generators (`pools.py`, `baselines/`, `fmg_adapter.py`), SRS (`robust_selection.py`), surrogate chains (`surrogates.py`), established selectors (`selection_baselines.py`), evaluation (`bridge.py`, `pipeline.py`, `stats.py`), SQL compilation (`sql_registry.py`) |
| `scripts/v2/` | drivers for every experiment (one script per analysis; each is resumable or rewrites its own output folder) |
| `scripts/v2/paper_v3/` | the paper's figures (`figs_v5.py`, `figs_motivation.py`), tables (`tables.py`) and summary statistics (`round2.py`) |
| `scripts/v2/run_joint_bootstrap.py` | the primary statistic: a patient bootstrap applied jointly to all 16 runs, with random effects over the six pools |
| `src/icu/` (v1 modules), `scripts/run_mimic_experiment.py`, `nonstationary_miner.py` | the FMG rule miner, reused unchanged as one of the five generators |
| `tests/v2/` | unit tests (`pytest -q tests/v2`) |
| `launchers/` | the exact launch scripts used for the paper (workstation paths) |
| `docs/harmonization_spec.md` | feature mappings of the three ICU databases and the evidence behind them |

## Environment

Python 3.12 with `pip install -r requirements.txt`; Java 21 and SPMF 2.64b for the SPMF generators (see
`tools/spmf/README.md`). Set `CUDA_VISIBLE_DEVICES=""` for anything that imports torch. All commands run from the
repository root. Re-runs reproduce registries exactly under the same BLAS configuration; with multithreaded BLAS, AUPRC can
differ by up to 5e-4 and a probe's penalty can flip between near-tied values (16-run means move by at most 6e-4).

## Reproducing the paper

1. **Extract cohorts** (per source and task; the raw tables under `data/raw/`):
   `python scripts/v2/extract_v2_cohort.py --source mimic_iv --raw-dir <mimic-iv-3.1> --cache-dir data/processed/v2/cache/mimic_iv --out-dir data/processed/v2/extract/mimic_iv_gap1h_full --gap-hours 1`
   (eICU: `--source eicu --match-columns hospitalid`; HiRID: `--source hirid`; AKI: `--task aki_onset --gap-hours 6`).
   Drives: `python scripts/v2/extract_backblaze_cohort.py`.
2. **Track** (split, leakage gate, generators incl. FMG, external firing, pools, supervised references):
   `python scripts/v2/run_full_track.py --gap 1` (and `--gap 2`, `--gap 4`, `--gap 6 --task aki_onset`). The leakage gate
   exits with status 2 on a suspected leak.
3. **Selector tuning and per-generator baselines**: `scripts/v2/run_selector_analysis.py`, then
   `scripts/v2/run_selection_baselines.py [--k 5|10|20]`.
4. **Union pools** (the main experiments): `scripts/v2/run_union_selection.py --pools <pools.pkl> --e1-dir <E1 dir> [--k K]`;
   large pools: re-mine with `scripts/v2/run_discovery_comparison.py` (SPMF `pool_cap` 1000, ten motifs and shapelets per
   feature), then the same driver with `--k 10|20`. Add-on comparators (JMI, CMIM, L1 with schema dropout, stress curves):
   `scripts/v2/run_union_extra.py`.
5. **SRS with the AUPRC decision** (the paper's method): `scripts/v2/run_union_ap.py --phase select`, then `decide`,
   `evaluate`, `ablate`; NMIFS+AP, references and CV-seed replicates: `scripts/v2/run_nmifs_ap.py`,
   `scripts/v2/run_reference_registries.py --phase refs|seeds|compilable`; real targets and hospital schemas:
   `scripts/v2/run_real_targets.py`, `scripts/v2/run_hospital_schemas.py`; decision diagnostics:
   `scripts/v2/run_decision_diagnostics.py`; window-statistic references: `scripts/v2/run_stat_baselines.py`.
6. **Statistics**: `scripts/v2/run_joint_bootstrap.py --phase dump`, `--phase boot`, `--phase summarize`.
7. **SQL execution and timing** (idle machine): `scripts/v2/run_sql_execution.py`,
   `scripts/v2/run_selection_scalability.py`.
8. **Drive domain** (pre-registered): `scripts/v2/run_drive_track.py`.
9. **Figures and tables**: `scripts/v2/paper_v3/round2.py`, then `figs_motivation.py`, `figs_v5.py` and `tables.py` in
   `scripts/v2/paper_v3/`.

Outputs go to `artifacts/v2/` (never committed). Each driver documents its inputs, outputs and options in its module
docstring.
