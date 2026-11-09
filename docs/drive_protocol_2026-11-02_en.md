# Second domain, pre-registration: Backblaze hard-drive failure prediction (written 2026-11-02)

*English translation made on 2026-11-10 for the code release. The Chinese original (`docs/drive_protocol_2026-11-02.md`) is
authoritative; nothing below adds to or changes it. The paper calls this analysis pre-specified rather than pre-registered,
because the protocol was written before any selection on drive data but was not timestamped publicly before the runs. The
protocol and the results of Section 6 concern the drive pools without window-statistic predicates and the likelihood decision
(SRS-C-V); the paper's results on the pools with predicates are post hoc with respect to this protocol.*

**Written**: 2026-11-02, before any selection method had been run on drive data. Only the following had been done [the
original says "four things" and lists three items]:
- a feasibility scan: which attributes each vendor reports, and how many failures each has;
- a trial run of the extraction code on 2024 Q1, to check formats and sample sizes, which led to excluding solid-state drives;
- no candidates had been mined, and there were no model or evaluation results.

**Purpose**: to test whether the method generalizes beyond the ICU. The method is completely frozen; as in the confirmatory
test on the large AKI pool (readiness assessment §5), the hypotheses are written down first and the results looked at afterwards.

## 1. Data and task (`src/icu/v2/extract/backblaze.py`, `scripts/v2/extract_backblaze_cohort.py`)

- **Data**: Backblaze hard-drive data for all of 2024 (Q1–Q4), public, not subject to a data use agreement.
- **Task**: hard-drive failure prediction.
  - **Positives**: a drive marked as failed on day T; the window is the 14 days ending 2 days before T, [T − 15, T − 2]; at
    most one positive per drive.
  - **Controls**: each positive is matched to one drive of the same model (of the same vendor when the model has too few) that
    does not fail during the year, is in the data on the first day of the window, and is still present 30 days after T (or until
    the last day of the data). Sampling without replacement, seed 20261102.
- **Scope**: hard-disk drives only; solid-state system drives are excluded by model name.
- **Features**: the 20 SMART attributes reported by the source vendor (Seagate). Counts and temperatures use raw values;
  attributes 1, 3 and 7, whose encodings are vendor-specific, use normalized values; an attribute that is not reported is
  missing, with no imputation and no scaling.
- **Databases**:
  - source: Seagate, split by drive 70% / 10% / 20%, seed 20261102;
  - targets: HGST, Toshiba and WDC; all of their episodes are used for external evaluation;
  - each vendor's schema: the attributes that appear at least once in ≥ 95% of the episode windows.
- **No simulated targets with missing features**; the target mean is over the three real vendors only.

## 2. Method and comparators (exactly the ICU settings, no adjustment)

- **Generators**: PrefixSpan, ERMiner, matrix profile, shapelets, with the ICU parameters (minsup 0.2, pool cap 200, length
  3–5, top 3 per feature, 4 quantile bins). FMG is not used.
- **Selection**:
  - run with `run_union_selection.py`, known targets HGST, Toshiba and WDC;
  - then `run_union_extra.py` adds SRS-C-V without target schemas, JMI, CMIM, L1 trained with schema dropout, and the stress
    curves.
- **SRS-C-V**: dropout rate q chosen on validation from {0, 0.25}; fast version with shortlist 10; at most 5 surrogates per
  program; probe C chosen on validation.
- **Coarse groups of drive attributes** (`types.DRIVE_GROUP_BY_FEATURE`) for schema dropout, fixed before any result was seen.
- **Budgets**: K = 10 is the main setting, K = 5 and 20 are secondary.

## 3. Hypotheses written down in advance (K = 10)

1. **H1**: SRS-C-V is significantly better than mRMR on internal AUPRC and on mean target AUPRC (both with surrogate deployment).
2. **H2**: SRS-C-V is significantly better than MMRFS and L1 on both metrics.
3. **H3** (source-only setting): SRS-C-V without target schemas is significantly better than mRMR on mean target AUPRC.

Significance: the 95% CI of a patient-clustered paired bootstrap; here the patient is the drive.

## 4. Decision rule

- If H1 and H2 both hold: confirmed in the second domain.
- If only some hold:
  - report them as they are;
  - the paper describes the second domain as a partial replication and analyses why, for example whether programs rely more on
    Seagate-specific attributes, and how different the target distributions are.
- H3 is reported separately.
- Whatever the outcome, it goes into the paper and the documentation; no selective reporting.

## 5. What will not be done

- After seeing selection results, no change of task parameters (gap, window, control rule), features, groups, generator settings
  or method.
- No target is dropped, and the main budget is not changed.
- If a defect in the extraction is found (a data error, not a bad result), it is fixed, the runs are repeated, and the reason and
  the change are recorded in this document.

## 6. Results (appended after the runs completed on 2026-11-02; the pre-registered content above was not changed)

### 6.1 Data

- In 2024 there were 4,634 hard-drive failures, 4,383 of them with a complete window, all matched to a control of the same model.
- Source (Seagate): 3,796 episodes, 1,898 positives; split into 2,657 training, 380 validation and 759 test episodes.
- Targets:

  | Vendor | Episodes | Positives | Schema | Missing attributes |
  |---|---|---|---|---|
  | HGST | 2148 | 1074 | 14 | 187, 188, 190, 240, 241, 242 |
  | Toshiba | 2184 | 1092 | 13 | the 6 above, plus 197 (coverage below 95% over the year) |
  | WDC | 638 | 319 | 14 | as HGST |

- The union pool has 513 candidates.

### 6.2 Verdicts on the pre-registered hypotheses (K = 10)

| Hypothesis | Result |
|---|---|
| H1: vs mRMR, significantly better internally and on the target mean | **does not hold**: internal +0.010 (n.s.), target mean +0.012 (significant) |
| H2: vs MMRFS and L1, significantly better on both metrics | **does not hold**: both significant against L1; against MMRFS internal −0.003 (n.s.), target mean +0.014 (significant) |
| H3: without target schemas, significantly better than mRMR on the target mean | **holds**: +0.003 (significant) |

By the rule of Section 4: **partial replication**.

### 6.3 All results (SRS-C-V minus comparator, internal / target mean, * significant)

| Comparator | K = 5 | K = 10 | K = 20 |
|---|---|---|---|
| mRMR | −0.006 / +0.004* | +0.010 / +0.012* | +0.034* / +0.040* |
| MMRFS | +0.000 / +0.007* | −0.003 / +0.014* | −0.001 / +0.014* |
| CMIM | −0.008 / +0.004* | +0.003 / +0.007* | +0.008 / +0.024* |
| L1 trained with schema dropout | +0.007 / +0.004* | −0.006 / **−0.021 (significantly worse)** | −0.006 / +0.010* |
| JMI, L1, MMR, DPP, MMS, known-target filter + mRMR | target mean significantly better for all, by 0.11–0.24 | same | same |

### 6.4 Interpretation (post hoc; does not change the verdicts)

- **Internal AUPRC has a ceiling**: the good methods all reach 0.86–0.92 internally. They all capture the same dominant signal,
  attribute 187 (reported uncorrectable errors), so the selection method matters little; this is the same phenomenon as the
  convergence of methods on single-generator ICU pools.
- **The advantage when transferring to other vendors largely replicates**: over 3 budgets × all comparators, only one cell is
  worse on the target mean, K = 10 against L1 trained with schema dropout.
  - The reason for that cell: L1 trained with dropout selected more programs on media errors such as attributes 5, 197 and 198,
    which every vendor reports with a consistent meaning;
  - SRS-C-V still relies mainly on 187, which on the targets can only be replaced by surrogates.
- **Putting surrogates into the selection objective is crucial here**: SRS that ignores surrogates during selection
  (`srs_known`) reaches only 0.661 target mean at K = 10, against 0.752 for SRS-C.
- **A new limitation, not visible in the ICU**: some attributes are reported by every vendor but differ in meaning or scale; for
  example, the normalized value of attribute 1 (read error rate) and attribute 193 (load cycle count) have different ranges
  across vendors. A program being executable therefore does not mean that it is still accurate after transfer, while SRS's
  objective only models whether it can execute.
- **The version without target schemas**: at K = 5 validation chose q = 0, so it degenerated into schema-unaware selection, and
  its target mean was 0.124 lower than mRMR's. In this domain, knowing the target schemas therefore helps a lot.
