import pandas as pd


def build_icu_cohort(icustays: pd.DataFrame, min_hours: int = 12, cohort_limit: int = 0) -> pd.DataFrame:
    cohort = icustays.copy()
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    cohort = cohort.dropna(subset=["stay_id", "hadm_id", "intime", "outtime"])
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    cohort["hadm_id"] = cohort["hadm_id"].astype(int)
    if "subject_id" in cohort.columns:
        cohort["subject_id"] = cohort["subject_id"].astype(int)
    cohort["stay_duration_hours"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    cohort = cohort[cohort["stay_duration_hours"] >= float(min_hours)]
    sort_cols = [col for col in ("subject_id", "hadm_id", "stay_id") if col in cohort.columns]
    if sort_cols:
        cohort = cohort.sort_values(sort_cols).reset_index(drop=True)
    if cohort_limit and cohort_limit > 0:
        cohort = cohort.head(int(cohort_limit)).reset_index(drop=True)
    return cohort
