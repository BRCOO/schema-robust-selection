from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source_table: str
    itemids: tuple[int, ...]
    kind: str = "numeric"
    agg: str = "mean"
    fill: str = "interpolate"
    time_column: str = "charttime"
    value_column: str = "valuenum"
    transform: str | None = None


CORE_FEATURE_SPECS = (
    FeatureSpec("HR", "icu/chartevents.csv.gz", (220045,)),
    FeatureSpec("RR", "icu/chartevents.csv.gz", (220210,)),
    FeatureSpec("SpO2", "icu/chartevents.csv.gz", (220277,)),
    FeatureSpec("SBP", "icu/chartevents.csv.gz", (220179,)),
    FeatureSpec("DBP", "icu/chartevents.csv.gz", (220180,)),
    FeatureSpec("MAP", "icu/chartevents.csv.gz", (220181,)),
    FeatureSpec("Temp", "icu/chartevents.csv.gz", (223762, 223761), transform="temperature_c"),
    FeatureSpec("ventilation_on", "icu/chartevents.csv.gz", (220339, 223848, 223849, 223835), kind="presence", agg="max", fill="zero", value_column="value"),
    FeatureSpec("Lactate", "hosp/labevents.csv.gz", (50813,)),
    FeatureSpec("pH", "hosp/labevents.csv.gz", (50820,)),
    FeatureSpec("Creatinine", "hosp/labevents.csv.gz", (50912,)),
    FeatureSpec("Glucose", "hosp/labevents.csv.gz", (50931,)),
    FeatureSpec("Bilirubin_Total", "hosp/labevents.csv.gz", (50885,)),
    FeatureSpec("WBC", "hosp/labevents.csv.gz", (51300, 51301)),
    FeatureSpec("Platelets", "hosp/labevents.csv.gz", (51265,)),
    FeatureSpec("vasopressor_on", "icu/inputevents.csv.gz", (221289, 221662, 221749, 221906, 222315), kind="presence", agg="max", fill="zero", time_column="starttime", value_column="rate"),
    FeatureSpec("sedation_on", "icu/inputevents.csv.gz", (221668, 221744, 222168), kind="presence", agg="max", fill="zero", time_column="starttime", value_column="rate"),
    FeatureSpec("urine_output", "icu/outputevents.csv.gz", (226566, 226627, 226631, 227489), kind="sum", agg="sum", fill="zero", value_column="value"),
)


def get_feature_specs(feature_set: str = "core") -> tuple[FeatureSpec, ...]:
    if feature_set != "core":
        raise ValueError(f"unknown MIMIC-IV feature set: {feature_set}")
    return CORE_FEATURE_SPECS


def get_feature_names(feature_set: str = "core") -> list[str]:
    return [spec.name for spec in get_feature_specs(feature_set)]
