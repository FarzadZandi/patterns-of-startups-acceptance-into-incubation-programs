"""Build calibrated fs-QCA input file."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .private_aliases import expand_private_aliases
except ImportError:
    from private_aliases import expand_private_aliases


def _direct_calibrate(series, lo, mid, hi):
    """Fuzzy direct-method calibration with three empirical anchors.

    Maps:  x <= lo   ->  0.05  (fully out)
           x == mid  ->  0.50  (crossover)
           x >= hi   ->  0.95  (fully in)
    Values between anchors are linearly interpolated.
    """
    import numpy as np
    calibrated = np.interp(
        series.values,
        xp=[lo, mid, hi],
        fp=[0.05, 0.50, 0.95]
    )
    calibrated = np.clip(calibrated, 0.05, 0.95)
    return pd.Series(calibrated, index=series.index)


DECIDED_DECISIONS = ["Accepted", "Rejected", "Waitlist"]

PA_MAP = {
    "Absent/Vacuous": 0.05,
    "Shallow belief": 0.10,
    "Tacit pain": 0.20,
    "Theory in use": 0.25,
    "Problem as hypothesis": 0.40,
    "Signal-from-noise awareness": 0.55,
    "Empirically informed framing": 0.75,
    "Validated & structured problem theory": 0.90,
    "Pivot insight & belief revision": 0.95,
}

SF_MAP = {
    "Weak/speculative link": 0.05,
    "Plausible, mechanism described": 0.30,
    "Strong alignment, credible testing": 0.70,
    "High fit, validated in context": 0.90,
}

FEEDBACK_WEIGHTS = {
    "POS_REINF": 0.0,
    "CLARIFY_REQ": 0.2,
    "EXT_SIGNAL": 0.3,
    "DIAG_PROB": 0.5,
    "TEAM_CAP": 0.5,
    "PRESC_ACT": 0.7,
    "TEST_GUIDE": 0.85,
    "DISCONFIRM": 1.0,
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", re.sub(r"\s+", "_", str(col).strip().lower())).strip("_")
        for col in df.columns
    ]
    return df


def _normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return (
        text.casefold()
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("&", "and")
    )


def _calibrate_ordinal(value: object, mapping: dict[str, float], label: str) -> float:
    text = _normalize_text(value)
    for phrase, fuzzy in mapping.items():
        if _normalize_text(phrase) in text:
            return fuzzy
    if not pd.isna(value):
        print(f"WARNING: Could not map {label} ordinal value: {value!r}")
    return np.nan


def _normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(now called|new name|or|called)\b.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    return text.casefold()


def _name_aliases(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value)
    aliases = {_normalize_name(text)}
    aliases.update(_normalize_name(part) for part in re.split(r"\n+|/|,", text))
    aliases.update(_normalize_name(part) for part in re.findall(r"\(([^)]*)\)", text))
    aliases.discard("")

    return expand_private_aliases(aliases, _normalize_name)


def _read_ordinal_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="CF_Coding_Batch04")
    df = _normalize_columns(df)
    required = ["startup_id", "startup_name", "problem_articulation", "solution_fit"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df[required].copy()
    out["PA_fuzzy"] = out["problem_articulation"].apply(
        lambda value: _calibrate_ordinal(value, PA_MAP, "PA")
    )
    out["SF_fuzzy"] = out["solution_fit"].apply(
        lambda value: _calibrate_ordinal(value, SF_MAP, "SF")
    )
    return out[["startup_id", "PA_fuzzy", "SF_fuzzy"]]


def _binary_fuzzy(series: pd.Series) -> pd.Series:
    return series.astype(int).map({0: 0.05, 1: 0.95})


def _load_topics(topic_path: Path, condition_labels: pd.DataFrame) -> pd.DataFrame:
    topics = pd.read_csv(topic_path)
    keep = {
        "topic_3_share": "T_customer",
        "topic_7_share": "T_market",
        "topic_8_share": "T_revenue",
        "topic_11_share": "T_testing",
    }
    missing = [col for col in keep if col not in topics.columns]
    if missing:
        raise ValueError(f"{topic_path} is missing required topic columns: {missing}")

    direct_ids = set(condition_labels["startup_id"])
    if not set(topics["startup_id"]).issubset(direct_ids):
        alias_to_id: dict[str, str] = {}
        for row in condition_labels.itertuples(index=False):
            for alias in _name_aliases(row.startup_name):
                alias_to_id[alias] = row.startup_id
        topics = topics.copy()
        topics["startup_id"] = topics["startup_id"].apply(
            lambda value: alias_to_id.get(_normalize_name(value), value)
        )
        unmapped = sorted(set(topics["startup_id"]) - direct_ids)
        if unmapped:
            print(f"WARNING: Topic rows not mapped to condition-label startup_id: {unmapped}")

    topics = topics[["startup_id", *keep.keys()]].rename(columns=keep)

    # --- Recalibrate topic shares with direct method ---
    # Raw shares are all below 0.5 - calibrate before saving.
    # Anchors: p10 (fully out), p50 (crossover), p90 (fully in)
    # computed from the empirical distribution across 70 startups.
    topics["T_testing_f"] = _direct_calibrate(
        topics["T_testing"], lo=0.0085, mid=0.0286, hi=0.0635
    )
    topics["T_customer_f"] = _direct_calibrate(
        topics["T_customer"], lo=0.0386, mid=0.0859, hi=0.1812
    )
    topics["T_market_f"] = _direct_calibrate(
        topics["T_market"], lo=0.0283, mid=0.0619, hi=0.1341
    )
    topics["T_revenue_f"] = _direct_calibrate(
        topics["T_revenue"], lo=0.0141, mid=0.0578, hi=0.1402
    )
    return topics


def _write_r_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '''library(QCA)
d <- read.csv("../data/processed/qca_calibrated.csv", row.names="startup_id")
conds <- c("PA_fuzzy","SF_fuzzy","TGB_f","TGT_f","TGR_f","TEAM_f","PLAT_f","B2B_f")

cat("\\n=== NECESSITY (ladder outcome) ===\\n")
print(superSubset(d, outcome="OUTCOME_ladder", conditions=conds,
      relation="necessity", incl.cut=0.9))

cat("\\n=== NECESSITY (accept outcome) ===\\n")
print(superSubset(d, outcome="OUTCOME_accept", conditions=conds,
      relation="necessity", incl.cut=0.9))

cat("\\n=== TRUTH TABLE (accept outcome) ===\\n")
tt <- truthTable(d, outcome="OUTCOME_accept", conditions=conds,
                 incl.cut=0.8, n.cut=1, show.cases=TRUE, sort.by="incl")
print(tt)

cat("\\n=== MINIMIZATION (intermediate solution) ===\\n")
print(minimize(tt, details=TRUE, include="?"))
''',
        encoding="utf-8",
    )


def run_calibration(
    batch3_path="labels/Batch_3_CF_Coding.xlsx",
    batch4_path="labels/Batch_4_CF_Coding.xlsx",
    batch5_path="labels/Batch_5_CF_Coding.xlsx",
    condition_labels_path="labels/condition_labels.csv",
    feedback_labels_path="labels/feedback_labels.csv",
    topic_path="data/processed/topic_membership.csv",
    output_path="data/processed/qca_calibrated.csv",
    r_script_path="qca/fsqca.R",
) -> pd.DataFrame:
    batch_paths = [Path(batch3_path), Path(batch4_path), Path(batch5_path)]
    missing = [str(path) for path in batch_paths if not path.exists()]
    if missing:
        print("Missing batch coding Excel files:")
        for path in missing:
            print(f"  - {path}")
        print("Please copy Batch_3_CF_Coding.xlsx, Batch_4_CF_Coding.xlsx, and Batch_5_CF_Coding.xlsx into cf-nlp-qca/labels/ first.")
        return pd.DataFrame()

    ordinal = pd.concat([_read_ordinal_file(path) for path in batch_paths], ignore_index=True)
    print(f"Ordinal PA/SF rows: {len(ordinal)}")

    condition_labels = pd.read_csv(condition_labels_path)
    condition = condition_labels[
        [
            "startup_id",
            "startup_name",
            "decision",
            "TG_belief",
            "TG_test",
            "TG_respond",
            "TEAM_struct",
            "PLATFORM",
            "B2B_pure",
        ]
    ].copy()
    condition["TGB_f"] = _binary_fuzzy(condition["TG_belief"])
    condition["TGT_f"] = _binary_fuzzy(condition["TG_test"])
    condition["TGR_f"] = _binary_fuzzy(condition["TG_respond"])
    condition["TEAM_f"] = _binary_fuzzy(condition["TEAM_struct"])
    condition["PLAT_f"] = condition["PLATFORM"].astype(int).map({0: 0.95, 1: 0.05})
    condition["B2B_f"] = _binary_fuzzy(condition["B2B_pure"])
    condition = condition[["startup_id", "startup_name", "decision", "TGB_f", "TGT_f", "TGR_f", "TEAM_f", "PLAT_f", "B2B_f"]]
    print(f"Condition-label rows: {len(condition)}")

    topics = _load_topics(Path(topic_path), condition_labels)
    print(f"Topic-membership rows: {len(topics)}")

    feedback = pd.read_csv(feedback_labels_path)
    feedback = feedback[["startup_id", "decision", *FEEDBACK_WEIGHTS.keys()]].copy()
    def _ladder_mean(row):
        weights = [w for c, w in FEEDBACK_WEIGHTS.items() if int(row[c]) == 1]
        return float(np.mean(weights)) if weights else 0.5

    def _ladder_max(row):
        weights = [w for c, w in FEEDBACK_WEIGHTS.items() if int(row[c]) == 1]
        return float(max(weights)) if weights else 0.5

    def _ladder_dominant(row):
        # Single highest-weight code; if tie, take the max
        active = {c: w for c, w in FEEDBACK_WEIGHTS.items() if int(row[c]) == 1}
        return float(max(active.values())) if active else 0.5

    feedback["OUTCOME_ladder"] = feedback.apply(_ladder_mean, axis=1)
    feedback["OUTCOME_ladder_max"] = feedback.apply(_ladder_max, axis=1)
    feedback["OUTCOME_ladder_dominant"] = feedback.apply(_ladder_dominant, axis=1)
    feedback["OUTCOME_accept"] = feedback["decision"].eq("Accepted").astype(int).map({0: 0.05, 1: 0.95})
    feedback = feedback[
        [
            "startup_id",
            "OUTCOME_ladder",
            "OUTCOME_ladder_max",
            "OUTCOME_ladder_dominant",
            "OUTCOME_accept",
        ]
    ]
    print(f"Feedback-label rows: {len(feedback)}")

    joined = condition.merge(ordinal, on="startup_id", how="inner")
    print(f"Rows after condition + ordinal join: {len(joined)}")
    joined = joined.merge(topics, on="startup_id", how="inner")
    print(f"Rows after topic join: {len(joined)}")
    joined = joined.merge(feedback, on="startup_id", how="inner")
    print(f"Rows after feedback join: {len(joined)}")
    joined = joined[joined["decision"].isin(DECIDED_DECISIONS)].copy()
    print(f"Rows after decided-decision filter: {len(joined)}")

    final_columns = [
        "startup_id",
        "decision",
        "PA_fuzzy",
        "SF_fuzzy",
        "TGB_f",
        "TGT_f",
        "TGR_f",
        "TEAM_f",
        "PLAT_f",
        "B2B_f",
        "T_customer",
        "T_customer_f",
        "T_market",
        "T_market_f",
        "T_revenue",
        "T_revenue_f",
        "T_testing",
        "T_testing_f",
        "OUTCOME_ladder",
        "OUTCOME_ladder_max",
        "OUTCOME_ladder_dominant",
        "OUTCOME_accept",
    ]
    final = joined[final_columns].copy()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False)
    _write_r_script(Path(r_script_path))

    print(f"\nShape of final file: {final.shape}")
    print("\nFirst 5 rows:")
    print(final.head().to_string(index=False))

    fuzzy_cols = [col for col in final_columns if col not in ["startup_id", "decision"]]
    print("\nDistribution of each fuzzy condition (min, mean, max):")
    print(final[fuzzy_cols].agg(["min", "mean", "max"]).T.to_string())

    print("\nOUTCOME_ladder distribution:")
    print(final["OUTCOME_ladder"].describe().to_string())

    print("\nAccepted vs Rejected rows:")
    print(final["decision"].value_counts().to_string())
    print(f"\nSaved calibrated QCA input to {output}")
    print(f"Wrote fs-QCA script to {r_script_path}")

    return final
