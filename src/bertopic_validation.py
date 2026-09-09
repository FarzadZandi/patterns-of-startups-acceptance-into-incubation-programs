"""Validate the saved BERTopic model using existing pipeline artifacts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .private_aliases import expand_private_aliases
except ImportError:
    from private_aliases import expand_private_aliases


TOPIC_LABELS = {
    0: "Product/Platform",
    1: "Problem-Solution (informal)",
    2: "Program fit [NOISE]",
    3: "Customer/Business Model",
    4: "Data/Algorithm",
    5: "Q&A filler [NOISE]",
    6: "AI/Cloud/LLMs",
    7: "Market/Competition",
    8: "Revenue/Metrics",
    9: "Program logistics [NOISE]",
    10: "Pitch ceremony [NOISE]",
    11: "Pilot/MVP/Testing",
}
USABLE_TOPICS = (0, 1, 3, 4, 6, 7, 8, 11)

KMEANS_JUSTIFICATION = """KMeans (k=12) was selected over HDBSCAN for topic clustering because
the corpus is small (N=70 startups, ~9,600 founder segments). HDBSCAN
assigns low-density points to a noise cluster (-1), which in small
corpora typically captures 40-60% of documents — rendering the
topic model uninformative. KMeans forces all documents into a cluster,
maximising topic coverage at the cost of some within-cluster coherence,
which is acceptable when the goal is fuzzy set membership scores
rather than sharp topic boundaries."""


def _normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(
        r"\b(now called|new name|or|called)\b.*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[^A-Za-z0-9]+", "", text).casefold()


def _name_aliases(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value)
    aliases = {_normalize_name(text)}
    aliases.update(_normalize_name(part) for part in re.split(r"\n+|/|,", text))
    aliases.update(_normalize_name(part) for part in re.findall(r"\(([^)]*)\)", text))
    aliases.discard("")

    return expand_private_aliases(aliases, _normalize_name)


def _align_topic_startup_ids(
    topics: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    topics = topics.copy()
    known_ids = set(labels["startup_id"].astype(str))
    if set(topics["startup_id"].astype(str)).issubset(known_ids):
        return topics

    alias_to_id: dict[str, str] = {}
    for row in labels[["startup_id", "startup_name"]].itertuples(index=False):
        for alias in _name_aliases(row.startup_name):
            alias_to_id[alias] = str(row.startup_id)

    topics["startup_id"] = topics["startup_id"].apply(
        lambda value: alias_to_id.get(_normalize_name(value), value)
    )
    return topics


def _coherence_score(topic_model, founder_texts: list[str]) -> float | None:
    try:
        from gensim.corpora.dictionary import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except ImportError:
        print("gensim is not installed; run `pip install gensim` to compute coherence.")
        print("Skipping c_v coherence calculation.")
        return None

    texts = [doc.lower().split() for doc in founder_texts]
    dictionary = Dictionary(texts)
    topic_word_lists = []
    for topic_id in sorted(set(topic_model.topics_)):
        if topic_id == -1:
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)][:10]
        topic_word_lists.append(words)

    cm = CoherenceModel(
        topics=topic_word_lists,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence = cm.get_coherence()
    print(f"Mean c_v coherence: {coherence:.4f}")
    print("Acceptable threshold: >= 0.40 for small corpora")
    return float(coherence)


def _decision_comparison(
    topic_path: Path,
    labels_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    topics = pd.read_csv(topic_path)
    labels = pd.read_csv(labels_path)
    topics = _align_topic_startup_ids(topics, labels)
    merged = topics.merge(
        labels[["startup_id", "decision"]],
        on="startup_id",
        how="inner",
    )

    rows = []
    for topic_id in range(12):
        column = f"topic_{topic_id}_share"
        if column not in merged.columns:
            merged[column] = 0.0

        accepted = merged.loc[merged["decision"].eq("Accepted"), column].dropna()
        rejected = merged.loc[
            merged["decision"].isin(["Rejected", "Waitlist"]),
            column,
        ].dropna()
        if len(accepted) and len(rejected):
            p_value = float(
                mannwhitneyu(
                    accepted,
                    rejected,
                    alternative="greater",
                ).pvalue
            )
        else:
            p_value = np.nan

        acc_mean = float(accepted.mean()) if len(accepted) else np.nan
        rej_mean = float(rejected.mean()) if len(rejected) else np.nan
        rows.append(
            {
                "Topic": topic_id,
                "Topic_Label": TOPIC_LABELS[topic_id],
                "Acc_mean": acc_mean,
                "Rej_mean": rej_mean,
                "Difference": acc_mean - rej_mean,
                "p_value": p_value,
                "Significant": "Yes*" if p_value < 0.10 else "No",
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "topic_decision_comparison.csv", index=False)
    print("\nTopic-by-decision comparison:")
    print(comparison.to_string(index=False))
    return comparison


def _ctfidf_topic_scores(
    topic_model,
    documents: list[str],
    assignments: np.ndarray,
) -> np.ndarray:
    document_counts = topic_model.vectorizer_model.transform(documents)
    document_ctfidf = topic_model.ctfidf_model.transform(document_counts)
    similarities = cosine_similarity(document_ctfidf, topic_model.c_tf_idf_)
    outlier_offset = int(getattr(topic_model, "_outliers", 0))
    topic_rows = assignments.astype(int) + outlier_offset
    valid = (topic_rows >= 0) & (topic_rows < similarities.shape[1])
    scores = np.full(len(assignments), np.nan)
    positions = np.arange(len(assignments))[valid]
    scores[valid] = similarities[positions, topic_rows[valid]]
    return scores


def _representative_quotes(
    topic_model,
    founder_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame | None:
    assignments = np.asarray(topic_model.topics_)
    if len(assignments) != len(founder_df):
        print("Topic assignments not aligned — skipping representative quotes.")
        return None

    quotes = founder_df[["startup_id", "text"]].reset_index(drop=True).copy()
    quotes["Topic"] = assignments
    quotes["_word_count"] = quotes["text"].fillna("").str.split().str.len()
    quotes["_topic_score"] = _ctfidf_topic_scores(
        topic_model,
        quotes["text"].fillna("").tolist(),
        assignments,
    )
    quotes = quotes[
        quotes["Topic"].isin(USABLE_TOPICS)
        & quotes["_word_count"].between(10, 60)
    ].copy()
    quotes = quotes.sort_values(
        ["Topic", "_topic_score", "_word_count"],
        ascending=[True, False, False],
    )
    quotes = quotes.groupby("Topic", as_index=False, group_keys=False).head(3)
    quotes["Topic_Label"] = quotes["Topic"].map(TOPIC_LABELS)
    quotes = quotes.rename(columns={"text": "Quote"})
    quotes = quotes[["Topic", "Topic_Label", "startup_id", "Quote"]]
    quotes.to_csv(output_dir / "topic_representative_quotes.csv", index=False)

    print("\nRepresentative quote sample:")
    print(
        quotes.groupby("Topic", as_index=False, group_keys=False)
        .head(2)
        .to_string(index=False)
    )
    return quotes


def run_validation(
    model_path="data/processed/bertopic_model",
    segments_path="data/interim/segments.parquet",
    topic_path="data/processed/topic_membership.csv",
    labels_path="labels/condition_labels.csv",
    output_dir="data/processed",
) -> dict[str, object]:
    """Run validation analyses without fitting or updating the topic model."""
    from bertopic import BERTopic

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    founder_df = pd.read_parquet(segments_path)
    founder_df = founder_df[founder_df["role"].eq("founder")].copy()
    topic_model = BERTopic.load(str(model_path))

    print("=== c_v coherence ===")
    coherence = _coherence_score(
        topic_model,
        founder_df["text"].fillna("").tolist(),
    )

    print("\n=== Topic-by-decision comparison ===")
    comparison = _decision_comparison(
        Path(topic_path),
        Path(labels_path),
        output,
    )

    print("\n=== Representative quotes ===")
    quotes = _representative_quotes(topic_model, founder_df, output)

    print("\n=== KMeans justification ===")
    print(KMEANS_JUSTIFICATION)

    return {
        "coherence": coherence,
        "comparison": comparison,
        "quotes": quotes,
    }
