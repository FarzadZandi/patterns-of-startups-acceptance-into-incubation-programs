"""Founder-side SetFit condition classifiers."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold

try:
    from .private_aliases import expand_private_aliases
except ImportError:
    from private_aliases import expand_private_aliases


CONDITIONS = {
    "PA_high": "PA_high",
    "SF_high": "SF_high",
    "TG_belief": "TG_belief",
    "TG_test": "TG_test",
    "TG_respond": "TG_respond",
}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DECIDED_DECISIONS = ["Accepted", "Rejected", "Waitlist"]


def _normalize_startup_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(now called|new name|or|called)\b.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    return text.casefold()


def _startup_aliases(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value)
    aliases = {_normalize_startup_name(text)}
    aliases.update(_normalize_startup_name(part) for part in re.split(r"\n+|/|,", text))
    aliases.update(_normalize_startup_name(part) for part in re.findall(r"\(([^)]*)\)", text))
    aliases.discard("")

    return expand_private_aliases(aliases, _normalize_startup_name)


def _load_training_frame(labels_path, segments_path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    labels = labels[labels["decision"].isin(DECIDED_DECISIONS)].copy()

    segments = pd.read_parquet(segments_path)
    documents = (
        segments[segments["role"] == "founder"]
        .groupby("startup_id")["text"]
        .apply(lambda values: " [SEP] ".join(values.astype(str)))
        .reset_index(name="text")
    )
    print(f"Startups with documents: {len(documents):,}")

    document_lookup: dict[str, str] = {}
    for row in documents.itertuples(index=False):
        document_lookup[_normalize_startup_name(row.startup_id)] = row.text

    texts: list[str | None] = []
    for row in labels.itertuples(index=False):
        text = None
        for alias in _startup_aliases(row.startup_name):
            if alias in document_lookup:
                text = document_lookup[alias]
                break
        if text is None:
            warnings.warn(
                f"No founder segments found for label startup_id={row.startup_id} "
                f"startup_name={row.startup_name!r}",
                stacklevel=2,
            )
        texts.append(text)

    labels["text"] = texts
    training_df = labels.dropna(subset=["text"]).copy()

    def _truncate(text, max_words=256):
        words = str(text).split()
        return " ".join(words[:max_words])

    training_df["text"] = training_df["text"].apply(_truncate)
    mean_words = training_df["text"].str.split().str.len().mean()
    print(f"Mean document length after truncation: {mean_words:.1f} words")
    return training_df


def _make_dataset(texts, labels):
    from datasets import Dataset

    return Dataset.from_dict({"text": list(texts), "label": [int(label) for label in labels]})


def _train_model(texts, labels, output_dir, num_epochs, num_iterations):
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(MODEL_NAME)
    args = TrainingArguments(
        output_dir=str(output_dir),
        batch_size=8,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        show_progress_bar=False,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
    )
    if not hasattr(args, "eval_strategy"):
        args.eval_strategy = getattr(args, "evaluation_strategy", "no")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_make_dataset(texts, labels),
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()
    return model


def _positive_probability(model, texts) -> np.ndarray:
    proba = np.asarray(model.predict_proba(list(texts)))
    if proba.ndim == 1:
        return proba.astype(float)

    classes = getattr(getattr(model, "model_head", None), "classes_", None)
    if classes is not None and 1 in list(classes):
        positive_idx = list(classes).index(1)
    else:
        positive_idx = min(1, proba.shape[1] - 1)
    return proba[:, positive_idx].astype(float)


def _stratified_predictions(name, texts, labels, output_dir, n_splits=3):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probabilities = np.zeros(len(labels), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(texts, labels), start=1):
        model = _train_model(
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            Path(output_dir) / "_cv" / name / f"stratified_{fold_idx}",
            num_epochs=1,
            num_iterations=5,
        )
        probabilities[test_idx] = _positive_probability(model, [texts[i] for i in test_idx])

    return probabilities


def run_setfit_conditions(
    labels_path="labels/condition_labels.csv",
    segments_path="data/interim/segments.parquet",
    output_dir="data/processed",
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_df = _load_training_frame(labels_path, segments_path)
    all_texts = training_df["text"].tolist()
    result = pd.DataFrame({"startup_id": training_df["startup_id"].tolist()})
    original_labels = pd.DataFrame({"startup_id": training_df["startup_id"].tolist()})

    for idx, (name, column) in enumerate(CONDITIONS.items()):
        labels = training_df[column].astype(int).to_numpy()
        texts = list(all_texts)
        print(f"\n--- Training condition {name} ({idx+1}/{len(CONDITIONS)}) ---")
        print(f"  Positive class: {labels.sum()}/{len(labels)}")
        print("Using 3-fold stratified CV")

        probabilities = _stratified_predictions(
            name=name,
            texts=texts,
            labels=labels,
            output_dir=output_path,
            n_splits=3,
        )

        predictions = (probabilities >= 0.5).astype(int)
        kappa = cohen_kappa_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        print(f"Condition {name}: κ={kappa:.3f}, macro-F1={f1:.3f}")

        final_model = _train_model(
            texts,
            labels,
            output_path / f"setfit_{name}",
            num_epochs=4,
            num_iterations=20,
        )
        final_model.save_pretrained(output_path / f"setfit_{name}")

        result[f"{name}_prob"] = _positive_probability(final_model, texts)
        original_labels[name] = labels

    result.to_csv(output_path / "conditions_setfit.csv", index=False)
    print(result.head(10))

    validity = result.merge(original_labels, on="startup_id")
    corr_rows = []
    for name in CONDITIONS:
        corr_rows.append(
            {
                "condition": name,
                "correlation": validity[f"{name}_prob"].corr(validity[name]),
            }
        )
    corr = pd.DataFrame(corr_rows)
    print("\nCorrelation with original binary labels:")
    print(corr.to_string(index=False))
    print(f"\nSaved conditions_setfit.csv with shape {result.shape}")

    return result


def smoke_test(
    labels_path="labels/condition_labels.csv",
    segments_path="data/interim/segments.parquet",
    output_dir="data/processed",
):
    """Train on only 2 conditions x 6 examples to verify the pipeline runs."""
    from pathlib import Path
    import numpy as np

    print("=== SetFit smoke test ===")
    training_df = _load_training_frame(labels_path, segments_path)

    # Use only the first condition and only 6 balanced examples
    name = "PA_high"
    column = "PA_high"
    labels = training_df[column].astype(int).to_numpy()
    texts = training_df["text"].tolist()

    # Pick 3 positive + 3 negative examples
    pos_idx = np.where(labels == 1)[0][:3]
    neg_idx = np.where(labels == 0)[0][:3]
    tiny_idx = list(pos_idx) + list(neg_idx)
    tiny_texts = [texts[i] for i in tiny_idx]
    tiny_labels = [labels[i] for i in tiny_idx]

    print(f"Training on {len(tiny_texts)} examples (3 pos + 3 neg)")
    model = _train_model(
        tiny_texts, tiny_labels,
        Path(output_dir) / "smoke_test",
        num_epochs=1,
        num_iterations=3,
    )
    proba = _positive_probability(model, tiny_texts)
    print(f"Predicted probabilities: {proba.round(3)}")
    print(f"True labels:             {tiny_labels}")
    print("=== Smoke test complete — pipeline is working ===")
    return True
