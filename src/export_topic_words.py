"""Export BERTopic topic words for researcher labeling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from bertopic import BERTopic


def run(
    model_path="data/processed/bertopic_model",
    segments_path="data/interim/segments.parquet",
    output_path="data/processed/topic_words.xlsx",
) -> pd.DataFrame:
    topic_model = BERTopic.load(model_path)

    segments = pd.read_parquet(segments_path)
    founder_df = segments[segments["role"] == "founder"].copy()
    topics = list(topic_model.topics_)
    if len(topics) != len(founder_df):
        raise ValueError(
            "Model topic assignments do not match founder segment count: "
            f"{len(topics)} topics vs {len(founder_df)} founder segments."
        )

    topic_counts = Counter(topics)
    feature_names = topic_model.vectorizer_model.get_feature_names_out()
    topic_info = topic_model.get_topic_info()
    topic_row_lookup = {int(topic_id): idx for idx, topic_id in enumerate(topic_info["Topic"].tolist())}

    rows = []
    for topic_id in sorted(topic_counts):
        row_idx = topic_row_lookup[int(topic_id)]
        scores = topic_model.c_tf_idf_.getrow(row_idx).toarray().ravel()
        top_indices = scores.argsort()[-15:][::-1]
        topic_words = [(feature_names[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        row = {
            "Topic_ID": int(topic_id),
            "Segment_Count": int(topic_counts[topic_id]),
        }
        for idx in range(15):
            if idx < len(topic_words):
                word, score = topic_words[idx]
                row[f"Word_{idx + 1:02d}"] = word
                row[f"Score_{idx + 1:02d}"] = round(float(score), 4)
            else:
                row[f"Word_{idx + 1:02d}"] = ""
                row[f"Score_{idx + 1:02d}"] = ""
        row["Your_Label"] = ""
        rows.append(row)

    columns = (
        ["Topic_ID", "Segment_Count"]
        + [f"Word_{idx:02d}" for idx in range(1, 16)]
        + [f"Score_{idx:02d}" for idx in range(1, 16)]
        + ["Your_Label"]
    )
    topic_words_df = pd.DataFrame(rows, columns=columns)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        topic_words_df.to_excel(writer, sheet_name="Topics", index=False)
        worksheet = writer.sheets["Topics"]
        worksheet.freeze_panes = "A2"
        for column_cells in worksheet.columns:
            header = column_cells[0].value
            column_letter = column_cells[0].column_letter
            if header == "Topic_ID":
                width = 10
            elif header == "Segment_Count":
                width = 16
            elif header == "Your_Label":
                width = 30
            elif str(header).startswith("Word_"):
                width = 18
            elif str(header).startswith("Score_"):
                width = 12
            else:
                width = 12
            worksheet.column_dimensions[column_letter].width = width

    print(topic_words_df.to_string(index=False))
    print(f"\nSaved topic words to {output}")
    return topic_words_df


if __name__ == "__main__":
    run()
