"""BERTopic runner for founder-side theme discovery."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def run_bertopic(
    segments_path="data/interim/segments.parquet",
    output_dir="data/processed",
    n_clusters=12,
    random_state=42,
) -> pd.DataFrame:
    try:
        df = pd.read_parquet(segments_path)
        founder_df = df[df["role"] == "founder"].copy()
        print(f"Founder segments: {len(founder_df):,}")
        print(f"Unique startups: {founder_df['startup_id'].nunique():,}")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(
            founder_df["text"].tolist(),
            show_progress_bar=True,
            batch_size=32,
        )
        print(f"Embedding shape: {embeddings.shape}")

        from bertopic import BERTopic
        from sklearn.cluster import KMeans
        from umap import UMAP

        def fit_model(n_components: int):
            umap_model = UMAP(
                n_neighbors=5,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                random_state=random_state,
            )
            cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
            )
            seed_topic_list = [
                ["problem", "pain", "need", "customer", "segment", "market"],
                ["assumption", "hypothesis", "mechanism", "causal", "theory", "belief"],
                ["test", "experiment", "pilot", "mvp", "validate", "feedback"],
                ["churn", "conversion", "retention", "willingness", "pay", "metric"],
                ["pivot", "revise", "update", "iterate", "change", "learn"],
                ["team", "founder", "experience", "background", "co-founder"],
                ["revenue", "subscription", "pricing", "model", "b2b", "saas"],
            ]
            topic_model = BERTopic(
                embedding_model=None,
                umap_model=umap_model,
                hdbscan_model=cluster_model,
                seed_topic_list=seed_topic_list,
                calculate_probabilities=True,
                verbose=True,
            )
            topics, probs = topic_model.fit_transform(
                founder_df["text"].tolist(),
                embeddings=embeddings,
            )
            return topic_model, topics, probs

        try:
            topic_model, topics, probs = fit_model(n_components=5)
        except (MemoryError, ValueError) as error:
            print(
                "WARNING: BERTopic fit_transform failed with "
                f"{type(error).__name__}: {error}. Retrying with UMAP n_components=3."
            )
            topic_model, topics, probs = fit_model(n_components=3)

        founder_df = founder_df.copy()
        founder_df["topic"] = topics

        print(topic_model.get_topic_info())
        for topic_id in sorted(set(topics)):
            if topic_id == -1:
                continue
            words = [word for word, _ in topic_model.get_topic(topic_id)][:8]
            print(f"Topic {topic_id}: {words}")

        pivot = (
            founder_df.groupby("startup_id")["topic"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
        pivot.columns = [f"topic_{column}_share" for column in pivot.columns]
        pivot = pivot.reset_index()

        os.makedirs(output_dir, exist_ok=True)
        pivot.to_csv(Path(output_dir) / "topic_membership.csv", index=False)
        print(pivot.shape)
        print(pivot.head())

        topic_model.save(f"{output_dir}/bertopic_model")
        return pivot
    except Exception as error:
        print(f"ERROR: {error}")
        raise


def update_topic_representation(
    segments_path="data/interim/segments.parquet",
    model_path="data/processed/bertopic_model",
) -> None:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    # Load original founder documents (needed by update_topics)
    df = pd.read_parquet(segments_path)
    docs = df[df["role"] == "founder"]["text"].tolist()
    print(f"Loaded {len(docs):,} founder documents for representation update.")

    # Load saved model
    topic_model = BERTopic.load(model_path)

    # Update representation with English stopwords + bigrams
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
    )
    topic_model.update_topics(docs, vectorizer_model=vectorizer)

    # Print improved topic info
    print("\n=== UPDATED TOPIC INFO ===")
    print(topic_model.get_topic_info())

    print("\n=== TOP 10 WORDS PER TOPIC (stopwords removed) ===")
    for topic_id in sorted(set(topic_model.topics_)):
        if topic_id == -1:
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)][:10]
        print(f"Topic {topic_id:>2}: {words}")

    # Re-save updated model
    topic_model.save(model_path)
    print(f"\nUpdated model saved to {model_path}")
