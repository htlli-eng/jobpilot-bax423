from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "jobs_cleaned.csv"
EMBEDDINGS_OUTPUT_PATH = DATA_DIR / "job_embeddings.npy"
INDEX_OUTPUT_PATH = DATA_DIR / "job_embedding_index.csv"
ANN_INDEX_OUTPUT_PATH = DATA_DIR / "job_annoy_index.ann"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_JOBS = 20_000
BATCH_SIZE = 128
ANNOY_TREES = 50
RANDOM_SEED = 42
TEXT_COLUMNS = ["title", "company", "location", "description"]
INDEX_COLUMNS = ["row_index", "job_id", "title", "company", "location"]
JOB_FAMILY_BUCKETS = {
    "ml_ai_data_science": [
        "machine learning",
        "ml engineer",
        "artificial intelligence",
        "ai engineer",
        "applied scientist",
        "deep learning",
        "pytorch",
        "tensorflow",
        "nlp",
        "computer vision",
        "data scientist",
    ],
    "data_analytics_bi": [
        "data analyst",
        "business intelligence",
        "bi developer",
        "analytics engineer",
        "sql",
        "tableau",
        "power bi",
    ],
    "software_engineering": [
        "software engineer",
        "backend",
        "frontend",
        "full stack",
        "developer",
        "java",
        "python",
        "javascript",
    ],
    "infra_cloud_devops": [
        "devops",
        "cloud engineer",
        "site reliability",
        "sre",
        "kubernetes",
        "docker",
        "aws",
        "azure",
        "gcp",
        "data platform",
        "mlops",
    ],
    "product_project_management": [
        "product manager",
        "project manager",
        "program manager",
        "product owner",
        "scrum",
    ],
    "business_finance_operations": [
        "business analyst",
        "financial analyst",
        "operations analyst",
        "strategy",
        "consulting",
        "supply chain",
    ],
    "marketing_sales_customer": [
        "marketing analyst",
        "growth",
        "sales",
        "account manager",
        "customer success",
    ],
    "design_ux": [
        "ux",
        "ui",
        "product designer",
        "user research",
        "visual designer",
    ],
    "research_science": [
        "researcher",
        "research scientist",
        "scientist",
        "quantitative researcher",
    ],
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def build_job_text(row: pd.Series) -> str:
    parts = [clean_text(row.get(column, "")) for column in TEXT_COLUMNS]
    return "\n".join(part for part in parts if part)


def normalize_for_matching(value: object) -> str:
    return f" {re.sub(r'[^a-z0-9+#.]+', ' ', str(value).lower()).strip()} "


def bucket_pattern(keywords: list[str]) -> re.Pattern:
    pieces = [
        rf"(?<![a-z0-9]){re.escape(normalize_for_matching(keyword).strip())}(?![a-z0-9])"
        for keyword in keywords
    ]
    return re.compile("|".join(pieces))


def load_jobs(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run python3 ingest_jobs.py first."
        )

    jobs = pd.read_csv(path)
    for column in TEXT_COLUMNS + ["job_id"]:
        if column not in jobs.columns:
            jobs[column] = ""
        jobs[column] = jobs[column].fillna("").astype(str)

    jobs["row_index"] = jobs.index
    jobs["embedding_text"] = jobs.apply(build_job_text, axis=1)
    jobs = jobs[jobs["embedding_text"].str.len() > 0].copy()
    return jobs


def select_jobs_for_embedding(
    jobs: pd.DataFrame,
    max_jobs: int = MAX_JOBS,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    rng_seed = RANDOM_SEED
    selected_indices: set[int] = set()
    bucket_counts: dict[str, int] = {}
    bucket_cap = max(1, max_jobs // (len(JOB_FAMILY_BUCKETS) + 1))
    normalized_text = jobs["embedding_text"].apply(normalize_for_matching)

    for bucket_name, keywords in JOB_FAMILY_BUCKETS.items():
        matches = jobs[normalized_text.str.contains(bucket_pattern(keywords), regex=True)]
        available_matches = matches[~matches.index.isin(selected_indices)]
        selected_bucket = available_matches.sample(
            n=min(bucket_cap, len(available_matches)),
            random_state=rng_seed,
        )
        selected_indices.update(selected_bucket.index.tolist())
        bucket_counts[bucket_name] = len(selected_bucket)

    remaining_slots = max_jobs - len(selected_indices)
    remaining_jobs = jobs[~jobs.index.isin(selected_indices)]
    general_count = 0
    if remaining_slots > 0 and not remaining_jobs.empty:
        selected_general = remaining_jobs.sample(
            n=min(remaining_slots, len(remaining_jobs)),
            random_state=rng_seed,
        )
        selected_indices.update(selected_general.index.tolist())
        general_count = len(selected_general)

    selected_jobs = jobs.loc[sorted(selected_indices)].copy()
    selected_jobs = selected_jobs.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    return selected_jobs, bucket_counts, general_count


def print_coverage_summary(
    total_jobs: int,
    max_jobs: int,
    selected_count: int,
    bucket_counts: dict[str, int],
    general_count: int,
) -> None:
    print("Embedding sampling coverage summary")
    print(f"Total jobs in cleaned dataset: {total_jobs:,}")
    print(f"MAX_JOBS: {max_jobs:,}")
    for bucket_name, count in bucket_counts.items():
        print(f"{bucket_name}: {count:,}")
    print(f"general_random: {general_count:,}")
    print(f"Final selected job count: {selected_count:,}")


def build_annoy_index(
    embeddings: np.ndarray,
    output_path: Path = ANN_INDEX_OUTPUT_PATH,
    num_trees: int = ANNOY_TREES,
) -> None:
    embedding_dimension = embeddings.shape[1]
    annoy_index = AnnoyIndex(embedding_dimension, "angular")
    for position, embedding in enumerate(embeddings):
        annoy_index.add_item(position, embedding.tolist())

    print(f"Building Annoy angular index with {num_trees} trees...")
    annoy_index.build(num_trees)
    annoy_index.save(str(output_path))


def build_embeddings() -> None:
    print("Loading cleaned job data...")
    all_jobs = load_jobs()
    jobs, bucket_counts, general_count = select_jobs_for_embedding(all_jobs)
    print_coverage_summary(
        total_jobs=len(all_jobs),
        max_jobs=MAX_JOBS,
        selected_count=len(jobs),
        bucket_counts=bucket_counts,
        general_count=general_count,
    )
    print(f"Model: {MODEL_NAME}")

    if jobs.empty:
        print("No jobs available for embedding.")
        return

    model = SentenceTransformer(MODEL_NAME)
    texts = jobs["embedding_text"].tolist()

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    EMBEDDINGS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)
    jobs[INDEX_COLUMNS].to_csv(INDEX_OUTPUT_PATH, index=False)
    build_annoy_index(embeddings)

    print("Embedding build complete")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved to: {EMBEDDINGS_OUTPUT_PATH}")
    print(f"Index saved to: {INDEX_OUTPUT_PATH}")
    print(f"Annoy index saved to: {ANN_INDEX_OUTPUT_PATH}")


if __name__ == "__main__":
    build_embeddings()
