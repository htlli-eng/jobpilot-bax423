from __future__ import annotations

from pathlib import Path
import hashlib
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOT_DATA_PATH = DATA_DIR / "raw_jobs.csv"
SAMPLE_DATA_PATH = DATA_DIR / "jobs_sample.csv"
CLEANED_DATA_PATH = DATA_DIR / "jobs_cleaned.csv"
BATCH_SIZE = 4
KEY_FIELDS = ["title", "company", "location", "description"]
OUTPUT_COLUMNS = [
    "job_id",
    "title",
    "company",
    "location",
    "raw_location",
    "location_source",
    "salary",
    "description",
    "apply_link",
    "dedupe_key",
]


def clean_text(value: object) -> str:
    """Normalize whitespace and remove leading/trailing spaces."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_salary(value: object) -> str:
    """Keep salary readable while normalizing spacing."""
    return clean_text(value)


def make_dedupe_key(row: pd.Series) -> str:
    key_text = "|".join(clean_text(row[field]).lower() for field in KEY_FIELDS)
    return hashlib.sha256(key_text.encode("utf-8")).hexdigest()


def clean_batch(batch: pd.DataFrame) -> pd.DataFrame:
    cleaned = batch.copy()
    cleaned["title"] = cleaned["title"].apply(clean_text)
    cleaned["company"] = cleaned["company"].apply(clean_text)
    cleaned["location"] = cleaned["location"].apply(clean_text)
    if "raw_location" not in cleaned.columns:
        cleaned["raw_location"] = cleaned["location"]
    if "location_source" not in cleaned.columns:
        cleaned["location_source"] = "raw_location_field"
    cleaned["raw_location"] = cleaned["raw_location"].apply(clean_text)
    cleaned["location_source"] = cleaned["location_source"].apply(clean_text)
    cleaned["salary"] = cleaned["salary"].apply(clean_salary)
    cleaned["description"] = cleaned["description"].apply(clean_text)
    cleaned["apply_link"] = cleaned["apply_link"].apply(clean_text)
    cleaned["dedupe_key"] = cleaned.apply(make_dedupe_key, axis=1)
    return cleaned[OUTPUT_COLUMNS]


def get_raw_data_path() -> Path:
    return SNAPSHOT_DATA_PATH if SNAPSHOT_DATA_PATH.exists() else SAMPLE_DATA_PATH


def ingest_jobs(
    raw_path: Path | None = None,
    cleaned_path: Path = CLEANED_DATA_PATH,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    raw_path = raw_path or get_raw_data_path()
    raw_jobs = pd.read_csv(raw_path)
    cleaned_batches: list[pd.DataFrame] = []

    for start in range(0, len(raw_jobs), batch_size):
        batch = raw_jobs.iloc[start : start + batch_size]
        cleaned_batches.append(clean_batch(batch))

    ingested_jobs = pd.concat(cleaned_batches, ignore_index=True)
    cleaned_jobs = ingested_jobs.drop_duplicates(
        subset="dedupe_key", keep="first"
    ).reset_index(drop=True)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_jobs.to_csv(cleaned_path, index=False)

    raw_rows = len(raw_jobs)
    ingested_rows = len(ingested_jobs)
    final_cleaned_rows = len(cleaned_jobs)
    duplicate_rows_removed = ingested_rows - final_cleaned_rows

    print("Job ingestion pipeline summary")
    print(f"Input path: {raw_path}")
    print(f"Raw rows: {raw_rows}")
    print(f"Ingested rows: {ingested_rows}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Final cleaned rows: {final_cleaned_rows}")
    print(f"Saved cleaned data to: {cleaned_path}")

    return cleaned_jobs


if __name__ == "__main__":
    ingest_jobs()
