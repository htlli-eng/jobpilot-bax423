from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_JOBS_PATH = DATA_DIR / "jobs_cleaned.csv"
EMBEDDING_INDEX_PATH = DATA_DIR / "job_embedding_index.csv"
SUMMARY_OUTPUT_PATH = DATA_DIR / "job_coverage_summary.csv"
DESCRIPTION_SCAN_CHARS = 1_200

ROLE_GROUPS = {
    "ML/AI": [
        "machine learning",
        "ml engineer",
        "ai engineer",
        "artificial intelligence",
        "applied scientist",
        "data scientist",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "nlp",
        "deep learning",
    ],
    "Analytics": [
        "data analyst",
        "business analyst",
        "bi analyst",
        "business intelligence",
        "analytics engineer",
        "tableau",
        "dashboard",
        "reporting",
        "sql",
    ],
    "MLOps / Infrastructure": [
        "mlops",
        "ml platform",
        "kubernetes",
        "docker",
        "spark",
        "kafka",
        "aws",
        "gcp",
        "azure",
        "microservices",
        "model serving",
    ],
    "Research": [
        "research scientist",
        "applied scientist",
        "ai research",
        "computer vision",
        "nlp",
        "publication",
        "lab",
        "phd",
    ],
    "Visa / Sponsorship": [
        "h-1b",
        "h1b",
        "sponsorship",
        "visa",
        "immigration",
        "opt",
        "cpt",
    ],
    "Contract / Temp": [
        "contract",
        "contractor",
        "temporary",
        "temp",
    ],
    "Seniority": [
        "senior",
        "sr",
        "staff",
        "principal",
        "lead",
        "director",
        "manager",
        "junior",
        "entry level",
        "intern",
    ],
}
BAY_AREA_TERMS = [
    "bay area",
    "san francisco",
    "san jose",
    "oakland",
    "palo alto",
    "mountain view",
    "sunnyvale",
    "santa clara",
    "berkeley",
    "menlo park",
    "redwood city",
]


def normalize_for_regex(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9+#.-]+", " ", str(value).lower()).strip()


ROLE_GROUP_PATTERNS = {
    group: re.compile(
        "|".join(
            rf"(?<![a-z0-9]){re.escape(normalize_for_regex(term))}(?![a-z0-9])"
            for term in terms
        )
    )
    for group, terms in ROLE_GROUPS.items()
}


def normalize_text(value: object) -> str:
    normalized = normalize_for_regex(value)
    return f" {normalized} " if normalized else " "


def contains_any(text: str, terms: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(normalize_for_regex(term))}(?![a-z0-9])", normalized)
        for term in terms
    )


def build_search_text(jobs: pd.DataFrame) -> pd.Series:
    for column in ["title", "company", "location", "description"]:
        if column not in jobs.columns:
            jobs[column] = ""
    title_company_location = (
        jobs[["title", "company", "location"]].fillna("").astype(str).agg(" ".join, axis=1)
    )
    description = (
        jobs["description"]
        .fillna("")
        .astype(str)
        .str.slice(0, DESCRIPTION_SCAN_CHARS)
    )
    return title_company_location + " " + description


def has_known_salary(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return bool(text and text != "unknown" and re.search(r"\d", text))


def is_us_job(location: object) -> bool:
    text = normalize_text(location)
    return any(term in text for term in [" us ", " usa ", " united states "])


def is_remote_job(search_text: object) -> bool:
    text = normalize_text(search_text)
    return " remote " in text or " work from home " in text


def is_california_or_bay_area(location: object, search_text: object) -> bool:
    combined = f"{location} {search_text}"
    location_text = normalize_text(location)
    normalized = normalize_text(combined)
    if " california " in normalized or " ca us " in location_text or " ca usa " in location_text:
        return True
    return any(term in normalized for term in BAY_AREA_TERMS)


def get_embedded_subset(jobs: pd.DataFrame) -> pd.DataFrame | None:
    if not EMBEDDING_INDEX_PATH.exists():
        return None
    embedding_index = pd.read_csv(EMBEDDING_INDEX_PATH)
    if "row_index" not in embedding_index.columns:
        return None
    row_indices = embedding_index["row_index"].dropna().astype(int)
    row_indices = row_indices[row_indices.between(0, len(jobs) - 1)]
    return jobs.iloc[row_indices.tolist()].copy()


def summarize_dataset(name: str, jobs: pd.DataFrame) -> list[dict[str, object]]:
    jobs = jobs.copy()
    search_text = build_search_text(jobs)
    normalized_search_text = search_text.apply(normalize_text)
    total_jobs = len(jobs)

    rows: list[dict[str, object]] = [
        summary_row(name, "total jobs", total_jobs, total_jobs),
        summary_row(name, "jobs with known salary", jobs["salary"].apply(has_known_salary).sum(), total_jobs),
        summary_row(name, "US jobs", jobs["location"].apply(is_us_job).sum(), total_jobs),
        summary_row(name, "remote jobs", search_text.apply(is_remote_job).sum(), total_jobs),
        summary_row(
            name,
            "California / Bay Area jobs",
            [
                is_california_or_bay_area(location, text)
                for location, text in zip(jobs["location"], search_text, strict=False)
            ].count(True),
            total_jobs,
        ),
    ]

    for group_name, terms in ROLE_GROUPS.items():
        match_count = normalized_search_text.str.contains(
            ROLE_GROUP_PATTERNS[group_name],
            regex=True,
            na=False,
        ).sum()
        rows.append(summary_row(name, group_name, match_count, total_jobs))

    return rows


def summary_row(dataset_name: str, metric: str, count: int, total_jobs: int) -> dict[str, object]:
    share = count / total_jobs if total_jobs else 0
    return {
        "dataset": dataset_name,
        "metric": metric,
        "count": int(count),
        "total_jobs": int(total_jobs),
        "share": round(share, 4),
    }


def print_summary(name: str, jobs: pd.DataFrame) -> None:
    print(f"\n{name}", flush=True)
    print("-" * len(name), flush=True)
    for row in summarize_dataset(name, jobs):
        print(f"{row['metric']}: {row['count']:,} ({row['share']:.1%})", flush=True)


def print_top_titles(name: str, jobs: pd.DataFrame) -> None:
    jobs = jobs.copy()
    search_text = build_search_text(jobs)
    normalized_search_text = search_text.apply(normalize_text)
    print(f"\nTop titles by role group: {name}", flush=True)
    print("-" * (26 + len(name)), flush=True)

    for group_name, terms in ROLE_GROUPS.items():
        matched_jobs = jobs[
            normalized_search_text.str.contains(
                ROLE_GROUP_PATTERNS[group_name],
                regex=True,
                na=False,
            )
        ]
        print(f"\n{group_name} ({len(matched_jobs):,} matches)", flush=True)
        if matched_jobs.empty:
            print("  No matching titles found.", flush=True)
            continue

        top_titles = matched_jobs["title"].fillna("Unknown").value_counts().head(20)
        for title, count in top_titles.items():
            print(f"  {count:>4}  {title}", flush=True)


def print_comparison(summary: pd.DataFrame) -> None:
    full = summary[summary["dataset"] == "Full cleaned dataset"].set_index("metric")
    embedded = summary[summary["dataset"] == "Embedded subset"].set_index("metric")
    if embedded.empty:
        print("\nNo embedded subset found for comparison.", flush=True)
        return

    print("\nFull vs embedded subset comparison", flush=True)
    print("----------------------------------", flush=True)
    for metric in ["ML/AI", "Analytics", "MLOps / Infrastructure", "Research"]:
        full_count = int(full.loc[metric, "count"])
        embedded_count = int(embedded.loc[metric, "count"])
        coverage = embedded_count / full_count if full_count else 0
        print(
            f"{metric}: embedded {embedded_count:,} of full {full_count:,} "
            f"({coverage:.1%} of matching full-dataset jobs)",
            flush=True,
        )


def main() -> None:
    if not CLEANED_JOBS_PATH.exists():
        raise FileNotFoundError(f"Could not find {CLEANED_JOBS_PATH}")

    jobs = pd.read_csv(CLEANED_JOBS_PATH)
    embedded_jobs = get_embedded_subset(jobs)

    summary_rows = summarize_dataset("Full cleaned dataset", jobs)
    print_summary("Full cleaned dataset", jobs)
    print_top_titles("Full cleaned dataset", jobs)

    if embedded_jobs is not None:
        summary_rows.extend(summarize_dataset("Embedded subset", embedded_jobs))
        print_summary("Embedded subset", embedded_jobs)
        print_top_titles("Embedded subset", embedded_jobs)
    else:
        print(f"\nEmbedding index not found or invalid: {EMBEDDING_INDEX_PATH}", flush=True)

    summary = pd.DataFrame(summary_rows)
    print_comparison(summary)

    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"\nSaved coverage summary to: {SUMMARY_OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
