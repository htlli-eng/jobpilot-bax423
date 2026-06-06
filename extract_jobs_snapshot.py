from __future__ import annotations

from pathlib import Path
import csv
import html
import json
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "techmap-jobs-dump-2021-09.json"
OUTPUT_PATH = DATA_DIR / "raw_jobs.csv"
MAX_VALID_RECORDS = 50_000
PROGRESS_EVERY = 5_000
DESCRIPTION_MAX_CHARS = 3_000
OUTPUT_FIELDS = [
    "job_id",
    "title",
    "company",
    "location",
    "raw_location",
    "location_source",
    "salary",
    "description",
    "apply_link",
]


def get_path(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_value(record: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        value = get_path(record, path)
        if value not in (None, ""):
            return value
    return None


def clean_text(value: Any, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def extract_job_id(record: dict[str, Any], scanned_count: int) -> str:
    job_id = first_value(record, ["idInSource", "_id.$oid"])
    return clean_text(job_id) or f"row-{scanned_count}"


def extract_title(record: dict[str, Any]) -> str:
    value = first_value(
        record,
        [
            "json.schemaOrg.title",
            "position.name",
            "position",
            "name",
        ],
    )
    return clean_text(value)


def extract_company(record: dict[str, Any]) -> str:
    value = first_value(
        record,
        [
            "orgCompany.name",
            "json.schemaOrg.hiringOrganization.name",
        ],
    )
    return clean_text(value) or "Unknown"


def extract_location(record: dict[str, Any]) -> tuple[str, str, str]:
    schema_address = get_path(record, "json.schemaOrg.jobLocation.address")
    if isinstance(schema_address, dict):
        parts = [
            clean_text(schema_address.get("addressLocality")),
            clean_text(schema_address.get("addressRegion")),
            clean_text(schema_address.get("addressCountry")),
        ]
        location = ", ".join(part for part in parts if part) or clean_text(schema_address)
        if location:
            return location, clean_text(schema_address), "job_posting_location"

    schema_location = get_path(record, "json.schemaOrg.jobLocation")
    if schema_location:
        location = clean_text(schema_location)
        return location, location, "job_posting_location"

    address_parts = [
        clean_text(get_path(record, "orgAddress.city")),
        clean_text(get_path(record, "orgAddress.state")),
        clean_text(get_path(record, "orgAddress.country")),
    ]
    org_location = ", ".join(part for part in address_parts if part)
    if org_location:
        return org_location, org_location, "company_location_if_only_available"

    return "Unknown", "", "unknown"


def extract_salary(record: dict[str, Any]) -> str:
    value = first_value(
        record,
        [
            "salary",
            "json.schemaOrg.baseSalary",
            "json.schemaOrg.estimatedSalary",
        ],
    )
    return clean_text(value) or "Unknown"


def extract_description(record: dict[str, Any]) -> str:
    value = first_value(
        record,
        [
            "text",
            "json.schemaOrg.description",
            "html",
        ],
    )
    return clean_text(value, max_chars=DESCRIPTION_MAX_CHARS)


def extract_apply_link(record: dict[str, Any]) -> str:
    value = first_value(record, ["url", "json.schemaOrg.url"])
    return clean_text(value)


def map_record(record: dict[str, Any], scanned_count: int) -> dict[str, str] | None:
    title = extract_title(record)
    description = extract_description(record)

    if not title or not description:
        return None
    location, raw_location, location_source = extract_location(record)

    return {
        "job_id": extract_job_id(record, scanned_count),
        "title": title,
        "company": extract_company(record),
        "location": location,
        "raw_location": raw_location,
        "location_source": location_source,
        "salary": extract_salary(record),
        "description": description,
        "apply_link": extract_apply_link(record),
    }


def extract_snapshot() -> None:
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records_scanned = 0
    valid_records_extracted = 0

    with INPUT_PATH.open("r", encoding="utf-8", errors="replace") as input_file:
        with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for line in input_file:
                if valid_records_extracted >= MAX_VALID_RECORDS:
                    break

                records_scanned += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(record, dict):
                    continue

                mapped_record = map_record(record, records_scanned)
                if mapped_record is None:
                    continue

                writer.writerow(mapped_record)
                valid_records_extracted += 1

                if valid_records_extracted % PROGRESS_EVERY == 0:
                    print(f"Extracted {valid_records_extracted:,} valid records...")

    print("Job snapshot extraction summary")
    print(f"Records scanned: {records_scanned:,}")
    print(f"Valid records extracted: {valid_records_extracted:,}")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_snapshot()
