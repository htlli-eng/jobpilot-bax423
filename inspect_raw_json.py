from __future__ import annotations

from pathlib import Path
import json
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_JSON_PATH = DATA_DIR / "techmap-jobs-dump-2021-09.json"
MAX_LINES = 5
CHUNK_SIZE = 64 * 1024
MAX_ARRAY_BYTES = 1024 * 1024
CANDIDATE_FIELD_NAMES = {
    "job title": ["title", "job_title", "jobtitle", "position", "name"],
    "company": ["company", "company_name", "employer", "organization"],
    "location": ["location", "city", "state", "country", "address"],
    "salary": ["salary", "salary_min", "salary_max", "pay", "compensation"],
    "description": ["description", "job_description", "body", "summary"],
    "apply link": ["apply_link", "url", "link", "job_url", "application_url"],
}


def preview_value(value: Any, max_length: int = 120) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def print_record(record: Any, record_number: int) -> None:
    print(f"\nSample record {record_number}:")
    if isinstance(record, dict):
        print(f"Top-level keys: {', '.join(record.keys())}")
        for key, value in list(record.items())[:10]:
            print(f"  {key}: {preview_value(value)}")
    else:
        print(f"Type: {type(record).__name__}")
        print(preview_value(record))


def collect_keys(value: Any, prefix: str = "", max_depth: int = 3) -> set[str]:
    if max_depth < 0:
        return set()

    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            keys.add(full_key)
            keys.update(collect_keys(child, full_key, max_depth - 1))
    elif isinstance(value, list) and value:
        keys.update(collect_keys(value[0], prefix, max_depth - 1))

    return keys


def print_candidate_fields(records: list[Any]) -> None:
    all_keys: set[str] = set()
    for record in records:
        all_keys.update(collect_keys(record))

    print("\nCandidate field matches:")
    for label, candidates in CANDIDATE_FIELD_NAMES.items():
        matches = [
            key
            for key in sorted(all_keys)
            if key.split(".")[-1].lower() in candidates
        ]
        if matches:
            print(f"  {label}: {', '.join(matches)}")
        else:
            print(f"  {label}: no obvious match found")


def inspect_ndjson(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number > MAX_LINES:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Could not parse line {line_number} as JSON: {error}")
                break
    return records


def inspect_json_array(path: Path) -> list[Any]:
    decoder = json.JSONDecoder()
    text = ""

    with path.open("r", encoding="utf-8", errors="replace") as file:
        while len(text) < MAX_ARRAY_BYTES:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            text += chunk

            stripped = text.lstrip()
            if not stripped.startswith("["):
                return []

            array_body = stripped[1:].lstrip()
            records: list[Any] = []
            while array_body and len(records) < MAX_LINES:
                if array_body.startswith("]"):
                    return records
                if array_body.startswith(","):
                    array_body = array_body[1:].lstrip()
                    continue
                try:
                    record, end_index = decoder.raw_decode(array_body)
                except json.JSONDecodeError:
                    break
                records.append(record)
                array_body = array_body[end_index:].lstrip()

            if records:
                return records

    return []


def detect_format(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as file:
        sample = file.read(CHUNK_SIZE).lstrip()

    if not sample:
        return "empty"
    if sample.startswith("["):
        return "json_array"
    if sample.startswith("{"):
        first_line = sample.splitlines()[0]
        try:
            json.loads(first_line)
            return "ndjson"
        except json.JSONDecodeError:
            return "single_json_object_or_unknown"
    return "unknown"


def main() -> None:
    if not RAW_JSON_PATH.exists():
        print(f"Raw JSON file not found: {RAW_JSON_PATH}")
        print("Place the Kaggle file there, then run this script again.")
        return

    file_size_gb = RAW_JSON_PATH.stat().st_size / (1024**3)
    detected_format = detect_format(RAW_JSON_PATH)

    print("Raw JSON schema inspection")
    print(f"File: {RAW_JSON_PATH}")
    print(f"Approx file size: {file_size_gb:.2f} GB")
    print(f"Detected format: {detected_format}")
    print(f"Read limit: first {MAX_LINES} records or about {MAX_ARRAY_BYTES:,} bytes")

    if detected_format == "ndjson":
        records = inspect_ndjson(RAW_JSON_PATH)
    elif detected_format == "json_array":
        records = inspect_json_array(RAW_JSON_PATH)
    else:
        records = inspect_json_array(RAW_JSON_PATH) or inspect_ndjson(RAW_JSON_PATH)

    if not records:
        print("\nNo sample records could be parsed from the safe preview.")
        print("The file may use a nested or compressed format.")
        return

    for index, record in enumerate(records[:MAX_LINES], start=1):
        print_record(record, index)

    print_candidate_fields(records)


if __name__ == "__main__":
    main()
