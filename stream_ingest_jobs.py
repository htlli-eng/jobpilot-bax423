from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import time
from typing import Iterator

import pandas as pd

from extract_jobs_snapshot import map_record
from ingest_jobs import OUTPUT_COLUMNS, clean_batch


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_PATH = DATA_DIR / "raw_jobs.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "jobs_streamed.csv"
DEFAULT_LOG_PATH = DATA_DIR / "stream_ingestion_log.csv"
DEFAULT_BATCH_SIZE = 500


def iter_csv_batches(input_path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(input_path, chunksize=batch_size)


def iter_ndjson_batches(input_path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    records: list[dict[str, str]] = []
    scanned_count = 0

    with input_path.open("r", encoding="utf-8", errors="replace") as input_file:
        for line in input_file:
            scanned_count += 1
            line = line.strip()
            if not line:
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(raw_record, dict):
                continue

            mapped_record = map_record(raw_record, scanned_count)
            if mapped_record is None:
                continue

            records.append(mapped_record)
            if len(records) >= batch_size:
                yield pd.DataFrame(records)
                records = []

    if records:
        yield pd.DataFrame(records)


def iter_input_batches(input_path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    if input_path.suffix.lower() == ".json":
        yield from iter_ndjson_batches(input_path, batch_size)
    else:
        yield from iter_csv_batches(input_path, batch_size)


def stream_ingest_jobs(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    seen_dedupe_keys: set[str] = set()
    streamed_batches: list[pd.DataFrame] = []
    log_rows: list[dict[str, int]] = []

    for batch_id, raw_batch in enumerate(iter_input_batches(input_path, batch_size), start=1):
        if max_batches is not None and batch_id > max_batches:
            break

        incoming_records = len(raw_batch)
        cleaned_batch = clean_batch(raw_batch)
        new_records = cleaned_batch[
            ~cleaned_batch["dedupe_key"].isin(seen_dedupe_keys)
        ].copy()
        new_records = new_records.drop_duplicates(subset="dedupe_key", keep="first")

        kept_records = len(new_records)
        duplicate_records = incoming_records - kept_records
        seen_dedupe_keys.update(new_records["dedupe_key"].tolist())

        if kept_records:
            streamed_batches.append(new_records)

        cumulative_unique_records = len(seen_dedupe_keys)
        log_rows.append(
            {
                "batch_id": batch_id,
                "incoming_records": incoming_records,
                "kept_records": kept_records,
                "duplicate_records": duplicate_records,
                "cumulative_unique_records": cumulative_unique_records,
            }
        )

        print(
            f"Batch {batch_id}: incoming={incoming_records:,}, "
            f"kept={kept_records:,}, duplicates={duplicate_records:,}, "
            f"cumulative_unique={cumulative_unique_records:,}"
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if streamed_batches:
        streamed_jobs = pd.concat(streamed_batches, ignore_index=True)
    else:
        streamed_jobs = pd.DataFrame(columns=OUTPUT_COLUMNS)

    streamed_jobs.to_csv(output_path, index=False)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    print("Streaming ingestion summary")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Log path: {log_path}")
    print(f"Batches processed: {len(log_rows):,}")
    print(f"Final unique records: {len(streamed_jobs):,}")

    return streamed_jobs


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Simulate micro-batch streaming ingestion for JobPilot jobs."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    stream_ingest_jobs(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        sleep_seconds=args.sleep_seconds,
    )
