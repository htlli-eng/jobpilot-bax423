# JobPilot

JobPilot is a BAX-423 final project MVP skeleton for a smart job matcher and resume builder.

This first version is intentionally simple. It loads a local sample job dataset, lets a user enter job preferences, ranks postings with a transparent scoring function, and allows the top 10 matches to be downloaded as CSV.

## Features

- Streamlit web app
- Local CSV job data
- Simple local ingestion and deduplication pipeline
- Sidebar inputs for target role, skills, preferred location, minimum salary, and dealbreakers
- Resume PDF upload with simple keyword-based skill extraction
- Start Matching button so users can review profile inputs before ranking jobs
- Embedding-based candidate retrieval with rule-based re-ranking when embeddings are available
- Multi-stage ranking with target role relevance, skill overlap, location fit, salary fit, dealbreaker penalties, and embedding similarity bonus
- Target-role-aware role fit scoring for ML/AI, analytics, MLOps/infrastructure, and research roles
- Role fit strictness control: Flexible allows broader exploration, Balanced is the default, and Strict prioritizes jobs closely aligned with the target role
- Structured rule-based re-ranking controls for seniority preference, capped minimum salary, maximum required years of experience, contract roles, sponsorship, company type, defense/military avoidance, and salary strictness
- General constraint extraction and explainability for years required, employment type, contract/temp risk, unpaid or commission-only roles, seniority, junior/entry-level roles, manager/director roles, defense or clearance signals, sponsorship signals, location flags, and inferred job family
- Transparent location handling with `raw_location`, `location_source`, `location_confidence`, remote/hybrid/US/California/Bay Area flags, and location pass/fail diagnostics
- Active filter pass/fail columns for experience limits, contract/temp avoidance, unpaid/commission-only avoidance, seniority, manager/director roles, and location strictness
- Cleaner salary display and stronger US/location preference handling for real job data
- Clean Top 10 ranked jobs table with user-facing recommendation columns
- Debug expander with detailed constraint and location checks for persona evaluation
- User-facing CSV download with title, company, location, description, apply link, match score, skills, and explanation
- Full debug CSV download for persona pass-criteria and Precision@10 analysis
- Adaptive feedback learning with Accept / Reject / Skip rewards and transparent feedback-based re-ranking

## Project Structure

```text
.
├── app.py
├── build_embeddings.py
├── embedding_utils.py
├── extract_jobs_snapshot.py
├── ingest_jobs.py
├── inspect_raw_json.py
├── resume_utils.py
├── stream_ingest_jobs.py
├── requirements.txt
├── README.md
└── data
    ├── jobs_sample.csv
    └── jobs_cleaned.csv
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Inspect the raw Kaggle JSON schema safely:

```bash
python inspect_raw_json.py
```

This looks for `data/techmap-jobs-dump-2021-09.json` and reads only a small preview so the full raw file is not loaded into memory.

Extract an offline job snapshot from the raw NDJSON file:

```bash
python3 extract_jobs_snapshot.py
```

This streams `data/techmap-jobs-dump-2021-09.json` line by line, extracts up to 50,000 valid postings, and writes `data/raw_jobs.csv`.

Run the local ingestion pipeline:

```bash
python3 ingest_jobs.py
```

This reads `data/raw_jobs.csv` when it exists, otherwise falls back to `data/jobs_sample.csv`. It simulates batch ingestion, cleans key fields, removes duplicate postings, and writes `data/jobs_cleaned.csv`.

Run the simulated streaming ingestion pipeline:

```bash
python3 stream_ingest_jobs.py --input data/raw_jobs.csv --output data/jobs_streamed.csv --batch-size 500 --max-batches 3
```

JobPilot uses the Kaggle NDJSON file as the external job source. `extract_jobs_snapshot.py` creates a manageable `data/raw_jobs.csv` snapshot from that large source. `ingest_jobs.py` performs batch cleaning and deduplication for the standard app dataset. `stream_ingest_jobs.py` demonstrates the streaming-style pipeline requirement by reading incoming jobs in micro-batches, cleaning each batch, deduplicating across all batches, appending structured records, and writing `data/jobs_streamed.csv` plus `data/stream_ingestion_log.csv`. This simulates real-time ingestion without requiring live API calls during every app run.

The main ranked jobs table is intentionally demo-friendly: it shows clean recommendation fields such as rank, title, company, location, salary, required years, inferred employment type, matched skills, final score, and active-constraint pass/fail status. Detailed internal fields are kept in the `Debug: constraint checks` expander and the full debug CSV.

The top jobs CSV is user-facing and includes assignment-required job details: title, company, location, description, apply URL, final score, matched skills, and a concise match explanation. A separate full debug CSV includes all ranked columns for persona pass-criteria analysis and Precision@10 evaluation.

Location is treated as the job posting work location, office location, or remote location when the raw source provides one. JobPilot should not present company headquarters as the job location by default. If only company-level or ambiguous location metadata is available, the app labels it with `location_source` such as `company_location_if_only_available` or `ambiguous_location` and lower `location_confidence`. Older cleaned datasets without provenance columns still load, but re-running extraction and ingestion will preserve newer `raw_location` and `location_source` fields.

JobPilot also extracts general job constraints from the posting text using simple regex and keyword rules. Extracted fields include experience requirement (`required_years_min`, `years_requirement_text`), inferred employment type, contract/temp status, unpaid/commission-only status, seniority flags, manager/director flags, company size signals, defense/military/clearance signals, and location flags. Defense and military detection uses title, company, location, and description signals, including clearance requirements, government contractor indicators, known defense contractor names, and defense-related locations. These fields support transparent rule-based re-ranking and persona pass-criteria evaluation. Active sidebar filters produce pass/fail columns such as `passes_years_filter`, `passes_contract_filter`, `passes_unpaid_filter`, and `passes_all_active_constraints`.

Company size inference is approximate unless the source data provides an exact employee count. The current cleaned Kaggle snapshot does not include an employee-count column, so JobPilot uses low-confidence signals from company name and posting text, such as startup/seed-stage/small-team language or Fortune 500/global/enterprise language. The debug table exposes `company_size_inferred`, `company_size_confidence`, `company_size_match_text`, and `passes_company_size_filter` for transparency.

Rule-based re-ranking uses structured controls for common constraints instead of persona-specific hard-coding. Seniority is categorical because job level is not continuous; users can choose options such as entry-level/new grad only, junior to mid-level, senior+ preferred, or exclusions for senior/junior roles. Salary uses a capped numeric slider from `$0` to `$250,000` with Soft, Medium, or Strict handling. Required years uses an optional bounded slider from `0` to `10` years, with unknown requirements allowed instead of automatically failing. These controls make the same ranking layer adaptable for different personas and users.

JobPilot includes a lightweight adaptive feedback layer inspired by contextual bandits. Users can mark displayed jobs as `Accept`, `Reject`, or `Skip`; these labels become reward signals of `+1`, `-1`, and `0`. Accepted jobs boost transparent signals such as role family, matched skills, company size, and title terms. Rejected jobs penalize similar signals, including rejected employment types. The app computes `feedback_adjustment_score` and `adjusted_final_score = final_score + feedback_adjustment_score`, then re-ranks the existing candidate set. This is a simple explainable re-ranking method, not a full DQN. A simulated feedback round is available to demonstrate measurable improvement through baseline average score, adjusted average score, accepted-style matches, and estimated feedback precision.

Build offline job embeddings:

```bash
python3 build_embeddings.py
```

This uses role-balanced, coverage-aware sampling from `data/jobs_cleaned.csv`, embeds up to 20,000 title/company/location/description records with `all-MiniLM-L6-v2`, and writes `data/job_embeddings.npy` plus `data/job_embedding_index.csv`.

Analyze cleaned-data and embedding-subset coverage:

```bash
python3 analyze_job_coverage.py
```

This reports coverage for salary, location, role groups, sponsorship signals, contract/temp signals, and seniority signals, then saves `data/job_coverage_summary.csv`.

When those embedding files exist, the app first retrieves candidate jobs using semantic similarity and then applies the existing rule-based ranking. If the embedding files are missing, the app falls back to full-dataset rule-based ranking.

Run the app:

```bash
streamlit run app.py
```

The app loads `data/jobs_cleaned.csv` when it exists. If the cleaned file has not been created yet, it falls back to `data/jobs_sample.csv`.

## Notes

This MVP does not include API ingestion, feedback learning, or resume generation yet. Those can be added later after the local matching workflow is stable.
