from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "job_embeddings.npy"
INDEX_PATH = DATA_DIR / "job_embedding_index.csv"
ANN_INDEX_PATH = DATA_DIR / "job_annoy_index.ann"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 500
EMBEDDING_RELEVANT_SECTIONS = {
    "background",
    "skills",
    "technical skills",
    "experience",
    "projects",
    "education",
}
EMBEDDING_EXCLUDED_SECTIONS = {
    "target role",
    "target roles",
    "preferences",
    "preference",
    "dealbreakers",
    "dealbreaker",
    "salary",
    "location preference",
    "visa preference",
    "constraints",
}
NEGATIVE_CONSTRAINT_PATTERNS = [
    r"^no\s+senior\b",
    r"^no\s+staff\b",
    r"^no\s+manager\b",
    r"^no\s+director\b",
    r"^no\s+5\+?\s*years\b",
    r"^no\s+defense\b",
    r"^no\s+military\b",
    r"^no\s+clearance\b",
    r"^no\s+dod\b",
]


def embedding_files_exist() -> bool:
    return EMBEDDINGS_PATH.exists() and INDEX_PATH.exists()


@lru_cache(maxsize=1)
def load_job_embeddings(path: str = str(EMBEDDINGS_PATH)) -> np.ndarray:
    return np.load(path)


@lru_cache(maxsize=1)
def load_embedding_index(path: str = str(INDEX_PATH)) -> pd.DataFrame:
    index = pd.read_csv(path)
    index["row_index"] = index["row_index"].astype(int)
    return index


@lru_cache(maxsize=1)
def load_annoy_index(
    path: str = str(ANN_INDEX_PATH),
    embedding_dimension: int | None = None,
):
    from annoy import AnnoyIndex

    dimension = embedding_dimension or load_job_embeddings().shape[1]
    annoy_index = AnnoyIndex(dimension, "angular")
    if not annoy_index.load(path):
        raise OSError(f"Could not load Annoy index from {path}.")
    return annoy_index


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def normalize_section_heading(line: str) -> str:
    heading = line.strip().rstrip(":").lower()
    heading = re.sub(r"[^a-z0-9 /+-]+", "", heading)
    return re.sub(r"\s+", " ", heading).strip()


def is_section_heading(line: str) -> bool:
    heading = normalize_section_heading(line)
    known_headings = EMBEDDING_RELEVANT_SECTIONS | EMBEDDING_EXCLUDED_SECTIONS
    return heading in known_headings or (
        len(line.strip()) <= 40 and line.strip().endswith(":")
    )


def is_negative_constraint_line(line: str) -> bool:
    normalized = re.sub(r"^[^a-z0-9]+", "", line.strip().lower())
    return any(re.search(pattern, normalized) for pattern in NEGATIVE_CONSTRAINT_PATTERNS)


def split_inline_section(line: str) -> tuple[str, str]:
    if ":" not in line:
        return "", line
    possible_heading, content = line.split(":", 1)
    heading = normalize_section_heading(possible_heading)
    known_headings = EMBEDDING_RELEVANT_SECTIONS | EMBEDDING_EXCLUDED_SECTIONS
    if heading in known_headings:
        return heading, content.strip()
    return "", line


def extract_embedding_relevant_resume_text(resume_text: str) -> str:
    """Keep semantic resume signal while removing preferences and constraints."""
    if not resume_text:
        return ""

    cleaned_lines: list[str] = []
    active_section = ""
    found_relevant_section = False

    for raw_line in resume_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        inline_section, inline_content = split_inline_section(line)
        if inline_section:
            active_section = inline_section
            if active_section in EMBEDDING_RELEVANT_SECTIONS:
                found_relevant_section = True
                if inline_content and not is_negative_constraint_line(inline_content):
                    cleaned_lines.append(inline_content)
            continue

        heading = normalize_section_heading(line)
        if is_section_heading(line):
            active_section = heading
            if active_section in EMBEDDING_RELEVANT_SECTIONS:
                found_relevant_section = True
            continue

        if is_negative_constraint_line(line):
            continue
        if active_section in EMBEDDING_EXCLUDED_SECTIONS:
            continue
        if found_relevant_section and active_section not in EMBEDDING_RELEVANT_SECTIONS:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def build_user_profile_text(
    target_role: str,
    combined_skills_text: str,
    resume_text: str = "",
    max_resume_chars: int = 1_500,
) -> str:
    parts = [
        f"Target role: {target_role.strip()}",
        f"Skills: {combined_skills_text.strip()}",
    ]
    if resume_text:
        resume_excerpt = extract_embedding_relevant_resume_text(resume_text)
        if resume_excerpt:
            parts.append(f"Resume excerpt: {resume_excerpt[:max_resume_chars].strip()}")

    profile_text = "\n".join(part for part in parts if part.split(":", 1)[-1].strip())
    print("User profile text used for embedding:")
    print(profile_text)
    return profile_text


def generate_user_embedding(profile_text: str) -> np.ndarray:
    model = load_embedding_model()
    embedding = model.encode(
        [profile_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0]


def compute_cosine_similarity(
    user_embedding: np.ndarray,
    job_embeddings: np.ndarray,
) -> np.ndarray:
    user_norm = np.linalg.norm(user_embedding)
    job_norms = np.linalg.norm(job_embeddings, axis=1)
    denominator = np.maximum(job_norms * user_norm, 1e-12)
    return np.dot(job_embeddings, user_embedding) / denominator


def retrieve_top_candidates_bruteforce(
    profile_text: str,
    top_k: int = DEFAULT_TOP_K,
    user_embedding: np.ndarray | None = None,
) -> pd.DataFrame:
    job_embeddings = load_job_embeddings()
    embedding_index = load_embedding_index()

    if len(job_embeddings) != len(embedding_index):
        raise ValueError(
            "Embedding array length does not match job_embedding_index.csv length."
        )

    if user_embedding is None:
        user_embedding = generate_user_embedding(profile_text)
    similarities = compute_cosine_similarity(user_embedding, job_embeddings)

    candidate_count = min(top_k, len(similarities))
    if candidate_count == 0:
        return pd.DataFrame(columns=["row_index", "embedding_similarity"])
    top_positions = np.argpartition(-similarities, candidate_count - 1)[:candidate_count]
    top_positions = top_positions[np.argsort(-similarities[top_positions])]

    candidates = embedding_index.iloc[top_positions].copy()
    candidates["embedding_similarity"] = similarities[top_positions]
    result = candidates[["row_index", "embedding_similarity"]]
    result.attrs["retrieval_method"] = "bruteforce"
    return result


def retrieve_top_candidates_ann(
    profile_text: str,
    top_k: int = DEFAULT_TOP_K,
    user_embedding: np.ndarray | None = None,
) -> pd.DataFrame:
    job_embeddings = load_job_embeddings()
    embedding_index = load_embedding_index()
    if len(job_embeddings) != len(embedding_index):
        raise ValueError(
            "Embedding array length does not match job_embedding_index.csv length."
        )

    if user_embedding is None:
        user_embedding = generate_user_embedding(profile_text)
    candidate_count = min(top_k, len(job_embeddings))
    if candidate_count == 0:
        return pd.DataFrame(columns=["row_index", "embedding_similarity"])

    annoy_index = load_annoy_index(embedding_dimension=job_embeddings.shape[1])
    positions = annoy_index.get_nns_by_vector(
        user_embedding.tolist(),
        candidate_count,
    )
    candidate_embeddings = job_embeddings[positions]
    similarities = compute_cosine_similarity(user_embedding, candidate_embeddings)
    order = np.argsort(-similarities)
    sorted_positions = np.asarray(positions)[order]

    candidates = embedding_index.iloc[sorted_positions].copy()
    candidates["embedding_similarity"] = similarities[order]
    result = candidates[["row_index", "embedding_similarity"]]
    result.attrs["retrieval_method"] = "ann"
    return result


def retrieve_top_candidates(profile_text: str, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
    user_embedding = generate_user_embedding(profile_text)
    if ANN_INDEX_PATH.exists():
        try:
            return retrieve_top_candidates_ann(
                profile_text,
                top_k=top_k,
                user_embedding=user_embedding,
            )
        except Exception as error:
            print(f"Annoy retrieval unavailable; using brute-force fallback: {error}")

    return retrieve_top_candidates_bruteforce(
        profile_text,
        top_k=top_k,
        user_embedding=user_embedding,
    )
