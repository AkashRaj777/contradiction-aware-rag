"""
Central configuration for the Contradiction-Aware RAG project.
Adjust paths and model names here instead of scattering magic strings.
"""
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_PATH = DATA_DIR / "eval" / "questions.jsonl"
CHROMA_DIR = ROOT_DIR / ".chroma"
RUNS_DB_PATH = ROOT_DIR / "runs.db"

# --- Chunking ---
CHUNK_SIZE = 600       # characters per chunk (simple splitter)
CHUNK_OVERLAP = 80

# --- Retrieval ---
TOP_K = 6              # chunks retrieved per query
BM25_WEIGHT = 0.4      # blend: final = w*bm25_norm + (1-w)*vector_sim
MIN_RELEVANCE_SCORE = 0.35   # below this → weak evidence
MIN_STRONG_CHUNKS = 2        # need at least this many relevant chunks to answer

# --- Corrective loop ---
MAX_RETRIES = 1        # one query rewrite + re-retrieve

# --- OpenAI ---
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.1
MAX_GENERATION_TOKENS = 700

# --- Local models (free, run on CPU) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Decision thresholds ---
CONTRADICTION_THRESHOLD = 0.55   # LLM judge: score above = conflict
