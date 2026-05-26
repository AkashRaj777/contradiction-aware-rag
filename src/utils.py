"""
Shared helpers: environment, eval loading, text chunking.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import config


def load_env() -> None:
    """Load API keys from .env in project root."""
    load_dotenv(config.ROOT_DIR / ".env")


def get_openai_client():
    """Return an OpenAI client after ensuring the API key exists."""
    from openai import OpenAI

    load_env()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=key)


def load_eval_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """Read questions.jsonl into a list of dicts."""
    path = path or config.EVAL_PATH
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping character chunks.
    Simple approach — easy to read and debug.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def doc_id_from_filename(filename: str) -> str:
    """refund_policy_2024.md -> refund_policy_2024"""
    return Path(filename).stem
