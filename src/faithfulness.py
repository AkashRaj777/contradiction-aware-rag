"""
Self-check: verify the final answer is supported by cited chunks.
"""
from __future__ import annotations

import json

import config
from src.grade import GradedChunk
from src.utils import get_openai_client


def check_faithfulness(answer: str, chunks: list[GradedChunk]) -> tuple[bool, str]:
    """
    Returns (is_faithful, reason).
    Uses LLM judge — simple and effective for resume demo scale.
    """
    if not chunks:
        return False, "No source chunks provided."

    context = "\n".join(f"[{g.chunk.doc_id}] {g.chunk.text[:400]}" for g in chunks[:4])
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=0,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": (
                    'Reply JSON only: {"faithful": true/false, "reason": "..."}. '
                    "Faithful means the answer is supported by the sources, not hallucinated."
                ),
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nAnswer:\n{answer}",
            },
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return bool(data.get("faithful", False)), str(data.get("reason", ""))
    except json.JSONDecodeError:
        ok = "true" in raw.lower() and "false" not in raw.lower()
        return ok, raw[:200]
