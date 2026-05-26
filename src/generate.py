"""
OpenAI calls: answer generation, query rewrite, and LLM judges.
"""
from __future__ import annotations

import json

import config
from src.grade import GradedChunk
from src.utils import get_openai_client


def _chat(system: str, user: str) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=config.OPENAI_TEMPERATURE,
        max_tokens=config.MAX_GENERATION_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def rewrite_query(original: str, weak_context_summary: str) -> str:
    """
    Corrective step: suggest a better search query when retrieval was weak.
    """
    system = (
        "You improve search queries for an internal company policy search engine. "
        "Return ONLY the rewritten query, no explanation."
    )
    user = (
        f"Original question: {original}\n\n"
        f"Retrieved context was weak or off-topic:\n{weak_context_summary}\n\n"
        "Rewrite the query to find specific policy documents."
    )
    return _chat(system, user)


def detect_contradiction(query: str, chunks: list[GradedChunk]) -> tuple[bool, str, float]:
    """
    Ask the LLM if top chunks disagree on facts relevant to the question.
    Returns: (has_conflict, explanation, confidence 0-1)
    """
    if len(chunks) < 2:
        return False, "Not enough sources to compare.", 0.0

    context = ""
    for i, g in enumerate(chunks[:4], 1):
        context += f"\n--- Source {i} ({g.chunk.doc_id}) ---\n{g.chunk.text}\n"

    system = (
        "You detect factual contradictions between policy documents. "
        "Reply in JSON only: "
        '{"has_conflict": true/false, "explanation": "...", "confidence": 0.0-1.0}'
    )
    user = f"Question: {query}\n\nSources:{context}\n\nDo sources conflict on facts that matter to this question?"
    raw = _chat(system, user)

    try:
        # Strip markdown fences if model adds them
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return (
            bool(data.get("has_conflict", False)),
            str(data.get("explanation", "")),
            float(data.get("confidence", 0.5)),
        )
    except json.JSONDecodeError:
        # Fallback: keyword heuristic
        has = "conflict" in raw.lower() or "disagree" in raw.lower()
        return has, raw[:300], 0.5 if has else 0.2


def generate_answer(
    query: str,
    chunks: list[GradedChunk],
    mode: str,
    conflict_note: str = "",
) -> str:
    """
    mode: 'answer' | 'conflict' | 'refuse'
    """
    context = ""
    for g in chunks[:5]:
        context += f"\n[{g.chunk.doc_id}] (relevance={g.relevance})\n{g.chunk.text}\n"

    if mode == "refuse":
        return (
            "I don't have enough reliable information in the policy corpus to answer this question. "
            "Please consult HR, Finance, or your manager for topics not covered by internal policies."
        )

    if mode == "conflict":
        system = (
            "You are a policy assistant. Sources CONFLICT. Present BOTH sides clearly with document IDs. "
            "Do not merge into a single answer. Cite [doc_id] for each claim."
        )
        user = f"Question: {query}\n\nConflict note: {conflict_note}\n\nSources:{context}"
        return _chat(system, user)

    system = (
        "Answer ONLY using the provided sources. Cite document IDs like [password_policy_2025]. "
        "If sources are insufficient, say you cannot answer."
    )
    user = f"Question: {query}\n\nSources:{context}"
    return _chat(system, user)


def rewrite_query_simple(original: str) -> str:
    """Lightweight rewrite without extra context (used in pipeline)."""
    return _chat(
        "Return only a rewritten search query for company policies.",
        f"Original: {original}",
    )
