"""
Full Contradiction-Aware RAG pipeline with corrective retrieval loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from src.db import log_run
from src.faithfulness import check_faithfulness
from src.generate import detect_contradiction, generate_answer, rewrite_query
from src.grade import GradedChunk, RelevanceGrader
from src.ingest import CorpusIndex
from src.retrieve import HybridRetriever


@dataclass
class PipelineResult:
    """Everything the UI or eval script needs from one question."""
    question: str
    answer: str
    decision: str          # answer | conflict | refuse
    confidence: str        # high | medium | low
    citations: list[str] = field(default_factory=list)
    rewritten_query: str | None = None
    retries: int = 0
    contradiction_explanation: str = ""
    metadata: dict = field(default_factory=dict)


class ContradictionRAGPipeline:
    """
    Steps:
      1. Retrieve (hybrid)
      2. Grade relevance
      3. If weak -> rewrite query and retry (corrective loop)
      4. Detect contradictions
      5. Decide: answer / conflict / refuse
      6. Generate + faithfulness check
    """

    def __init__(self):
        self.index = CorpusIndex()
        self.retriever = HybridRetriever(self.index)
        self.grader = RelevanceGrader()

    def _strong_chunks(self, graded: list[GradedChunk]) -> list[GradedChunk]:
        return [g for g in graded if g.relevance >= config.MIN_RELEVANCE_SCORE]

    def run(self, question: str, log: bool = True) -> PipelineResult:
        query = question
        retries = 0
        rewritten = None
        graded: list[GradedChunk] = []

        # --- Retrieve + grade, with optional corrective retry ---
        for attempt in range(config.MAX_RETRIES + 1):
            chunks = self.retriever.retrieve(query)
            graded = self.grader.grade(query, chunks)
            strong = self._strong_chunks(graded)

            if len(strong) >= config.MIN_STRONG_CHUNKS:
                graded = strong
                break

            if attempt < config.MAX_RETRIES:
                weak_summary = "\n".join(g.chunk.text[:120] for g in graded[:3]) or "No context."
                rewritten = rewrite_query(question, weak_summary)
                query = rewritten
                retries += 1
            else:
                graded = strong  # may be empty

        strong = self._strong_chunks(graded)

        # --- Refuse if evidence is still too weak ---
        if len(strong) < config.MIN_STRONG_CHUNKS:
            answer = generate_answer(question, [], mode="refuse")
            result = PipelineResult(
                question=question,
                answer=answer,
                decision="refuse",
                confidence="low",
                citations=[],
                rewritten_query=rewritten,
                retries=retries,
                metadata={"strong_chunks": 0},
            )
            if log:
                log_run(question, "full", "refuse", answer, "low", result.metadata)
            return result

        # --- Contradiction detection ---
        has_conflict, conflict_note, conf_score = detect_contradiction(question, strong)

        if has_conflict and conf_score >= config.CONTRADICTION_THRESHOLD:
            decision = "conflict"
            answer = generate_answer(question, strong, mode="conflict", conflict_note=conflict_note)
            confidence = "medium"
        else:
            decision = "answer"
            answer = generate_answer(question, strong, mode="answer")
            confidence = "high"

        # --- Faithfulness gate ---
        faithful, reason = check_faithfulness(answer, strong)
        if not faithful and decision == "answer":
            # One downgrade: refuse rather than ship a hallucination
            answer = generate_answer(question, [], mode="refuse")
            decision = "refuse"
            confidence = "low"
            metadata_extra = {"faithfulness_fail": reason}
        else:
            metadata_extra = {"faithfulness": reason}

        citations = list({g.chunk.doc_id for g in strong})
        result = PipelineResult(
            question=question,
            answer=answer,
            decision=decision,
            confidence=confidence,
            citations=citations,
            rewritten_query=rewritten,
            retries=retries,
            contradiction_explanation=conflict_note,
            metadata={
                "strong_chunks": len(strong),
                "conflict_score": conf_score,
                **metadata_extra,
            },
        )
        if log:
            log_run(question, "full", decision, answer, confidence, result.metadata)
        return result
