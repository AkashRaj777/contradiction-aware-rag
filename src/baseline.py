"""
Naive RAG baseline: retrieve once, no grading, no contradiction check, no refusal logic.
Used for comparison in eval.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.generate import generate_answer
from src.grade import GradedChunk
from src.ingest import CorpusIndex
from src.retrieve import HybridRetriever


@dataclass
class BaselineResult:
    question: str
    answer: str
    decision: str  # always 'answer' for baseline
    citations: list[str]


class NaiveRAG:
    """Simple retrieve-top-k -> generate pipeline."""

    def __init__(self):
        self.index = CorpusIndex()
        self.retriever = HybridRetriever(self.index)

    def run(self, question: str) -> BaselineResult:
        chunks = self.retriever.retrieve(question)
        # Fake uniform relevance so generate_answer accepts them
        graded = [GradedChunk(chunk=c, relevance=0.9) for c in chunks]
        answer = generate_answer(question, graded, mode="answer")
        citations = list({c.doc_id for c in chunks})
        return BaselineResult(
            question=question,
            answer=answer,
            decision="answer",
            citations=citations,
        )
