"""
Grade retrieved chunks for relevance using a cross-encoder reranker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from src.retrieve import RetrievedChunk


@dataclass
class GradedChunk:
    chunk: RetrievedChunk
    relevance: float  # 0–1 from cross-encoder sigmoid


class RelevanceGrader:
    """Scores (query, chunk) pairs — higher means more relevant."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(config.RERANKER_MODEL)
        return self._model

    def grade(self, query: str, chunks: list[RetrievedChunk]) -> list[GradedChunk]:
        if not chunks:
            return []

        pairs = [[query, c.text] for c in chunks]
        raw_scores = self._load_model().predict(pairs)

        graded = []
        for chunk, raw in zip(chunks, raw_scores):
            # Map raw logits to a simple 0–1 scale via sigmoid
            rel = float(1 / (1 + np.exp(-raw))) if hasattr(np, "exp") else float(raw)
            graded.append(GradedChunk(chunk=chunk, relevance=round(rel, 4)))

        graded.sort(key=lambda g: g.relevance, reverse=True)
        return graded
