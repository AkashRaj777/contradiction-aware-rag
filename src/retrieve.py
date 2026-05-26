"""
Hybrid retrieval: combine BM25 (keyword) and vector similarity (semantic).
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

import config
from src.ingest import CorpusIndex


@dataclass
class RetrievedChunk:
    """One piece of evidence returned from the index."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source_file: str


class HybridRetriever:
    """
    Keeps an in-memory BM25 index over all chunks and uses Chroma for vectors.
    """

    def __init__(self, index: CorpusIndex):
        self.index = index
        self._load_bm25_cache()

    def _load_bm25_cache(self) -> None:
        """Pull all documents from Chroma once to build BM25."""
        data = self.index.collection.get(include=["documents", "metadatas"])
        self.chunk_ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        tokenized = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or config.TOP_K

        # --- Vector scores from Chroma ---
        query_emb = self.index.embed_query(query)
        vec_result = self.index.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k * 2, len(self.chunk_ids)),
            include=["documents", "metadatas", "distances"],
        )
        vec_ids = vec_result["ids"][0]
        vec_dist = vec_result["distances"][0]
        # Chroma cosine distance: lower is better -> convert to similarity
        vec_sim = {cid: 1.0 - d for cid, d in zip(vec_ids, vec_dist)}

        # --- BM25 scores ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_norm = {cid: s / max_bm25 for cid, s in zip(self.chunk_ids, bm25_scores)}

        # --- Blend scores for every chunk that appeared in vector top set ---
        blended: list[tuple[str, float]] = []
        for cid in vec_ids:
            v = vec_sim.get(cid, 0.0)
            b = bm25_norm.get(cid, 0.0)
            final = config.BM25_WEIGHT * b + (1 - config.BM25_WEIGHT) * v
            blended.append((cid, final))

        blended.sort(key=lambda x: x[1], reverse=True)
        top = blended[:top_k]

        results: list[RetrievedChunk] = []
        id_to_doc = {cid: (doc, meta) for cid, doc, meta in zip(self.chunk_ids, self.documents, self.metadatas)}

        for cid, score in top:
            doc, meta = id_to_doc[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    text=doc,
                    score=round(score, 4),
                    source_file=meta.get("source_file", ""),
                )
            )
        return results
