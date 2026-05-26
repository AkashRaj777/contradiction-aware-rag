"""
Load markdown corpus, chunk documents, embed with sentence-transformers,
and store in ChromaDB.
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

import config
from src.utils import chunk_text, doc_id_from_filename


class CorpusIndex:
    """Wraps embedding model + Chroma collection for retrieval."""

    def __init__(self):
        self._embedder = None
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="policy_corpus",
            metadata={"hnsw:space": "cosine"},
        )

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._embedder

    def reset(self) -> None:
        """Delete and recreate the collection (use before full re-ingest)."""
        try:
            self.client.delete_collection("policy_corpus")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="policy_corpus",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_corpus(self, corpus_dir: Path | None = None) -> int:
        """
        Read all .md files, chunk, embed, upsert into Chroma.
        Returns number of chunks indexed.
        """
        corpus_dir = corpus_dir or config.CORPUS_DIR
        self.reset()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []

        chunk_count = 0
        for md_path in sorted(corpus_dir.glob("*.md")):
            doc_id = doc_id_from_filename(md_path.name)
            raw = md_path.read_text(encoding="utf-8")
            chunks = chunk_text(raw, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}__chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({"doc_id": doc_id, "source_file": md_path.name, "chunk_index": i})
                chunk_count += 1

        # Batch embed for speed
        vectors = self._load_embedder().encode(documents, show_progress_bar=True).tolist()
        embeddings = vectors

        # Chroma has a batch size limit — insert in batches of 100
        batch = 100
        for i in range(0, len(ids), batch):
            self.collection.add(
                ids=ids[i : i + batch],
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
                embeddings=embeddings[i : i + batch],
            )

        return chunk_count

    def embed_query(self, query: str) -> list[float]:
        return self._load_embedder().encode([query])[0].tolist()


def run_ingest() -> None:
    """CLI entry: build the vector index from data/corpus/*.md"""
    index = CorpusIndex()
    n = index.ingest_corpus()
    print(f"Indexed {n} chunks from {config.CORPUS_DIR}")


if __name__ == "__main__":
    run_ingest()
