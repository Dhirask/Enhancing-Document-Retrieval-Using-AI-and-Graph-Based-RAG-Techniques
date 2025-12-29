from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import RetrievalConfig
from .embeddings import TextEmbedder
from .graph_store import GraphStore
from .ingestion import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    semantic: List[RetrievedChunk]
    graph: List[RetrievedChunk]
    merged: List[RetrievedChunk]


class Retriever:
    def __init__(self, config: RetrievalConfig, embedder: TextEmbedder, graph_store: GraphStore) -> None:
        self.config = config
        self.embedder = embedder
        self.graph_store = graph_store
        self._id_to_chunk: Dict[str, Chunk] = {}
        self._chunk_ids: List[str] = []
        self._faiss_index = None
        self._dim = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return

        # Skip chunks already indexed to avoid duplicate IDs and stale mappings.
        new_chunks = [c for c in chunks if c.id not in self._id_to_chunk]
        if not new_chunks:
            return

        vectors = np.asarray(self.embedder.encode([c.text for c in new_chunks]), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embedder returned invalid shape for vectors.")
        dim = vectors.shape[1]
        if self._dim is None:
            self._dim = dim
        elif dim != self._dim:
            raise ValueError("Embedding dimension mismatch across indexed chunks.")

        try:
            import faiss  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("faiss is required for retrieval. Install with `pip install faiss-cpu`." ) from exc

        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(self._dim)
        self._faiss_index.add(vectors)

        for chunk in new_chunks:
            self._id_to_chunk[chunk.id] = chunk
            self._chunk_ids.append(chunk.id)

    def retrieve(self, query: str, entry_entities: List[str]) -> RetrievalResult:
        semantic_hits = self._semantic_search(query)
        graph_hits = self._graph_expand(entry_entities)
        merged = self._merge(semantic_hits, graph_hits)
        return RetrievalResult(semantic=semantic_hits, graph=graph_hits, merged=merged)

    def _semantic_search(self, query: str) -> List[RetrievedChunk]:
        if not self._faiss_index or not self._chunk_ids:
            return []
        q_vec = np.asarray(self.embedder.encode([query]), dtype=np.float32)
        if q_vec.ndim != 2:
            return []
        scores, idxs = self._faiss_index.search(q_vec, min(self.config.top_k_vectors, len(self._chunk_ids)))  # type: ignore[arg-type]
        hits: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._chunk_ids):
                continue
            chunk_id = self._chunk_ids[idx]
            chunk = self._id_to_chunk.get(chunk_id)
            if not chunk:
                continue
            hits.append(RetrievedChunk(chunk=chunk, score=float(score)))
        return hits

    def _graph_expand(self, entry_entities: List[str]) -> List[RetrievedChunk]:
        if not entry_entities:
            return []
        node_ids = self.graph_store.neighbors(entry_entities, max_hops=2, limit=self.config.top_k_graph)
        hits: List[RetrievedChunk] = []
        for node_id in node_ids:
            chunk = self._id_to_chunk.get(node_id)
            if not chunk:
                continue
            hits.append(RetrievedChunk(chunk=chunk, score=1.0))
        return hits

    def _merge(self, semantic: List[RetrievedChunk], graph: List[RetrievedChunk]) -> List[RetrievedChunk]:
        merged = {hit.chunk.id: hit for hit in semantic}
        for hit in graph:
            if hit.chunk.id in merged:
                merged[hit.chunk.id].score = (
                    self.config.alpha_semantic * merged[hit.chunk.id].score
                    + (1 - self.config.alpha_semantic) * hit.score
                )
            else:
                merged[hit.chunk.id] = hit
        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x.score, reverse=True)
        return merged_list
