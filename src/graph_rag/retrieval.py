import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import RetrievalConfig, GraphConfig
from .embeddings import TextEmbedder
from .graph_store import GraphStore
from .ingestion import Chunk

logger = logging.getLogger(__name__)


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
    def __init__(self, config: RetrievalConfig, embedder: TextEmbedder,
                 graph_store: GraphStore, graph_config: GraphConfig = None) -> None:
        self.config = config
        self.embedder = embedder
        self.graph_store = graph_store
        self.graph_config = graph_config
        self._id_to_chunk: Dict[str, Chunk] = {}
        self._chunk_ids: List[str] = []
        self._faiss_index = None
        self._dim = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return

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
        except ImportError as exc:  # pragma: no cover
            raise ImportError("faiss is required for retrieval. Install with `pip install faiss-cpu`.") from exc

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

        logger.info(
            f"Retrieval: {len(semantic_hits)} semantic, {len(graph_hits)} graph, "
            f"{len(merged)} merged"
        )
        return RetrievalResult(semantic=semantic_hits, graph=graph_hits, merged=merged)

    def _semantic_search(self, query: str) -> List[RetrievedChunk]:
        if not self._faiss_index or not self._chunk_ids:
            return []
        q_vec = np.asarray(self.embedder.encode([query]), dtype=np.float32)
        if q_vec.ndim != 2:
            return []
        scores, idxs = self._faiss_index.search(q_vec, min(self.config.top_k_vectors, len(self._chunk_ids)))
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
        """Expand from entity surface forms through the knowledge graph.

        1. Entity linking: resolve surface forms → entry chunk IDs
        2. Graph traversal: multi-hop from entry chunks → neighbor chunks
        3. Score by inverse hop distance
        """
        if not entry_entities:
            logger.debug("No entry entities for graph expansion")
            return []

        # Step 1: Entity linking — surface forms to chunk IDs
        entry_chunk_ids = self.graph_store.find_entry_chunks(
            entry_entities, limit=self.config.top_k_graph
        )
        if not entry_chunk_ids:
            logger.debug(f"Entity linking found no chunks for: {entry_entities}")
            return []
        logger.debug(f"Entity linking: {entry_entities} -> {entry_chunk_ids}")

        # Step 2: Graph traversal (type filtering handled by graph_store internally)
        max_hops = self.graph_config.max_hops if self.graph_config else 2

        neighbor_results = self.graph_store.neighbors(
            entry_chunk_ids,
            max_hops=max_hops,
            limit=self.config.top_k_graph,
        )

        # Step 3: Build hits with hop-distance decay scoring
        hits: List[RetrievedChunk] = []
        seen = set(entry_chunk_ids)

        # Include entry chunks themselves (distance = 0, score = 1.0)
        for chunk_id in entry_chunk_ids:
            chunk = self._id_to_chunk.get(chunk_id)
            if chunk:
                hits.append(RetrievedChunk(chunk=chunk, score=1.0))

        # Include graph-traversed neighbors with decay
        for chunk_id, hops in neighbor_results:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = self._id_to_chunk.get(chunk_id)
            if not chunk:
                continue
            # Inverse hop-distance decay: score = 1.0 / (1 + hops)
            score = 1.0 / (1.0 + hops)
            hits.append(RetrievedChunk(chunk=chunk, score=score))

        logger.info(f"Graph expansion: {len(hits)} chunks (from {len(entry_entities)} entities)")
        return hits

    def _merge(self, semantic: List[RetrievedChunk], graph: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Merge semantic and graph results with alpha blending."""
        alpha = self.config.alpha_semantic

        # Start with semantic results
        merged: Dict[str, RetrievedChunk] = {}
        for hit in semantic:
            merged[hit.chunk.id] = RetrievedChunk(chunk=hit.chunk, score=hit.score)

        for hit in graph:
            if hit.chunk.id in merged:
                # Blend: alpha * semantic + (1 - alpha) * graph
                sem_score = merged[hit.chunk.id].score
                merged[hit.chunk.id] = RetrievedChunk(
                    chunk=hit.chunk,
                    score=alpha * sem_score + (1 - alpha) * hit.score,
                )
            else:
                # Graph-only result: use (1 - alpha) * graph_score
                merged[hit.chunk.id] = RetrievedChunk(
                    chunk=hit.chunk,
                    score=(1 - alpha) * hit.score,
                )

        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x.score, reverse=True)
        return merged_list
