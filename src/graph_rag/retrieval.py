from dataclasses import dataclass
from typing import List, Sequence

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
        self._index: List[Chunk] = []  # placeholder for FAISS index storage

    def index(self, chunks: Sequence[Chunk]) -> None:
        # Real implementation would add to FAISS; here we keep in-memory list.
        self._index.extend(chunks)

    def retrieve(self, query: str, entry_entities: List[str]) -> RetrievalResult:
        semantic_hits = self._semantic_search(query)
        graph_hits = self._graph_expand(entry_entities)
        merged = self._merge(semantic_hits, graph_hits)
        return RetrievalResult(semantic=semantic_hits, graph=graph_hits, merged=merged)

    def _semantic_search(self, query: str) -> List[RetrievedChunk]:
        # Placeholder: cosine on embeddings; here we use text length similarity.
        q_len = len(query.split())
        scored = []
        for chunk in self._index:
            score = 1.0 / (1 + abs(len(chunk.text.split()) - q_len))
            scored.append(RetrievedChunk(chunk=chunk, score=score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: self.config.top_k_vectors]

    def _graph_expand(self, entry_entities: List[str]) -> List[RetrievedChunk]:
        node_ids = self.graph_store.neighbors(entry_entities, max_hops=self.config.top_k_graph)
        scored = []
        for node_id in node_ids:
            # In a real system, map node ids back to chunks; here we skip that mapping.
            pass
        return []

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
