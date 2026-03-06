import logging
from dataclasses import dataclass
from typing import List

from .config import RerankConfig
from .graph_store import GraphStore
from .retrieval import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    items: List[RetrievedChunk]


class Reranker:
    def __init__(self, graph_store: GraphStore, config: RerankConfig = None) -> None:
        self.graph_store = graph_store
        self.config = config or RerankConfig()

    def rerank(self, result: RetrievalResult) -> RerankedResult:
        """Blend merged score with normalized graph centrality."""
        ids = [hit.chunk.id for hit in result.merged]

        if not ids:
            logger.warning("No merged results to rerank")
            return RerankedResult(items=[])

        # Get normalized centrality scores (already in [0, 1] from graph_store)
        centrality = {nid: score for nid, score in self.graph_store.centrality_score(ids)}

        if not centrality:
            logger.warning("No centrality scores from graph; returning merged ranking unchanged")
            return RerankedResult(items=result.merged)

        w_sem = self.config.semantic_weight
        w_cent = self.config.centrality_weight

        rescored: List[RetrievedChunk] = []
        for hit in result.merged:
            c_score = centrality.get(hit.chunk.id, 0.0)
            final_score = w_sem * hit.score + w_cent * c_score
            rescored.append(RetrievedChunk(chunk=hit.chunk, score=final_score))

        rescored.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Reranked {len(rescored)} chunks (centrality weight={w_cent})")
        return RerankedResult(items=rescored)
