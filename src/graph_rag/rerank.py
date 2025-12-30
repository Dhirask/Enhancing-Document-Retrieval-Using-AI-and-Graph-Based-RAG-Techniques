import logging
from dataclasses import dataclass
from typing import List

from .graph_store import GraphStore
from .retrieval import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    items: List[RetrievedChunk]


class Reranker:
    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store

    def rerank(self, result: RetrievalResult) -> RerankedResult:
        """Blend semantic score with graph centrality prior, with fallback to semantic if no graph data."""
        ids = [hit.chunk.id for hit in result.merged]
        
        if not ids:
            logger.warning("No merged results to rerank")
            return RerankedResult(items=[])
        
        centrality = {nid: score for nid, score in self.graph_store.centrality_score(ids)}
        
        if not centrality:
            logger.warning("No centrality scores from graph; returning semantic ranking")
            return RerankedResult(items=result.merged)
        
        rescored: List[RetrievedChunk] = []
        for hit in result.merged:
            c_score = centrality.get(hit.chunk.id, 0.0)
            final_score = 0.7 * hit.score + 0.3 * c_score
            rescored.append(RetrievedChunk(chunk=hit.chunk, score=final_score))
        
        rescored.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Reranked {len(rescored)} chunks using centrality")
        return RerankedResult(items=rescored)
