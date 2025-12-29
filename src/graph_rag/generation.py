from dataclasses import dataclass
from typing import List

from .config import GenerationConfig
from .rerank import RerankedResult


@dataclass
class GenerationResult:
    answer: str
    citations: List[str]


class Generator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(self, reranked: RerankedResult, query: str) -> GenerationResult:
        # Placeholder generation: stitch top chunks.
        context = " \n".join([hit.chunk.text for hit in reranked.items[:3]])
        answer = f"Answer to '{query}' grounded in: {context[:200]}..."
        citations = [hit.chunk.id for hit in reranked.items[:3]]
        return GenerationResult(answer=answer, citations=citations)
