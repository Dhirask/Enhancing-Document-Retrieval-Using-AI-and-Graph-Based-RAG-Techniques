from typing import List

from .config import PipelineConfig
from .embeddings import TextEmbedder
from .generation import Generator
from .graph_store import GraphStore
from .ingestion import IngestionPipeline
from .rerank import Reranker
from .retrieval import Retriever


class GraphRAGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.ingestion = IngestionPipeline(config)
        self.graph_store = GraphStore(
            uri=config.graph.uri, user=config.graph.user, password=config.graph.password, database=config.graph.database
        )
        self.embedder = TextEmbedder(model_name=config.embedding.model_name, device=config.embedding.device)
        self.retriever = Retriever(config.retrieval, self.embedder, self.graph_store)
        self.reranker = Reranker(self.graph_store)
        self.generator = Generator(config.generation)

    def build_indexes(self, paths: List[str]) -> None:
        result = self.ingestion.ingest(paths)
        self.graph_store.upsert(result.chunks, result.entities, result.relations)
        self.retriever.index(result.chunks)

    def answer(self, query: str) -> str:
        entry_entities = []  # placeholder: extract from query
        retrieved = self.retriever.retrieve(query, entry_entities)
        reranked = self.reranker.rerank(retrieved)
        answer = self.generator.generate(reranked, query)
        return answer.answer
