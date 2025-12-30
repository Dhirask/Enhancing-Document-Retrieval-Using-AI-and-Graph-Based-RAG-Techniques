import os
import logging
from typing import List

from .config import PipelineConfig
from .embeddings import TextEmbedder
from .generation import Generator
from .graph_store import GraphStore
from .ingestion import IngestionPipeline
from .rerank import Reranker
from .retrieval import Retriever

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # --- Ingestion ---
        self.ingestion = IngestionPipeline(config)

        # --- GraphStore (env-based, passed explicitly) ---
        neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "ihatedhiras")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not neo4j_password:
            raise RuntimeError("NEO4J_PASSWORD environment variable is not set.")

        self.graph_store = GraphStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )

        # --- Embeddings & Retrieval ---
        self.embedder = TextEmbedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
        )
        self.retriever = Retriever(config.retrieval, self.embedder, self.graph_store)

        # --- Reranking & Generation ---
        self.reranker = Reranker(self.graph_store)
        self.generator = Generator(config.generation)

    def build_indexes(self, paths: List[str]) -> None:
        """Load documents, extract entities/relations, upsert to graph and index for retrieval."""
        logger.info(f"Building indexes from {len(paths)} paths")
        
        result = self.ingestion.ingest(paths)
        logger.info(f"Ingestion complete: {len(result.chunks)} chunks, {len(result.entities)} entities, {len(result.relations)} relations")
        
        if not result.chunks:
            raise ValueError("Ingestion produced no chunks; check input documents")
        
        self.graph_store.upsert(
            result.chunks,
            result.entities,
            result.relations,
        )
        logger.info("Upserted data to Neo4j")
        
        self.retriever.index(result.chunks)
        logger.info(f"Indexed {len(result.chunks)} chunks in FAISS")

    def answer(self, query: str) -> str:
        entry_entities = self._extract_query_entities(query)
        retrieved = self.retriever.retrieve(query, entry_entities)
        reranked = self.reranker.rerank(retrieved)
        generation = self.generator.generate(reranked, query)
        return generation.answer

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entity surface forms from the query using spaCy."""
        doc = self.ingestion._nlp(query)
        return list({ent.text for ent in doc.ents})
