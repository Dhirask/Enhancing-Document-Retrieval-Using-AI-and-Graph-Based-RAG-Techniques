import logging
from typing import List

from .config import PipelineConfig
from .embeddings import TextEmbedder
from .generation import Generator
from .graph_store import GraphStore
from .ingestion import IngestionPipeline
from .rerank import Reranker
from .retrieval import Retriever
from .subgraph_selector import select_reasoning_subgraph

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # --- Ingestion ---
        self.ingestion = IngestionPipeline(config)

        # --- GraphStore (credentials from config) ---
        self.graph_store = GraphStore(
            uri=config.graph.uri,
            user=config.graph.user,
            password=config.graph.password,
            database=config.graph.database,
        )

        # --- Embeddings & Retrieval ---
        self.embedder = TextEmbedder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
        )
        self.retriever = Retriever(
            config.retrieval, self.embedder, self.graph_store,
            graph_config=config.graph,
        )

        # --- Reranking & Generation ---
        self.reranker = Reranker(self.graph_store, config=config.rerank)
        self.generator = Generator(config.generation)

    def build_indexes(self, paths: List[str]) -> None:
        """Load documents, extract entities/relations, upsert to graph and index for retrieval."""
        logger.info(f"Building indexes from {len(paths)} paths")

        result = self.ingestion.ingest(paths)
        logger.info(
            f"Ingestion complete: {len(result.chunks)} chunks, "
            f"{len(result.entities)} entities, {len(result.relations)} relations"
        )

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

    def answer(self, query: str):
        """Full pipeline: extract entities -> retrieve -> rerank -> generate with graph triples.

        Returns:
            Tuple of (answer_text, graph_triples) so callers can display
            reasoning details if desired.
        """
        entry_entities = self._extract_query_entities(query)
        logger.info(f"Query entities: {entry_entities}")

        retrieved = self.retriever.retrieve(query, entry_entities)
        reranked = self.reranker.rerank(retrieved)

        # Fetch knowledge graph triples for the retrieved chunks
        chunk_ids = [hit.chunk.id for hit in reranked.items[:10]]
        raw_triples = self.graph_store.get_subgraph_triples(chunk_ids, limit=30)
        logger.info(f"Fetched {len(raw_triples)} raw graph triples")

        # Select a small, connected reasoning subgraph
        graph_triples = select_reasoning_subgraph(query, raw_triples, max_triples=10)
        logger.info(f"Selected {len(graph_triples)} reasoning subgraph triples")

        generation = self.generator.generate(reranked, query, graph_triples=graph_triples)
        return generation.answer, graph_triples

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entity surface forms from the query using spaCy NER.

        Falls back to noun chunks when NER finds no named entities,
        ensuring the graph path is used even for queries without proper nouns.
        """
        doc = self.ingestion._nlp(query)
        # Try NER first
        ner_entities = list({ent.text for ent in doc.ents})
        if ner_entities:
            return ner_entities

        # Fallback: extract meaningful noun chunks (skip pronouns, determiners, etc.)
        STOP_CHUNKS = {"it", "this", "that", "what", "which", "who", "how", "the", "a", "an"}
        noun_chunks = []
        for nc in doc.noun_chunks:
            text = nc.text.strip()
            if text.lower() not in STOP_CHUNKS and len(text) > 1:
                noun_chunks.append(text)
        if noun_chunks:
            logger.info(f"NER empty, falling back to noun chunks: {noun_chunks}")
        return noun_chunks
