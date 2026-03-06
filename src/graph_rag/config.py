from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    model_name: str = "all-mpnet-base-v2"
    device: str = "cpu"
    dim: int = 768


@dataclass
class GraphConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "Krsna0123#"
    database: str = "neo4j"
    max_hops: int = 2


@dataclass
class RetrievalConfig:
    top_k_vectors: int = 10
    top_k_graph: int = 10
    alpha_semantic: float = 0.6  # blend between semantic and graph scores


@dataclass
class RerankConfig:
    centrality_weight: float = 0.3  # weight for centrality in reranking
    semantic_weight: float = 0.7    # weight for semantic/merged score


@dataclass
class GenerationConfig:
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 512
    temperature: float = 0.2


@dataclass
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    chunk_size: int = 300          # words per chunk (smaller = more granular graph)
    chunk_overlap: int = 50
    allowed_formats: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".md"])
    cache_dir: Optional[str] = None
