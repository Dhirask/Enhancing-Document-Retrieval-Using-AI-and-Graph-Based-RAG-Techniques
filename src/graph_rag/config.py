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
    password: str = "password"
    database: str = "neo4j"
    max_hops: int = 2


@dataclass
class RetrievalConfig:
    top_k_vectors: int = 10
    top_k_graph: int = 10
    alpha_semantic: float = 0.6  # blend between semantic and graph scores


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
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    chunk_size: int = 500
    chunk_overlap: int = 50
    allowed_formats: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".md"])
    cache_dir: Optional[str] = None
