# Graph-RAG Architecture (Research Prototype)

## Overview
The system builds a hybrid retrieval pipeline that marries semantic vectors with a knowledge graph to support multi-hop reasoning and grounded answer generation. Components are modular to allow swapping models, databases, and rerankers.

## Data Flow
1. **Ingestion**: Load raw documents (PDF/TXT/MD), clean, and semantically chunk. Run NER + relation extraction to produce entities and edges.
2. **Graph Construction**: Create nodes for documents, chunks, and entities; add edges for `mentions`, `related_to`, `cites`, and `part_of`. Persist to Neo4j (or any graph DB with Cypher-like API).
3. **Embedding & Indexing**: Compute text embeddings for chunks (LLM encoder) and graph embeddings (GraphSAGE/Node2Vec). Store chunk vectors in FAISS and, optionally, node vectors for graph-aware retrieval.
4. **Query Processing**: Embed the user query, extract entities, and pick entry nodes in the graph (entity matches or high-similarity chunks).
5. **Hybrid Retrieval**: Combine vector similarity (FAISS) with graph traversal (multi-hop neighbors from entry nodes). Merge, deduplicate, and score candidates.
6. **Graph-Based Reranking**: Score with centrality, path relevance, and (optionally) a GNN inference pass. Blend semantic and graph scores to rerank.
7. **Context Construction**: Build a compact, non-redundant context window that preserves entity coherence and citation paths.
8. **LLM Generation**: Prompt an LLM with grounded context; optionally emit citations to source chunks/entities.

## Key Design Choices
- **Separation of concerns**: ingestion → graph store → retrieval → rerank → generation.
- **Graph-first retrieval**: queries seed graph traversal for multi-hop reasoning; vectors handle semantic recall.
- **Pluggable models**: embedder, NER/RE, and GNN are interfaces to ease swaps.
- **Safety**: explicit grounding to retrieved context to cut hallucinations; expose citations.

## Component Contracts (summary)
- `IngestionPipeline.ingest(paths: List[str]) -> IngestionResult`
- `GraphStore` handles node/edge upsert and traversals.
- `Retriever.retrieve(query: str) -> RetrievalResult`
- `Reranker.rerank(result: RetrievalResult) -> RerankedResult`
- `Generator.generate(reranked: RerankedResult) -> GenerationResult`

## Extensibility Points
- Swap FAISS with another ANN index.
- Swap Neo4j with Memgraph or a local NetworkX prototype.
- Replace GraphSAGE with Node2Vec/GIN; plug different LLMs for generation.
- Add evaluation hooks (precision@k, grounding coverage) and caching layers.
