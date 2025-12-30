"""
Validation and debugging script for GraphRAG pipeline.

Usage:
    python -m examples.validate_pipeline --config configs/example_config.yaml --data data/
"""

import argparse
import logging
from pathlib import Path
from sys import exit as sys_exit

from src.graph_rag.config import PipelineConfig
from src.graph_rag.pipeline import GraphRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def validate_pipeline(config_path: str, data_dir: str) -> bool:
    """Full validation: ingest → graph → retrieval → reranking → generation."""
    
    logger.info("=" * 80)
    logger.info("GraphRAG Pipeline Validation")
    logger.info("=" * 80)
    
    # Load config
    cfg = PipelineConfig()
    logger.info(f"Config loaded: embedding={cfg.embedding.model_name}, graph.uri={cfg.graph.uri}")
    
    # Init pipeline
    pipeline = GraphRAGPipeline(cfg)
    
    # Verify Neo4j connection
    logger.info("Verifying Neo4j connection...")
    if not pipeline.graph_store.verify_connection():
        logger.error("Neo4j connection failed")
        return False
    
    # Find sample docs
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} not found; skipping ingestion")
        return True
    
    sample_docs = list(data_path.glob("*.txt")) + list(data_path.glob("*.pdf"))
    if not sample_docs:
        logger.warning(f"No .txt or .pdf files in {data_dir}")
        return True
    
    sample_docs = [str(p) for p in sample_docs[:5]]  # Limit to 5 files
    logger.info(f"Found {len(sample_docs)} sample documents")
    
    # Build indexes
    try:
        pipeline.build_indexes(sample_docs)
    except Exception as e:
        logger.error(f"build_indexes failed: {e}")
        return False
    
    # Verify graph data
    logger.info("Verifying graph data...")
    try:
        with pipeline.graph_store._driver.session(database=pipeline.graph_store.database) as session:
            n_chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
            n_entities = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]
            n_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            
            logger.info(f"Graph contains: {n_chunks} chunks, {n_entities} entities, {n_rels} relations")
            
            if n_chunks == 0:
                logger.error("No chunks in graph!")
                return False
    except Exception as e:
        logger.error(f"Graph verification failed: {e}")
        return False
    
    # Test retrieval + generation
    query = "What is discussed in the documents?"
    logger.info(f"Testing retrieval & generation with query: '{query}'")
    
    try:
        answer = pipeline.answer(query)
        logger.info(f"Answer: {answer[:200]}...")
        
        if "No supporting context" in answer:
            logger.warning("Generator returned 'No supporting context' - retrieval may be empty")
            return False
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return False
    
    logger.info("=" * 80)
    logger.info("✓ Pipeline validation successful!")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GraphRAG pipeline")
    parser.add_argument("--config", default="configs/example_config.yaml", help="Config file path")
    parser.add_argument("--data", default="data", help="Data directory with documents")
    args = parser.parse_args()
    
    success = validate_pipeline(args.config, args.data)
    sys_exit(0 if success else 1)
