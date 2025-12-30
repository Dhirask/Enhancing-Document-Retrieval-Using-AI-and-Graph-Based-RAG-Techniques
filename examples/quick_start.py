#!/usr/bin/env python
"""
Quick-start script: Ingest sample data, build graph, and test end-to-end.

Usage:
    python examples/quick_start.py
"""

import logging
import os
from pathlib import Path

from src.graph_rag.config import PipelineConfig
from src.graph_rag.pipeline import GraphRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("GraphRAG Quick-Start")
    logger.info("=" * 80)
    
    # Verify environment
    required_vars = ["NEO4J_PASSWORD", "GEMINI_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        logger.error("Set them before running this script:")
        for v in missing:
            logger.error(f"  export {v}='...'")
        return False
    
    logger.info("✓ All required env vars set")
    
    # Init pipeline
    cfg = PipelineConfig()
    pipeline = GraphRAGPipeline(cfg)
    
    # Verify Neo4j
    logger.info("Connecting to Neo4j...")
    if not pipeline.graph_store.verify_connection():
        logger.error("Could not connect to Neo4j")
        return False
    
    logger.info("✓ Connected to Neo4j")
    
    # Find sample docs
    data_dir = Path("data")
    if not data_dir.exists():
        logger.info(f"Creating sample data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a minimal sample
        sample_file = data_dir / "sample.txt"
        sample_file.write_text(
            "Machine Learning is a subset of Artificial Intelligence. "
            "Neural Networks are inspired by biological neurons. "
            "Deep Learning uses multiple layers of neural networks. "
            "Transformers revolutionized Natural Language Processing. "
            "Attention mechanisms allow models to focus on relevant parts. "
        )
        logger.info(f"Created sample file: {sample_file}")
    
    sample_docs = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.pdf"))
    if not sample_docs:
        logger.error(f"No documents found in {data_dir}")
        return False
    
    logger.info(f"Found {len(sample_docs)} documents")
    
    # Build indexes
    try:
        logger.info("Building indexes (ingestion → Neo4j → FAISS)...")
        pipeline.build_indexes([str(p) for p in sample_docs])
        logger.info("✓ Indexes built successfully")
    except Exception as e:
        logger.error(f"✗ Build failed: {e}")
        return False
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How do transformers work?",
        "Explain neural networks",
    ]
    
    logger.info("\n" + "=" * 80)
    logger.info("Testing Retrieval & Generation")
    logger.info("=" * 80)
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        try:
            answer = pipeline.answer(query)
            logger.info(f"Answer: {answer}")
        except Exception as e:
            logger.error(f"✗ Generation failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Quick-start complete!")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
