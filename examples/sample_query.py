from pathlib import Path

from src.graph_rag.config import PipelineConfig
from src.graph_rag.pipeline import GraphRAGPipeline


if __name__ == "__main__":
    cfg = PipelineConfig()
    pipeline = GraphRAGPipeline(cfg)

    # Index sample docs (place your own files under data/)
    sample_docs = [str(p) for p in Path("data").glob("*.txt")]
    pipeline.build_indexes(sample_docs)

    question = "What is the main contribution of the paper?"
    answer = pipeline.answer(question)
    print("Q:", question)
    print("A:", answer)
