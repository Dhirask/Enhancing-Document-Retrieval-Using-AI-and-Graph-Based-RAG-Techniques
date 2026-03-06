import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Ensure src package is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph_rag.config import PipelineConfig  # noqa: E402
from src.graph_rag.graph_store import GraphStore  # noqa: E402
from src.graph_rag.pipeline import GraphRAGPipeline  # noqa: E402


def _ensure_clean_graph_once() -> None:
    """Clear Neo4j once per Streamlit session to avoid cross-session residue."""
    if st.session_state.get("neo4j_cleared"):
        return

    cfg = PipelineConfig()
    store = GraphStore(cfg.graph.uri, cfg.graph.user, cfg.graph.password, cfg.graph.database)
    try:
        store.clear_database()
    finally:
        store.close()

    st.session_state.neo4j_cleared = True


def reset_session(clear_graph: bool = False) -> None:
    """Reset session state and optionally clear graph data for demo isolation."""
    pipeline = st.session_state.get("pipeline")
    if clear_graph and pipeline:
        try:
            pipeline.graph_store.clear_database()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            st.warning(f"Graph cleanup skipped: {exc}")

    temp_dir = st.session_state.get("temp_dir")
    if temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    st.session_state.clear()


def ingest_pdf(uploaded_file) -> None:
    """Save uploaded PDF to temp dir and build indexes."""
    reset_session(clear_graph=True)
    _ensure_clean_graph_once()

    temp_dir = tempfile.mkdtemp(prefix="graphrag_")
    try:
        pdf_path = Path(temp_dir) / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.read())

        cfg = PipelineConfig()
        # Force OpenAI model for demo (env override optional)
        cfg.generation.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        pipeline = GraphRAGPipeline(cfg)

        with st.status("Indexing document...", expanded=True) as status:
            status.write("Chunking & entity extraction...")
            pipeline.build_indexes([str(pdf_path)])
            status.write("Stored chunks/entities in Neo4j & FAISS")
            status.update(label="Indexing complete", state="complete")

        st.session_state.pipeline = pipeline
        st.session_state.temp_dir = temp_dir
        st.session_state.ingested = True
        st.session_state.uploaded_path = str(pdf_path)
    except Exception:
        # Clean up the temp directory on failure before re-raising
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def answer_question(query: str) -> Tuple[str, List[dict]]:
    """Delegate to the pipeline — single source of truth for prompts & LLM."""
    if not st.session_state.get("ingested"):
        raise RuntimeError("Please upload and ingest a PDF first.")

    pipeline: GraphRAGPipeline = st.session_state.pipeline
    answer, graph_triples = pipeline.answer(query)
    return answer, graph_triples


def main() -> None:
    st.set_page_config(page_title="GraphRAG Demo", layout="centered")
    _ensure_clean_graph_once()
    st.title("Graph-based RAG")
    st.caption("Upload a PDF, and ask grounded questions.")

    # Upload & ingestion
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded is not None:
        if st.button("Ingest PDF", type="primary"):
            try:
                ingest_pdf(uploaded)
                st.success("Ingestion complete. You can now ask questions.")
            except Exception as exc:
                reset_session(clear_graph=False)
                st.error(f"Ingestion failed: {exc}")

    # Q&A
    question = st.text_input("Ask a question about the uploaded document", disabled=not st.session_state.get("ingested", False))
    if st.session_state.get("ingested") and question:
        if st.button("Get Answer", type="primary"):
            with st.spinner("Retrieving and generating..."):
                try:
                    answer, graph_triples = answer_question(question)
                    st.write("### Answer")
                    st.write(answer)

                    # Show reasoning details behind an expander
                    if graph_triples:
                        with st.expander("Show Reasoning (Entity Relationships)"):
                            st.write("**Knowledge Graph Relationships used for reasoning:**")
                            for t in graph_triples:
                                subj = t.get("subject", "?")
                                pred = t.get("predicate_label", t.get("predicate", "?"))
                                obj = t.get("object", "?")
                                chunk_id = t.get("chunk_id", "")
                                src = f"  \u2190 _{chunk_id}_" if chunk_id else ""
                                st.markdown(f"- **{subj}** \u2192 {pred} \u2192 **{obj}**{src}")
                except Exception as exc:
                    st.error(f"Failed to answer: {exc}")


if __name__ == "__main__":
    main()
