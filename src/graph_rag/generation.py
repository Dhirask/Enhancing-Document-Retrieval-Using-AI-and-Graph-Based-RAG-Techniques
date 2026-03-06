import logging
import os
from dataclasses import dataclass
from typing import Dict, List
from openai import OpenAI

from .config import GenerationConfig
from .rerank import RerankedResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    answer: str
    citations: List[str]


class Generator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=api_key)
        self.model_name = config.model_name

    def _build_context(self, reranked: RerankedResult, max_chars: int = 12000):
        seen, parts, citations, total = set(), [], [], 0

        for hit in reranked.items:
            cid = hit.chunk.id
            if cid in seen:
                continue

            text = " ".join(hit.chunk.text.split())
            if total + len(text) > max_chars:
                break

            parts.append(f"[{cid}] {text}")
            citations.append(cid)
            seen.add(cid)
            total += len(text)

        return "\n".join(parts), citations

    @staticmethod
    def _format_triples(triples: List[Dict]) -> str:
        """Format knowledge graph triples for the prompt.

        Uses arrow notation:  Entity A → relationship → Entity B
        Includes chunk provenance when available.
        """
        if not triples:
            return ""

        lines = []
        for t in triples:
            subj = t.get("subject", "?")
            pred_label = t.get("predicate_label", t.get("predicate", "?"))
            obj = t.get("object", "?")
            chunk_id = t.get("chunk_id", "")

            provenance = f"  [source: {chunk_id}]" if chunk_id else ""
            lines.append(f"{subj} \u2192 {pred_label} \u2192 {obj}{provenance}")

        return "\n".join(lines)

    def generate(
        self,
        reranked: RerankedResult,
        query: str,
        graph_triples: List[Dict] = None
    ) -> GenerationResult:

        context, citations = self._build_context(reranked)

        if not context:
            return GenerationResult(
                "No supporting context available to answer the query.",
                []
            )

        triples_text = self._format_triples(graph_triples) if graph_triples else ""

        # -------------------------------
        # USER PROMPT
        # -------------------------------

        # Build prompt with internal-only context markers
        parts: list[str] = []

        parts.append(
            "[INTERNAL CONTEXT — do NOT reproduce any of this verbatim]\n"
        )

        if triples_text:
            parts.append(f"Graph relationships:\n{triples_text}\n")

        parts.append(f"Document passages:\n{context}\n")

        parts.append("[END INTERNAL CONTEXT]\n")

        parts.append(
            "Using ONLY the internal context above, answer the following question "
            "in clear, concise prose.\n\n"
            "Rules:\n"
            "- Give a direct answer — no bullet-point relationship lists.\n"
            "- Do NOT mention chunk IDs, sources, citations, or triple notation.\n"
            "- Do NOT list entities or show reasoning paths.\n"
            "- If the context is insufficient, reply exactly: "
            "'Insufficient context to answer the query.'\n\n"
            f"Question: {query}"
        )

        # -------------------------------
        # LLM CALL
        # -------------------------------

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise, knowledgeable assistant. "
                            "You internally use knowledge graph relationships to reason "
                            "about entity connections, but you only output a clear, "
                            "concise answer. Never expose raw triples, reasoning steps, "
                            "or entity lists to the user."
                        )
                    },
                    {
                        "role": "user",
                        "content": "\n".join(parts)
                    }
                ],
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            answer = response.output_text.strip()

        except Exception as e:
            logger.error("LLM generation failed", exc_info=e)
            return GenerationResult(answer="LLM generation failed.", citations=[])

        return GenerationResult(answer=answer, citations=citations)