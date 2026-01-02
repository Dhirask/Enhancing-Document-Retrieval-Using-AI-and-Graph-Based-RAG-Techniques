import os
from dataclasses import dataclass
from typing import List
from openai import OpenAI

from .config import GenerationConfig
from .rerank import RerankedResult


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

        # Cheap + fast, perfect for RAG
        self.model_name = "gpt-4o-mini"

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

    def generate(self, reranked: RerankedResult, query: str) -> GenerationResult:
        context, citations = self._build_context(reranked)

        if not context:
            return GenerationResult(
                "No supporting context available to answer the query.",
                []
            )

        prompt = f"""
You are a precise assistant.
Answer ONLY using the provided context.
Cite sources using [chunk_id].

Query:
{query}

Context:
{context}
"""

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            answer = response.output_text.strip()

        except Exception as e:
            answer = f"LLM generation failed; error: {e}"

        return GenerationResult(answer=answer, citations=citations)
