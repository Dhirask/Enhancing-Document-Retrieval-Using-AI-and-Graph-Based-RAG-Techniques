import os
from dataclasses import dataclass
from typing import List
from google import genai  # correct import from the google-genai package

from .config import GenerationConfig
from .rerank import RerankedResult


@dataclass
class GenerationResult:
    answer: str
    citations: List[str]


class Generator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GENAI_API_KEY environment variable")

        # Initialize the current GenAI client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "text-bison-001"  

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
            return GenerationResult("No supporting context available to answer the query.", [])

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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                # optional config if needed:
                # config={"temperature": self.config.temperature,
                #         "max_output_tokens": self.config.max_tokens},
            )
            answer = response.text.strip()
        except Exception as e:
            answer = f"LLM generation failed; error: {e}"

        return GenerationResult(answer=answer, citations=citations)
