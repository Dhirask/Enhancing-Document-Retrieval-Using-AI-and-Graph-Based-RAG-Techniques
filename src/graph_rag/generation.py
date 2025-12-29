import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import google.genai as genai

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
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment for generation.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.config.model_name)

    def _build_context(self, reranked: RerankedResult, max_chars: int = 12000) -> Tuple[str, List[str]]:
        seen = set()
        parts: List[str] = []
        citations: List[str] = []
        total = 0
        for hit in reranked.items:
            if hit.chunk.id in seen:
                continue
            snippet = " ".join(hit.chunk.text.split()).strip()
            if not snippet:
                continue
            new_total = total + len(snippet)
            if new_total > max_chars:
                break
            parts.append(f"[{hit.chunk.id}] {snippet}")
            citations.append(hit.chunk.id)
            seen.add(hit.chunk.id)
            total = new_total
        return "\n".join(parts), citations

    def generate(self, reranked: RerankedResult, query: str) -> GenerationResult:
        context, citations = self._build_context(reranked)
        if not context:
            return GenerationResult(answer="No supporting context available to answer the query.", citations=[])

        system_prompt = (
            "You are a precise assistant. "
            "Use ONLY the provided context. "
            "Every factual claim MUST be cited with chunk IDs in brackets. "
            "If the answer cannot be derived from the context, reply: "
            "'Insufficient context to answer the query.'"
        )
        user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer the query grounded strictly in the context."

        try:
            resp = self._model.generate_content(
                [
                    {"role": "system", "parts": [system_prompt]},
                    {"role": "user", "parts": [user_prompt]},
                ],
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                },
            )
            message_content = resp.text or ""
            answer = message_content.strip() or "LLM returned empty content."
        except Exception:
            logger.exception("LLM generation failed")
            answer = "LLM generation failed; please try again later."

        return GenerationResult(answer=answer, citations=citations)
