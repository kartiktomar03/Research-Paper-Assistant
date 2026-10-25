"""Small wrapper around Groq chat and optional vision models."""
from __future__ import annotations

import base64
import os
from pathlib import Path
from groq import Groq


class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b").strip()
        self._client = None

    @property
    def client(self):
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 1800) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def stream_complete(self, system: str, user: str, max_tokens: int = 1800):
        """Yield actual model output deltas (not a simulated typing animation)."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            for choice in chunk.choices:
                delta = choice.delta
                content = getattr(delta, "content", None)
                if content:
                    yield content

    def summarize_stream(self, text: str, title: str = "Research section"):
        yield from self.stream_complete(
            "You are a careful academic research assistant. Ground every claim in the supplied source. If the text lacks a detail, say it is not specified. Format with concise Markdown headings and bullets.",
            f"Summarize this section titled {title!r}. Include: central idea, methods/arguments, findings, limitations or caveats stated in the text, and a simple explanation. Do not invent results.\n\nSOURCE TEXT:\n{text[:24000]}",
            max_tokens=1800,
        )

    def answer_stream(self, question: str, context: str):
        yield from self.stream_complete(
            "You answer questions about academic papers using only the supplied retrieved context. Cite source markers such as [Page 4] or [Table 2] in your answer. If the evidence is insufficient, say you could not find the answer in the uploaded paper. Never fabricate values or citations.",
            f"RETRIEVED CONTEXT:\n{context[:18000]}\n\nQUESTION:\n{question}",
            max_tokens=1400,
        )

    def summarize(self, text: str, title: str = "Research section") -> str:
        return self.complete(
            "You are a careful academic research assistant. Ground every claim in the supplied source. If the text lacks a detail, say it is not specified. Format with concise Markdown headings and bullets.",
            f"Summarize this section titled {title!r}. Include: central idea, methods/arguments, findings, limitations or caveats stated in the text, and a simple explanation. Do not invent results.\n\nSOURCE TEXT:\n{text[:24000]}",
            max_tokens=1800,
        )

    def answer(self, question: str, context: str) -> str:
        return self.complete(
            "You answer questions about academic papers using only the supplied retrieved context. Cite source markers such as [Page 4] or [Table 2] in your answer. If the evidence is insufficient, say you could not find the answer in the uploaded paper. Never fabricate values or citations.",
            f"RETRIEVED CONTEXT:\n{context[:18000]}\n\nQUESTION:\n{question}",
            max_tokens=1400,
        )

    def describe_image(self, image_path: Path, model: str) -> str:
        """Use an explicitly configured Groq vision model to describe a figure."""
        data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        suffix = image_path.suffix.lower()
        mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
        response = self.client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this research-paper figure accurately. Identify chart type, axes, labels, trends, legend and any readable values. Clearly state uncertainty; do not infer exact numbers that are not legible."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }],
            temperature=0.1,
            max_tokens=700,
        )
        return (response.choices[0].message.content or "").strip()

    def compare_stream(self, question: str, context: str):
        yield from self.stream_complete(
            "You are an evidence-grounded cross-paper research assistant. Compare the selected papers using ONLY the evidence supplied. "
            "Clearly attribute EVERY finding and numerical value to Paper 1, Paper 2, etc., and include page markers when available. "
            "Do not merge results across papers. State 'Not reported in retrieved evidence' for missing information. "
            "Do not rank accuracy across different datasets as directly comparable. "
            "Include a comparison table when the question calls for comparison. Preserve each paper's actual title.",
            f"EVIDENCE FROM SELECTED PAPERS:\n{context[:22000]}\n\nUSER QUESTION:\n{question}",
            max_tokens=2400,
        )

    def compare(self, question: str, context: str) -> str:
        return "".join(self.compare_stream(question, context))
