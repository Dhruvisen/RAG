"""
Ollama LLM wrapper for local, open-source text generation.

Supports:
- Streaming and non-streaming generation
- RAG-specific prompt construction
- HyDE (Hypothetical Document Embeddings) query generation
- Configurable model, temperature, and token limits
- Automatic connection validation with descriptive errors

Model recommendation (no GPU, CPU-only):
  qwen2.5:1.5b  - Alibaba Qwen2.5 1.5B (935MB, ~2-3s/query on CPU)
                   Grouped-query attention makes it 4-5x faster than llama3.2:3b.
                   Ideal for RAG: the model synthesises from context, not memory.

  Pull with: ollama pull qwen2.5:1.5b
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Generator, List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """
    Immutable configuration for the OllamaGenerator.

    Defaults are tuned for qwen2.5:1.5b on CPU:
      - max_tokens=384: RAG answers should be concise; fewer tokens = faster.
      - temperature=0.05: Near-deterministic - RAG needs factual, not creative output.
      - request_timeout=60: 1.5B model responds in <30s on CPU; 60s is safe headroom.
    """
    model: str = "qwen2.5:1.5b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.05   # near-deterministic for factual RAG generation
    max_tokens: int = 384       # concise answers = faster on CPU
    request_timeout: int = 60   # qwen2.5:1.5b responds well within 60s on CPU
    # HTTP retry settings for transient network errors
    http_retries: int = 3
    http_backoff_factor: float = 0.5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


class OllamaModelNotFoundError(RuntimeError):
    """Raised when the requested model is not pulled locally."""


class GenerationError(RuntimeError):
    """Raised when an LLM generation request fails."""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class OllamaGenerator:
    """
    HTTP client for Ollama's local inference API.

    Default model: qwen2.5:1.5b - optimised for CPU inference.
    Alibaba's Qwen2.5 uses grouped-query attention (GQA) which dramatically
    reduces memory bandwidth on CPU vs standard multi-head attention.

    Usage::

        # Default (fastest CPU model)
        gen = OllamaGenerator()

        # Custom model
        config = GeneratorConfig(model="qwen2.5:0.5b")  # even faster, smaller
        gen = OllamaGenerator(config)

        answer = gen.generate("What is RAG?", context_chunks=[...])

    All methods are synchronous. Use ``generate_stream`` for token-by-token
    output in interactive CLIs.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        self.config = config or GeneratorConfig()
        self._session = self._build_session()
        self._validate_connection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate a complete answer (non-streaming).

        Args:
            prompt: User question or instruction.
            context_chunks: Retrieved document chunks; each dict must contain
                            at least a ``"text"`` key.

        Returns:
            Generated answer string.

        Raises:
            GenerationError: On HTTP or JSON decoding failure.
        """
        full_prompt = self._build_rag_prompt(prompt, context_chunks)
        logger.debug("Sending generation request to Ollama (model=%s)", self.config.model)

        try:
            resp = self._session.post(
                f"{self.config.base_url}/api/generate",
                json=self._build_payload(full_prompt, stream=False),
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.RequestException as exc:
            raise GenerationError(f"Generation request failed: {exc}") from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise GenerationError(f"Unexpected response format from Ollama: {exc}") from exc

    def generate_stream(
        self,
        prompt: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens from the LLM for interactive display.

        Args:
            prompt: User question or instruction.
            context_chunks: Retrieved document chunks.

        Yields:
            Individual token strings as they arrive.

        Raises:
            GenerationError: On HTTP failure.
        """
        full_prompt = self._build_rag_prompt(prompt, context_chunks)
        logger.debug("Starting streaming generation (model=%s)", self.config.model)

        try:
            with self._session.post(
                f"{self.config.base_url}/api/generate",
                json=self._build_payload(full_prompt, stream=True),
                stream=True,
                timeout=self.config.request_timeout,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    data = json.loads(raw_line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            raise GenerationError(f"Streaming request failed: {exc}") from exc

    def generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE — generate a hypothetical passage that answers *query*.

        The returned text is embedded and used as the retrieval vector instead
        of the raw query, significantly improving recall on abstract questions.

        Args:
            query: The user's question.

        Returns:
            A 3-5 sentence hypothetical answer paragraph.
        """
        prompt = (
            "Write a concise, factual paragraph (3-5 sentences) that would "
            "directly answer the following question. Do not mention that it is "
            "hypothetical - write as if it is a real document excerpt.\n\n"
            f"Question: {query}\n\nParagraph:"
        )
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            # HyDE docs are short paragraphs - cap at 150 tokens for speed on CPU
            "options": {"temperature": 0.2, "num_predict": 150},
        }
        try:
            resp = self._session.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json().get("response", "").strip()
            logger.debug("HyDE generated %d chars for query: %s", len(result), query[:60])
            return result or query
        except requests.RequestException as exc:
            logger.warning("HyDE generation failed (%s), falling back to raw query.", exc)
            return query

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_connection(self) -> None:
        """Confirm Ollama is reachable and the model is available."""
        try:
            resp = self._session.get(
                f"{self.config.base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.config.base_url}. "
                "Ensure it is running: `ollama serve`"
            ) from exc
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"Ollama health check failed: {exc}") from exc

        available = [m["name"] for m in resp.json().get("models", [])]
        model_prefix = self.config.model.split(":")[0]
        if not any(model_prefix in name for name in available):
            raise OllamaModelNotFoundError(
                f"Model '{self.config.model}' is not available locally. "
                f"Pull it with: `ollama pull {self.config.model}`\n"
                f"Available models: {available or ['(none)']}"
            )
        logger.info("Ollama connected - model '%s' ready.", self.config.model)

    def _build_session(self) -> requests.Session:
        """Create a requests Session with retry logic for transient failures."""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.http_retries,
            backoff_factor=self.config.http_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _build_payload(self, prompt: str, stream: bool) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

    @staticmethod
    def _build_rag_prompt(
        question: str,
        context_chunks: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Construct a structured RAG prompt from question and context."""
        if not context_chunks:
            return (
                "You are a precise, helpful assistant.\n\n"
                f"Question: {question}\n\nAnswer:"
            )

        context_parts = []
        for idx, chunk in enumerate(context_chunks, start=1):
            source = chunk.get("metadata", {}).get("document_id", f"doc-{idx}")
            context_parts.append(f"[Source {idx} | {source}]\n{chunk['text']}")

        context_text = "\n\n---\n\n".join(context_parts)

        return (
            "You are a precise, helpful assistant. "
            "Answer the question using ONLY the provided context. "
            "If the context does not contain sufficient information, "
            "respond with: 'The provided documents do not contain enough "
            "information to answer this question.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer (concise and factual, cite [Source N] where relevant):"
        )
