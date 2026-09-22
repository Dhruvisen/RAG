"""
HyDE — Hypothetical Document Embeddings for improved retrieval.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
           https://arxiv.org/abs/2212.10496

Motivation:
  - Short queries ("What is CRAG?") embed poorly because they lack context.
  - A hypothetical answer ("CRAG stands for Corrective RAG, a technique that...")
    is longer and richer, producing a better embedding vector for retrieval.
  - HyDE improves recall especially for abstract, vague, or short queries.

Architecture:
  1. Generator produces a hypothetical answer paragraph for the query.
  2. The hypothetical paragraph is embedded (not the raw query).
  3. The embedding is used to retrieve from the vector store.

The original query is preserved for display and grading purposes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HyDEConfig:
    """Configuration for HyDE query expansion."""
    # Whether HyDE is enabled (can be disabled for latency-sensitive paths)
    enabled: bool = True
    # Embedding model must match the one used in the vector store
    embedding_model: str = "all-MiniLM-L6-v2"
    # If the hypothetical doc is shorter than this, fall back to raw query embedding
    min_hyp_doc_length: int = 50


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HyDEError(RuntimeError):
    """Raised when HyDE fails to initialise or expand a query."""


# ---------------------------------------------------------------------------
# HyDE expander
# ---------------------------------------------------------------------------

@dataclass
class HyDEResult:
    """Result of a HyDE expansion."""
    original_query: str
    hypothetical_document: str      # LLM-generated passage
    embedding: list[float]          # embedding of the hypothetical document
    used_hyde: bool                 # False if fallback to raw query was used


class HyDEExpander:
    """
    Expands a user query into a hypothetical answer document, then embeds it.

    This class is intentionally decoupled from the OllamaGenerator — it accepts
    a callable ``generator_fn`` so that it can work with any LLM backend.

    Usage::

        from src.generator import OllamaGenerator
        gen = OllamaGenerator()
        hyde = HyDEExpander(generator_fn=gen.generate_hypothetical_document)

        result = hyde.expand("What is self-correcting RAG?")
        # Use result.embedding for vector store retrieval
    """

    def __init__(
        self,
        generator_fn,          # Callable[[str], str]
        config: HyDEConfig | None = None,
    ) -> None:
        """
        Args:
            generator_fn: A callable that takes a query string and returns a
                          hypothetical answer string (e.g., OllamaGenerator.generate_hypothetical_document).
            config: Optional HyDEConfig; defaults are used if not provided.
        """
        self.config = config or HyDEConfig()
        self._generator_fn = generator_fn

        logger.info("Loading HyDE embedding model: %s", self.config.embedding_model)
        try:
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
        except Exception as exc:
            raise HyDEError(
                f"Failed to load embedding model '{self.config.embedding_model}': {exc}"
            ) from exc

    def expand(self, query: str) -> HyDEResult:
        """
        Generate a hypothetical document for the query and embed it.

        Args:
            query: The user's original question.

        Returns:
            ``HyDEResult`` containing the hypothetical document and its embedding.
            If HyDE is disabled or generation fails, falls back to embedding the
            raw query directly (``used_hyde=False``).
        """
        if not self.config.enabled:
            logger.debug("HyDE is disabled; embedding raw query.")
            return self._fallback(query, reason="HyDE disabled")

        try:
            hyp_doc = self._generator_fn(query)
        except Exception as exc:
            logger.warning("HyDE generation failed: %s — falling back to raw query.", exc)
            return self._fallback(query, reason=str(exc))

        if not hyp_doc or len(hyp_doc) < self.config.min_hyp_doc_length:
            logger.warning(
                "Hypothetical document too short (%d chars < %d min); falling back.",
                len(hyp_doc or ""), self.config.min_hyp_doc_length,
            )
            return self._fallback(query, reason="Hypothetical document too short")

        embedding = self._embed(hyp_doc)
        logger.info(
            "HyDE expanded query '%s...' → %d-char hypothetical doc.",
            query[:50], len(hyp_doc),
        )
        logger.debug("Hypothetical doc: %s...", hyp_doc[:200])

        return HyDEResult(
            original_query=query,
            hypothetical_document=hyp_doc,
            embedding=embedding,
            used_hyde=True,
        )

    def embed_raw_query(self, query: str) -> list[float]:
        """
        Embed the raw query directly (bypasses HyDE).
        Useful for comparison or when HyDE is disabled.
        """
        return self._embed(query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Embed a text string and return the embedding as a plain list."""
        return self._embedding_model.encode([text]).tolist()[0]

    def _fallback(self, query: str, reason: str) -> HyDEResult:
        """Return a HyDEResult using the raw query embedding."""
        logger.debug("HyDE fallback — reason: %s", reason)
        return HyDEResult(
            original_query=query,
            hypothetical_document=query,
            embedding=self._embed(query),
            used_hyde=False,
        )
