"""
Cross-encoder re-ranker for improving retrieval precision.

Problem: Vector similarity (cosine) retrieves semantically close chunks but
can miss relevance nuance. A cross-encoder jointly encodes (query, passage)
pairs for significantly better relevance scoring.

Strategy:
  1. Retrieve ``top_k * expansion_factor`` candidates via vector/hybrid search.
  2. Score all candidates with the cross-encoder.
  3. Return the top ``top_k`` by cross-encoder score.

This "retrieve broad, rank precise" pattern is standard in production IR systems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RerankerConfig:
    """Configuration for the cross-encoder re-ranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Retrieve this many more candidates than needed, then re-rank down to top_k
    expansion_factor: int = 3
    # Minimum score a chunk must have after re-ranking to be included
    min_score: float = -10.0  # very permissive; let caller decide filtering


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RerankerInitError(RuntimeError):
    """Raised when the cross-encoder model cannot be loaded."""


# ---------------------------------------------------------------------------
# Re-ranker
# ---------------------------------------------------------------------------

@dataclass
class RankedChunk:
    """A retrieved chunk annotated with its re-ranking score."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    original_rank: int    # rank before re-ranking (0-indexed)
    rerank_score: float   # cross-encoder score (higher = more relevant)
    rerank_rank: int      # rank after re-ranking (0-indexed)

    @property
    def rank_delta(self) -> int:
        """Positive means the chunk moved up; negative means it moved down."""
        return self.original_rank - self.rerank_rank


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved chunks using a cross-encoder model.

    The cross-encoder reads the full (query, passage) text together, giving
    it much richer relevance signal than bi-encoder cosine similarity.

    Usage::

        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(query="What is RAG?", chunks=[...], top_k=5)
        # ranked[0] is now the most relevant chunk
    """

    def __init__(self, config: RerankerConfig | None = None) -> None:
        self.config = config or RerankerConfig()
        logger.info("Loading re-ranker model: %s", self.config.model_name)
        try:
            self._model = CrossEncoder(self.config.model_name)
        except Exception as exc:
            raise RerankerInitError(
                f"Failed to load re-ranker model '{self.config.model_name}': {exc}"
            ) from exc
        logger.info("Re-ranker ready.")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[RankedChunk]:
        """
        Re-rank retrieved chunks for the given query.

        Args:
            query: The user's question.
            chunks: Raw retrieved chunks. Each must contain ``"text"``,
                    ``"chunk_id"``, and ``"metadata"`` keys.
            top_k: Number of top-ranked chunks to return.

        Returns:
            List of ``RankedChunk`` objects sorted by cross-encoder score
            (descending), limited to ``top_k`` items.

        Notes:
            - If ``len(chunks) <= top_k``, all chunks are returned (re-ranked).
            - Chunks that score below ``config.min_score`` are excluded.
        """
        if not chunks:
            logger.warning("Re-ranker received empty chunk list.")
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        raw_scores: List[float] = self._model.predict(pairs).tolist()

        # Build RankedChunk objects preserving original order
        candidates: List[RankedChunk] = [
            RankedChunk(
                chunk_id=chunk.get("chunk_id", f"chunk-{i}"),
                text=chunk["text"],
                metadata=chunk.get("metadata", {}),
                original_rank=i,
                rerank_score=round(float(score), 4),
                rerank_rank=-1,  # filled below
            )
            for i, (chunk, score) in enumerate(zip(chunks, raw_scores))
        ]

        # Filter by minimum score
        candidates = [c for c in candidates if c.rerank_score >= self.config.min_score]

        # Sort by cross-encoder score (highest first)
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)

        # Assign final re-rank positions
        for rank, candidate in enumerate(candidates):
            candidate.rerank_rank = rank

        result = candidates[:top_k]

        logger.info(
            "Re-ranked %d chunks -> returning top %d. "
            "Score range: [%.3f, %.3f]",
            len(chunks), len(result),
            result[-1].rerank_score if result else 0,
            result[0].rerank_score if result else 0,
        )

        for rc in result:
            direction = "^" if rc.rank_delta > 0 else ("v" if rc.rank_delta < 0 else "=")
            logger.debug(
                "  [rank %d%s%d] score=%.4f id=%s",
                rc.rerank_rank, direction, abs(rc.rank_delta), rc.rerank_score, rc.chunk_id,
            )

        return result

    def rerank_to_dicts(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper: re-rank and return plain dicts compatible with
        the existing RAG retrieve interface.

        Returns:
            List of chunk dicts with an added ``"rerank_score"`` key.
        """
        ranked = self.rerank(query, chunks, top_k)
        return [
            {
                "chunk_id": rc.chunk_id,
                "text": rc.text,
                "metadata": rc.metadata,
                "rerank_score": rc.rerank_score,
                "original_rank": rc.original_rank,
            }
            for rc in ranked
        ]
