"""
Hybrid search combining BM25 (keyword) and vector (semantic) retrieval.

Motivation:
  - Vector search excels at semantic similarity but misses exact keyword matches.
  - BM25 excels at keyword matches but ignores semantic meaning.
  - Combining both via Reciprocal Rank Fusion (RRF) outperforms either alone.

RRF formula:  score(d) = Σ  1 / (k + rank_in_list_i(d))
where k=60 is a smoothing constant (standard in IR literature).

The hybrid search is computed entirely in-process — no extra infrastructure needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HybridSearchConfig:
    """Configuration for the hybrid BM25 + vector search."""
    # RRF smoothing constant (k=60 recommended by the original RRF paper)
    rrf_k: int = 60
    # Weight applied to BM25 scores before fusion (1.0 = equal weight)
    bm25_weight: float = 1.0
    # Weight applied to vector scores before fusion (1.0 = equal weight)
    vector_weight: float = 1.0
    # Minimum BM25 token length to avoid noise from very short tokens
    min_token_length: int = 2


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HybridSearchError(RuntimeError):
    """Raised when hybrid search cannot be performed."""


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """A search result from the hybrid retriever."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    rrf_score: float
    bm25_rank: Optional[int]    # None if not in BM25 results
    vector_rank: Optional[int]  # None if not in vector results


class HybridSearcher:
    """
    Combines BM25 keyword search with vector search using Reciprocal Rank Fusion.

    The BM25 index is built on the fly from the provided corpus and is
    stateless across calls — it is re-built each time ``search`` is called.
    For large static corpora, consider caching the BM25 index externally.

    Usage::

        searcher = HybridSearcher()
        results = searcher.search(
            query="What is CRAG?",
            vector_results=[...],   # from your existing vector store
            corpus_chunks=[...],    # all stored chunks for BM25 indexing
            top_k=5,
        )
    """

    def __init__(self, config: HybridSearchConfig | None = None) -> None:
        self.config = config or HybridSearchConfig()
        logger.debug(
            "HybridSearcher initialised (rrf_k=%d, bm25_w=%.1f, vec_w=%.1f)",
            self.config.rrf_k, self.config.bm25_weight, self.config.vector_weight,
        )

    def search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        corpus_chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[HybridResult]:
        """
        Perform hybrid BM25 + vector search with RRF fusion.

        Args:
            query: User question string.
            vector_results: Already-retrieved vector search results (ordered by
                            cosine similarity, highest first). Each dict must
                            contain ``"chunk_id"`` and ``"text"``.
            corpus_chunks: The full set of stored chunks used to build the BM25
                           index. Each dict must contain ``"chunk_id"`` and ``"text"``.
            top_k: Number of top results to return after fusion.

        Returns:
            List of ``HybridResult`` sorted by RRF score (descending).

        Raises:
            HybridSearchError: If the corpus is empty.
        """
        if not corpus_chunks:
            raise HybridSearchError("corpus_chunks must not be empty for BM25 indexing.")

        bm25_results = self._bm25_search(query, corpus_chunks, top_k=len(corpus_chunks))
        fused = self._reciprocal_rank_fusion(
            bm25_ranked=bm25_results,
            vector_ranked=vector_results,
            corpus_chunks=corpus_chunks,
        )

        top_results = fused[:top_k]
        logger.info(
            "Hybrid search returned %d results (BM25 corpus size: %d, "
            "vector candidates: %d)",
            len(top_results), len(corpus_chunks), len(vector_results),
        )
        return top_results

    def search_to_dicts(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        corpus_chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Convenience wrapper returning plain dicts compatible with the RAG interface.
        """
        results = self.search(query, vector_results, corpus_chunks, top_k)
        return [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "metadata": r.metadata,
                "rrf_score": r.rrf_score,
                "bm25_rank": r.bm25_rank,
                "vector_rank": r.vector_rank,
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokeniser with minimum length filter."""
        return [
            token.lower()
            for token in text.split()
            if len(token) >= self.config.min_token_length
        ]

    def _bm25_search(
        self,
        query: str,
        corpus_chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Build a BM25 index over corpus_chunks and retrieve top_k results."""
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in corpus_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            logger.warning("BM25 tokenised query is empty — BM25 will contribute nothing.")
            return []

        scores = bm25.get_scores(tokenized_query)

        # Pair each chunk with its BM25 score and sort descending
        scored = sorted(
            zip(corpus_chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored[:top_k]:
            if score > 0:
                results.append({**chunk, "_bm25_score": float(score)})

        logger.debug(
            "BM25 search: %d non-zero scored chunks out of %d total.",
            len(results), len(corpus_chunks),
        )
        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_ranked: List[Dict[str, Any]],
        vector_ranked: List[Dict[str, Any]],
        corpus_chunks: List[Dict[str, Any]],
    ) -> List[HybridResult]:
        """
        Fuse BM25 and vector ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ weight_i / (k + rank_i)
        """
        k = self.config.rrf_k
        chunk_map: Dict[str, Dict[str, Any]] = {
            c["chunk_id"]: c for c in corpus_chunks
        }

        # Build rank lookup dicts (0-indexed)
        bm25_ranks: Dict[str, int] = {
            c["chunk_id"]: rank for rank, c in enumerate(bm25_ranked)
        }
        vector_ranks: Dict[str, int] = {
            c["chunk_id"]: rank for rank, c in enumerate(vector_ranked)
        }

        # Collect all unique chunk IDs across both result lists
        all_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        rrf_scores: Dict[str, float] = {}
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += self.config.bm25_weight / (k + bm25_ranks[chunk_id])
            if chunk_id in vector_ranks:
                score += self.config.vector_weight / (k + vector_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by RRF score and build HybridResult list
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        results: List[HybridResult] = []
        for chunk_id in sorted_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                # chunk only appeared in vector results, not in corpus map
                # find it in vector_ranked
                chunk = next(
                    (c for c in vector_ranked if c.get("chunk_id") == chunk_id), None
                )
            if chunk is None:
                logger.warning("Chunk %s not found in corpus_map; skipping.", chunk_id)
                continue

            results.append(HybridResult(
                chunk_id=chunk_id,
                text=chunk.get("text", ""),
                metadata=chunk.get("metadata", {}),
                rrf_score=round(rrf_scores[chunk_id], 6),
                bm25_rank=bm25_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
            ))

        return results
