"""
Retrieval and answer quality graders using cross-encoder models.

Two graders are provided:

1. ``RetrievalGrader`` — Scores each retrieved chunk against the query.
   Uses a cross-encoder (ms-marco) which is purpose-built for passage relevance.
   Chunks below the configured threshold are flagged as irrelevant.

2. ``AnswerGrader`` — Scores the generated answer against the original question
   and retrieved context to detect hallucinations or unsupported claims.

Both graders are **LLM-free** and run entirely on CPU using sentence-transformers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RETRIEVAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_ANSWER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class RelevanceLabel(str, Enum):
    """Human-readable relevance judgement for a retrieved chunk."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


@dataclass
class ChunkGrade:
    """Grading result for a single retrieved chunk."""
    chunk_id: str
    score: float
    label: RelevanceLabel
    text_preview: str  # first 120 chars for logging / debugging


@dataclass
class RetrievalGraderConfig:
    """Configuration for the retrieval grader."""
    model_name: str = _DEFAULT_RETRIEVAL_MODEL
    # Thresholds calibrated for ms-marco cross-encoder (logit range ~-10 to +10)
    # Lowered thresholds since markdown tables often score lower than natural language
    relevant_threshold: float = -3.0
    partial_threshold: float = -7.0
    # If the fraction of relevant+partial chunks is below this, trigger fallback
    min_acceptable_ratio: float = 0.3


@dataclass
class AnswerGraderConfig:
    """Configuration for the answer grader."""
    model_name: str = _DEFAULT_ANSWER_MODEL
    # Score below which the answer is considered hallucinated / unsupported
    supported_threshold: float = 0.4


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GraderInitError(RuntimeError):
    """Raised when a grader model cannot be loaded."""


# ---------------------------------------------------------------------------
# Retrieval Grader
# ---------------------------------------------------------------------------

class RetrievalGrader:
    """
    Grades retrieved chunks for relevance to a query using a cross-encoder.

    The cross-encoder is more accurate than cosine similarity because it jointly
    encodes the query and passage (rather than comparing independent embeddings).

    Usage::

        grader = RetrievalGrader()
        grades = grader.grade(query="What is RAG?", chunks=[...])
        good = grader.filter_relevant(grades)
    """

    def __init__(self, config: RetrievalGraderConfig | None = None) -> None:
        self.config = config or RetrievalGraderConfig()
        logger.info("Loading retrieval grader model: %s", self.config.model_name)
        try:
            self._model = CrossEncoder(self.config.model_name)
        except Exception as exc:
            raise GraderInitError(
                f"Failed to load cross-encoder '{self.config.model_name}': {exc}"
            ) from exc

    def grade(self, query: str, chunks: List[Dict[str, Any]]) -> List[ChunkGrade]:
        """
        Score each chunk against the query.

        Args:
            query: The user's question.
            chunks: Retrieved chunks; each dict must have ``"text"`` and ``"chunk_id"``.

        Returns:
            List of ``ChunkGrade`` objects, ordered by input order.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        raw_scores: List[float] = self._model.predict(pairs).tolist()

        grades: List[ChunkGrade] = []
        for chunk, score in zip(chunks, raw_scores):
            label = self._score_to_label(score)
            grade = ChunkGrade(
                chunk_id=chunk.get("chunk_id", "unknown"),
                score=round(float(score), 4),
                label=label,
                text_preview=chunk["text"][:120].replace("\n", " "),
            )
            grades.append(grade)
            logger.debug(
                "Chunk %s → score=%.4f label=%s", grade.chunk_id, grade.score, grade.label
            )

        return grades

    def filter_relevant(
        self, grades: List[ChunkGrade]
    ) -> Tuple[List[ChunkGrade], bool]:
        """
        Split grades into acceptable and insufficient groups.

        Args:
            grades: Output of ``grade()``.

        Returns:
            Tuple of:
            - List of relevant + partially relevant ``ChunkGrade`` objects.
            - Boolean ``needs_fallback``: True when retrieval quality is too low.
        """
        acceptable = [
            g for g in grades
            if g.label in (RelevanceLabel.RELEVANT, RelevanceLabel.PARTIALLY_RELEVANT)
        ]
        ratio = len(acceptable) / len(grades) if grades else 0.0
        needs_fallback = ratio < self.config.min_acceptable_ratio

        logger.info(
            "Retrieval grade: %d/%d acceptable (%.0f%%) — fallback=%s",
            len(acceptable), len(grades), ratio * 100, needs_fallback,
        )
        return acceptable, needs_fallback

    def _score_to_label(self, score: float) -> RelevanceLabel:
        if score >= self.config.relevant_threshold:
            return RelevanceLabel.RELEVANT
        if score >= self.config.partial_threshold:
            return RelevanceLabel.PARTIALLY_RELEVANT
        return RelevanceLabel.IRRELEVANT


# ---------------------------------------------------------------------------
# Answer Grader
# ---------------------------------------------------------------------------

@dataclass
class AnswerGrade:
    """Grading result for a generated answer."""
    score: float
    is_supported: bool
    reason: str  # brief human-readable explanation


class AnswerGrader:
    """
    Grades a generated answer for groundedness in the retrieved context.

    Detects hallucinations by scoring how well the answer is supported by
    the evidence passages. Uses the same ms-marco cross-encoder — we treat
    (answer, context_passage) pairs to estimate entailment.

    Usage::

        grader = AnswerGrader()
        grade = grader.grade(question="...", answer="...", context_chunks=[...])
        if not grade.is_supported:
            # trigger retry or fallback
    """

    def __init__(self, config: AnswerGraderConfig | None = None) -> None:
        self.config = config or AnswerGraderConfig()
        logger.info("Loading answer grader model: %s", self.config.model_name)
        try:
            self._model = CrossEncoder(self.config.model_name)
        except Exception as exc:
            raise GraderInitError(
                f"Failed to load answer grader '{self.config.model_name}': {exc}"
            ) from exc

    def grade(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> AnswerGrade:
        """
        Evaluate whether the generated answer is grounded in the context.

        Args:
            question: Original user question.
            answer: LLM-generated answer.
            context_chunks: Chunks used during generation.

        Returns:
            ``AnswerGrade`` with score, support flag, and reason.
        """
        if not context_chunks or not answer.strip():
            return AnswerGrade(
                score=0.0,
                is_supported=False,
                reason="Empty answer or no context provided.",
            )

        # Check if answer claims insufficient info (not a hallucination)
        insufficient_markers = [
            "do not contain enough information",
            "don't have enough information",
            "cannot answer",
            "not enough context",
        ]
        if any(marker in answer.lower() for marker in insufficient_markers):
            return AnswerGrade(
                score=1.0,
                is_supported=True,
                reason="Model correctly identified insufficient context.",
            )

        # Score the answer against each context passage, take the max
        pairs = [(answer, chunk["text"]) for chunk in context_chunks]
        scores: List[float] = self._model.predict(pairs).tolist()
        best_score = float(max(scores))
        is_supported = best_score >= self.config.supported_threshold

        reason = (
            f"Answer is {'supported' if is_supported else 'NOT supported'} "
            f"by context (best score: {best_score:.4f}, "
            f"threshold: {self.config.supported_threshold})."
        )
        logger.info("Answer grade: score=%.4f supported=%s", best_score, is_supported)

        return AnswerGrade(
            score=round(best_score, 4),
            is_supported=is_supported,
            reason=reason,
        )
