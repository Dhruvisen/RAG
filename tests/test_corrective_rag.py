"""
Unit and integration tests for the Corrective RAG pipeline.

Test strategy:
  - Unit tests use mocks to isolate each component.
  - Integration tests are marked with @pytest.mark.integration and require
    a running Ollama server + ChromaDB — skipped in CI by default.

Run unit tests only:
    pytest tests/ -m "not integration" -v

Run all tests (requires Ollama + ChromaDB running):
    pytest tests/ -v
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Suppress noisy logs during tests ─────────────────────────────────────────
logging.disable(logging.CRITICAL)


# =============================================================================
# Fixtures & helpers
# =============================================================================

def _make_chunk(chunk_id: str, text: str, score: float = 0.8) -> Dict[str, Any]:
    """Factory for a mock retrieved chunk dict."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"document_id": "doc-test"},
        "distance": score,
    }


SAMPLE_CHUNKS: List[Dict[str, Any]] = [
    _make_chunk("chunk_0", "Corrective RAG improves retrieval by grading chunks.", 0.9),
    _make_chunk("chunk_1", "HyDE generates a hypothetical document to improve embeddings.", 0.8),
    _make_chunk("chunk_2", "BM25 is a classic keyword-based retrieval algorithm.", 0.7),
    _make_chunk("chunk_3", "Cross-encoders score query-passage pairs jointly.", 0.6),
    _make_chunk("chunk_4", "Irrelevant text about cooking recipes.", 0.1),
]

SAMPLE_QUERY = "What is Corrective RAG and how does it improve retrieval?"


# =============================================================================
# generator.py tests
# =============================================================================

class TestOllamaGenerator:

    @patch("src.generator.requests.Session.get")
    @patch("src.generator.requests.Session.post")
    def test_generate_returns_stripped_response(self, mock_post, mock_get):
        """generate() should return the Ollama response text, stripped."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3.2:3b"}]
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "  This is the answer.  "}

        from src.generator import OllamaGenerator, GeneratorConfig
        gen = OllamaGenerator(GeneratorConfig(model="llama3.2:3b"))
        result = gen.generate("What is RAG?", context_chunks=SAMPLE_CHUNKS[:2])

        assert result == "This is the answer."
        assert mock_post.called

    @patch("src.generator.requests.Session.get")
    def test_connection_error_raises_typed_exception(self, mock_get):
        """Should raise OllamaConnectionError when Ollama is unreachable."""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("refused")

        from src.generator import OllamaGenerator, OllamaConnectionError
        with pytest.raises(OllamaConnectionError, match="Cannot reach Ollama"):
            OllamaGenerator()

    @patch("src.generator.requests.Session.get")
    def test_model_not_found_raises_typed_exception(self, mock_get):
        """Should raise OllamaModelNotFoundError when model is not pulled."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": []}

        from src.generator import OllamaGenerator, OllamaModelNotFoundError
        with pytest.raises(OllamaModelNotFoundError, match="not available locally"):
            OllamaGenerator()

    @patch("src.generator.requests.Session.get")
    @patch("src.generator.requests.Session.post")
    def test_build_rag_prompt_includes_sources(self, mock_post, mock_get):
        """RAG prompt should include numbered sources from context_chunks."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "answer"}

        from src.generator import OllamaGenerator, GeneratorConfig
        gen = OllamaGenerator(GeneratorConfig(model="llama3.2:3b"))
        gen.generate("test query", context_chunks=SAMPLE_CHUNKS[:2])

        call_args = mock_post.call_args
        prompt_sent = call_args[1]["json"]["prompt"]
        assert "[Source 1]" in prompt_sent
        assert "[Source 2]" in prompt_sent
        assert "test query" in prompt_sent


# =============================================================================
# grader.py tests
# =============================================================================

class TestRetrievalGrader:

    @patch("src.grader.CrossEncoder")
    def test_grade_returns_one_result_per_chunk(self, mock_ce_cls):
        """grade() must return exactly one ChunkGrade per input chunk."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.6, 0.3, 0.1, -0.5])
        mock_ce_cls.return_value = mock_model

        from src.grader import RetrievalGrader
        grader = RetrievalGrader()
        grades = grader.grade(SAMPLE_QUERY, SAMPLE_CHUNKS)

        assert len(grades) == len(SAMPLE_CHUNKS)
        assert all(g.chunk_id for g in grades)

    @patch("src.grader.CrossEncoder")
    def test_filter_relevant_flags_fallback(self, mock_ce_cls):
        """filter_relevant should trigger fallback when all scores are low."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-3.0, -4.0, -5.0])
        mock_ce_cls.return_value = mock_model

        from src.grader import RetrievalGrader, RetrievalGraderConfig
        config = RetrievalGraderConfig(min_acceptable_ratio=0.5)
        grader = RetrievalGrader(config)
        grades = grader.grade(SAMPLE_QUERY, SAMPLE_CHUNKS[:3])
        _, needs_fallback = grader.filter_relevant(grades)

        assert needs_fallback is True

    @patch("src.grader.CrossEncoder")
    def test_relevant_label_assigned_correctly(self, mock_ce_cls):
        """Scores above relevant_threshold should get RELEVANT label."""
        import numpy as np
        from src.grader import RetrievalGrader, RelevanceLabel
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.5])
        mock_ce_cls.return_value = mock_model

        grader = RetrievalGrader()
        grades = grader.grade("query", [SAMPLE_CHUNKS[0]])
        assert grades[0].label == RelevanceLabel.RELEVANT


class TestAnswerGrader:

    @patch("src.grader.CrossEncoder")
    def test_grade_supported_answer(self, mock_ce_cls):
        """A high-scoring answer should be marked as supported."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.85, 0.70])
        mock_ce_cls.return_value = mock_model

        from src.grader import AnswerGrader
        grader = AnswerGrader()
        grade = grader.grade(
            question=SAMPLE_QUERY,
            answer="CRAG improves retrieval by grading chunks.",
            context_chunks=SAMPLE_CHUNKS[:2],
        )
        assert grade.is_supported is True

    @patch("src.grader.CrossEncoder")
    def test_grade_insufficient_context_marker(self, mock_ce_cls):
        """Answers containing 'not enough information' should be marked supported."""
        mock_ce_cls.return_value = MagicMock()

        from src.grader import AnswerGrader
        grader = AnswerGrader()
        grade = grader.grade(
            question="What is X?",
            answer="The provided documents do not contain enough information.",
            context_chunks=SAMPLE_CHUNKS[:2],
        )
        assert grade.is_supported is True
        assert grade.score == 1.0


# =============================================================================
# reranker.py tests
# =============================================================================

class TestCrossEncoderReranker:

    @patch("src.reranker.CrossEncoder")
    def test_rerank_returns_top_k(self, mock_ce_cls):
        """rerank() must return at most top_k results."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.9, 0.3, 0.7, 0.1])
        mock_ce_cls.return_value = mock_model

        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(SAMPLE_QUERY, SAMPLE_CHUNKS, top_k=3)

        assert len(ranked) == 3

    @patch("src.reranker.CrossEncoder")
    def test_rerank_sorts_by_score_descending(self, mock_ce_cls):
        """Highest-scoring chunk should be first in result."""
        import numpy as np
        scores = [0.2, 0.9, 0.4, 0.1, 0.6]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores)
        mock_ce_cls.return_value = mock_model

        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        ranked = reranker.rerank(SAMPLE_QUERY, SAMPLE_CHUNKS, top_k=5)

        assert ranked[0].rerank_score >= ranked[1].rerank_score >= ranked[2].rerank_score

    @patch("src.reranker.CrossEncoder")
    def test_rerank_empty_input_returns_empty(self, mock_ce_cls):
        """Empty input should return empty list without error."""
        mock_ce_cls.return_value = MagicMock()

        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []


# =============================================================================
# hybrid_search.py tests
# =============================================================================

class TestHybridSearcher:

    def test_search_returns_top_k(self):
        """Hybrid search must return at most top_k results."""
        from src.utils.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        results = searcher.search(
            query="corrective RAG retrieval",
            vector_results=SAMPLE_CHUNKS,
            corpus_chunks=SAMPLE_CHUNKS,
            top_k=3,
        )
        assert len(results) <= 3

    def test_search_rrf_scores_are_positive(self):
        """All RRF scores must be positive."""
        from src.utils.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        results = searcher.search(
            query="BM25 keyword search",
            vector_results=SAMPLE_CHUNKS,
            corpus_chunks=SAMPLE_CHUNKS,
            top_k=5,
        )
        assert all(r.rrf_score > 0 for r in results)

    def test_search_raises_on_empty_corpus(self):
        """Should raise HybridSearchError when corpus is empty."""
        from src.utils.hybrid_search import HybridSearcher, HybridSearchError
        searcher = HybridSearcher()
        with pytest.raises(HybridSearchError):
            searcher.search("query", [], [], top_k=5)

    def test_search_results_sorted_by_rrf_descending(self):
        """Results must be sorted by RRF score, highest first."""
        from src.utils.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        results = searcher.search(
            query="cross-encoder re-ranking",
            vector_results=SAMPLE_CHUNKS,
            corpus_chunks=SAMPLE_CHUNKS,
            top_k=5,
        )
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# hyde.py tests
# =============================================================================

class TestHyDEExpander:

    @patch("src.utils.hyde.SentenceTransformer")
    def test_expand_uses_generator_output(self, mock_st_cls):
        """expand() should use the generator_fn output as the hypothetical doc."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        mock_st_cls.return_value = mock_model

        hyp_doc = "This is a detailed hypothetical document about Corrective RAG."
        generator_fn = MagicMock(return_value=hyp_doc)

        from src.utils.hyde import HyDEExpander, HyDEConfig
        expander = HyDEExpander(generator_fn=generator_fn, config=HyDEConfig(enabled=True))
        result = expander.expand("What is CRAG?")

        assert result.used_hyde is True
        assert result.hypothetical_document == hyp_doc
        assert result.original_query == "What is CRAG?"

    @patch("src.utils.hyde.SentenceTransformer")
    def test_expand_falls_back_on_generator_failure(self, mock_st_cls):
        """If generator_fn raises, HyDE should fall back to raw query embedding."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        mock_st_cls.return_value = mock_model

        def failing_generator(q):
            raise RuntimeError("LLM unavailable")

        from src.utils.hyde import HyDEExpander
        expander = HyDEExpander(generator_fn=failing_generator)
        result = expander.expand("What is CRAG?")

        assert result.used_hyde is False

    @patch("src.utils.hyde.SentenceTransformer")
    def test_expand_disabled_config_skips_generator(self, mock_st_cls):
        """When HyDE is disabled, generator should never be called."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        mock_st_cls.return_value = mock_model

        generator_fn = MagicMock()

        from src.utils.hyde import HyDEExpander, HyDEConfig
        expander = HyDEExpander(
            generator_fn=generator_fn,
            config=HyDEConfig(enabled=False),
        )
        result = expander.expand("test query")
        generator_fn.assert_not_called()
        assert result.used_hyde is False


# =============================================================================
# corrective_rag.py integration tests (require live services)
# =============================================================================

@pytest.mark.integration
class TestCorrectiveRAGIntegration:
    """
    Full pipeline integration tests.
    Requires: Ollama running with llama3.2:3b, ChromaDB on localhost:8001.
    Run with: pytest tests/ -m integration -v
    """

    @pytest.fixture(scope="class")
    def crag(self):
        from src.corrective_rag import CorrectiveRAG, CRAGConfig
        from src.generator import GeneratorConfig
        config = CRAGConfig(
            collection_name="test_integration",
            generator_config=GeneratorConfig(model="llama3.2:3b"),
        )
        return CorrectiveRAG(config)

    def test_ingest_and_query_returns_answer(self, crag):
        """Full pipeline should produce a non-empty answer string."""
        crag.ingest({
            "id": "test_doc_1",
            "text": (
                "Corrective RAG (CRAG) is a retrieval-augmented generation technique "
                "that grades retrieved chunks for relevance and falls back to web search "
                "when the local retrieval quality is insufficient."
            )
        })
        result = crag.query("What is CRAG?")
        assert isinstance(result.answer, str)
        assert len(result.answer) > 10
        assert result.trace.correction_attempts >= 1

    def test_ingest_batch_reports_summary(self, crag):
        """ingest_batch should return a summary dict."""
        docs = [
            {"id": f"batch_{i}", "text": f"Document {i} with some content."}
            for i in range(3)
        ]
        summary = crag.ingest_batch(docs)
        assert "success" in summary
        assert "failed" in summary
        assert summary["success"] + summary["failed"] == len(docs)
