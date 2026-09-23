"""
Corrective RAG (CRAG) — Self-correcting retrieval-augmented generation pipeline.

Reference: Shi et al., "Corrective Retrieval Augmented Generation" (2024)
           https://arxiv.org/abs/2401.15884

Pipeline overview:

  Query
    ↓
  [HyDE]           — Expand query into hypothetical document for better retrieval
    ↓
  [Hybrid Search]  — BM25 + Vector retrieval with RRF fusion
    ↓
  [Re-Ranking]     — Cross-encoder re-ranks top candidates
    ↓
  [Retrieval Grade] — Score each chunk; flag irrelevant ones
    ↓ (if retrieval quality is below threshold)
  [Web Fallback]   — DuckDuckGo search supplements context (open-source, no key)
    ↓
  [Generation]     — Ollama LLM generates answer from graded context
    ↓
  [Answer Grade]   — Check for hallucinations; retry loop if unsupported
    ↓
  Final Answer

The pipeline is fully configurable via ``CRAGConfig`` and raises typed exceptions
rather than returning None or silently swallowing errors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ddgs import DDGS

from .rag import RAG, VectorStoreType, EmbeddingModel
from .generator import OllamaGenerator, GeneratorConfig
from .grader import RetrievalGrader, AnswerGrader, RetrievalGraderConfig, AnswerGraderConfig
from .reranker import CrossEncoderReranker, RerankerConfig
from .utils.hybrid_search import HybridSearcher, HybridSearchConfig
from .utils.hyde import HyDEExpander, HyDEConfig
from .utils.chunker import ChunkingStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CRAGConfig:
    """
    Top-level configuration for the full CRAG pipeline.

    All sub-component configs can be overridden individually.

    CPU-optimised defaults (no GPU required):
      - LLM: qwen2.5:1.5b via Ollama (~935MB, ~2-3s/query on CPU)
      - Graders: cross-encoder/ms-marco-MiniLM-L-6-v2 (runs fast on CPU)
      - initial_top_k=8: balanced recall without over-fetching
      - final_top_k=4: fewer chunks = shorter prompt = faster generation
    """
    # --- Core RAG settings ---
    vector_store_type: VectorStoreType = VectorStoreType.CHROMADB
    embedding_model: EmbeddingModel = EmbeddingModel.MINI_LM_L6_V2
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HIERARCHICAL
    chunk_size: int = 400           # smaller chunks = more precise retrieval
    overlap: int = 40
    collection_name: str = "crag_collection"

    # --- Retrieval settings ---
    initial_top_k: int = 8           # fetch candidates for re-ranking (CPU-tuned)
    final_top_k: int = 4             # chunks passed to generator (shorter = faster)

    # --- Self-correction settings ---
    max_correction_attempts: int = 2  # 2 rounds sufficient on CPU for speed
    web_search_max_results: int = 4   # fewer web results = faster fallback

    # --- Sub-component configs (defaults tuned for CPU) ---
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)  # qwen2.5:1.5b
    retrieval_grader_config: RetrievalGraderConfig = field(default_factory=RetrievalGraderConfig)
    answer_grader_config: AnswerGraderConfig = field(default_factory=AnswerGraderConfig)
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)
    hybrid_search_config: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    hyde_config: HyDEConfig = field(default_factory=HyDEConfig)

    # --- Vector store connection ---
    vector_store_host: str = "localhost"
    vector_store_port: Optional[int] = None   # None = use store default

    # --- Feature flags ---
    enable_hyde: bool = True
    enable_hybrid_search: bool = True
    enable_reranking: bool = True
    enable_web_fallback: bool = True
    enable_answer_grading: bool = True


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    """Step-by-step trace of a CRAG pipeline execution (for debugging/logging)."""
    query: str
    hyde_used: bool = False
    hypothetical_document: Optional[str] = None
    vector_chunks_retrieved: int = 0
    hybrid_fusion_applied: bool = False
    reranking_applied: bool = False
    retrieval_grades: List[Dict[str, Any]] = field(default_factory=list)
    web_fallback_triggered: bool = False
    web_snippets_fetched: int = 0
    correction_attempts: int = 0
    answer_grade_score: float = 0.0
    answer_is_supported: bool = False
    latency_ms: float = 0.0


@dataclass
class CRAGResult:
    """Final result from the CRAG pipeline."""
    query: str
    answer: str
    source_chunks: List[Dict[str, Any]]
    trace: PipelineTrace


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CRAGError(RuntimeError):
    """Base exception for CRAG pipeline errors."""


class CRAGStorageError(CRAGError):
    """Raised when document ingestion fails."""


class CRAGQueryError(CRAGError):
    """Raised when a query cannot be processed."""


# ---------------------------------------------------------------------------
# CRAG Pipeline
# ---------------------------------------------------------------------------

class CorrectiveRAG:
    """
    Full Corrective RAG pipeline.

    All components are initialised lazily-at-construction and reused across
    multiple ``query()`` calls — initialise once, query many times.

    Usage::

        config = CRAGConfig(collection_name="my_docs")
        crag = CorrectiveRAG(config)

        # Ingest documents
        crag.ingest({"id": "doc1", "pdf_path": "/path/to/file.pdf"})

        # Query
        result = crag.query("What is self-correcting RAG?")
        print(result.answer)
        print(result.trace)
    """

    def __init__(self, config: CRAGConfig | None = None) -> None:
        self.config = config or CRAGConfig()
        logger.info("Initialising CorrectiveRAG pipeline...")
        self._init_components()
        logger.info("CorrectiveRAG pipeline ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, document: Dict[str, Any]) -> None:
        """
        Ingest a single document into the vector store.

        Args:
            document: Dict containing ``"id"`` and either ``"text"`` or
                      ``"pdf_path"`` (for PDF ingestion). Any additional
                      keys are stored as metadata.

        Raises:
            CRAGStorageError: If ingestion fails.
        """
        doc_id = document.get("id", "unknown")
        logger.info("Ingesting document id='%s'", doc_id)
        try:
            self._rag.store_document(document)
            # Also cache chunk text for BM25 indexing
            self._corpus_cache.clear()  # invalidate cache on new document
            logger.info("Document '%s' ingested successfully.", doc_id)
        except Exception as exc:
            raise CRAGStorageError(f"Failed to ingest document '{doc_id}': {exc}") from exc

    def ingest_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest multiple documents, collecting any per-document errors.

        Args:
            documents: List of document dicts.

        Returns:
            Summary dict: ``{"success": int, "failed": int, "errors": list}``.
        """
        summary: Dict[str, Any] = {"success": 0, "failed": 0, "errors": []}
        for doc in documents:
            try:
                self.ingest(doc)
                summary["success"] += 1
            except CRAGStorageError as exc:
                summary["failed"] += 1
                summary["errors"].append({"id": doc.get("id"), "error": str(exc)})
                logger.error("Batch ingest error for doc '%s': %s", doc.get("id"), exc)
        logger.info(
            "Batch ingest complete: %d success, %d failed.",
            summary["success"], summary["failed"],
        )
        return summary

    def query(self, question: str) -> CRAGResult:
        """
        Run the full CRAG pipeline for a user question.

        The pipeline:
          1. HyDE query expansion (optional)
          2. Hybrid BM25 + vector retrieval (optional)
          3. Cross-encoder re-ranking (optional)
          4. Retrieval grading + web fallback if needed
          5. LLM answer generation
          6. Answer grading + self-correction loop

        Args:
            question: User's natural language question.

        Returns:
            ``CRAGResult`` with the final answer, source chunks, and execution trace.

        Raises:
            CRAGQueryError: If the pipeline cannot produce any result.
        """
        start_time = time.perf_counter()
        trace = PipelineTrace(query=question)
        logger.info("CRAG query: '%s'", question)

        # Dynamic LLM-based intent routing: Does this query need RAG?
        logger.info("==================================================")
        logger.info("[STEP 1/6] Running intent classification...")
        needs_retrieval = self._check_intent(question)
        
        if not needs_retrieval:
            logger.info("[STEP 1/6] Result: NO RETRIEVAL NEEDED. Route -> Casual Conversation.")
            casual_prompt = f"System: You are a helpful AI assistant. Answer the user's casual greeting or small talk naturally.\nUser: {question}\nAssistant:"
            answer = self._generator.generate(casual_prompt, [])
            trace.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return CRAGResult(query=question, answer=answer, source_chunks=[], trace=trace)
        
        logger.info("[STEP 1/6] Result: RETRIEVAL REQUIRED. Route -> RAG Pipeline.")

        try:
            result = self._run_pipeline(question, trace)
        except Exception as exc:
            raise CRAGQueryError(f"Pipeline failed for query '{question}': {exc}") from exc
        finally:
            trace.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info("CRAG pipeline completed in %.0f ms.", trace.latency_ms)

        return result

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    def _run_pipeline(self, question: str, trace: PipelineTrace) -> CRAGResult:
        """Orchestrate the full pipeline and return the final CRAGResult."""

        retrieval_query = question

        # Step 1: Retrieve from vector store
        logger.info("[STEP 2/6] Retrieving from Vector Store...")
        vector_chunks = self._retrieve_vector(retrieval_query)
        trace.vector_chunks_retrieved = len(vector_chunks)
        logger.info("[STEP 2/6] Result: Retrieved %d chunks.", len(vector_chunks))
        for i, chunk in enumerate(vector_chunks, 1):
            preview = chunk.get("text", "")[:150].replace("\n", " ")
            score = chunk.get("distance", chunk.get("rerank_score", "N/A"))
            logger.info("  Chunk %d [%s] (score: %s): %s...", i, chunk.get("chunk_id", "?"), score, preview)

        # Step 2: Hybrid search (BM25 + vector fusion)
        if self.config.enable_hybrid_search and self._hybrid_searcher and self._corpus_cache:
            try:
                hybrid_chunks = self._hybrid_searcher.search_to_dicts(
                    query=question,
                    vector_results=vector_chunks,
                    corpus_chunks=self._get_corpus(),
                    top_k=self.config.initial_top_k,
                )
                # Convert HybridResult dicts back to standard chunk dicts
                vector_chunks = self._normalize_hybrid_chunks(hybrid_chunks, vector_chunks)
                trace.hybrid_fusion_applied = True
                logger.debug("Hybrid search fused %d candidates.", len(vector_chunks))
            except Exception as exc:
                logger.warning("Hybrid search failed (%s); continuing with vector-only.", exc)

        # Step 3: Re-rank
        logger.info("[STEP 3/6] Re-ranking candidates...")
        if self.config.enable_reranking and self._reranker:
            vector_chunks = self._reranker.rerank_to_dicts(
                query=question,
                chunks=vector_chunks,
                top_k=self.config.final_top_k,
            )
            trace.reranking_applied = True
            logger.info("[STEP 3/6] Result: Re-ranked top %d chunks.", self.config.final_top_k)
        else:
            logger.info("[STEP 3/6] Result: Re-ranking disabled or reranker not initialized.")

        working_chunks = vector_chunks[: self.config.final_top_k]
        logger.info("  ── Final context chunks ──")
        for i, chunk in enumerate(working_chunks, 1):
            preview = chunk.get("text", "")[:150].replace("\n", " ")
            score = chunk.get("rerank_score", chunk.get("distance", "N/A"))
            logger.info("  Chunk %d [%s] (score: %s): %s...", i, chunk.get("chunk_id", "?"), score, preview)

        # Step 4: Grade retrieved chunks
        logger.info("[STEP 4/6] Grading retrieved chunks & checking web fallback...")
        grades = self._retrieval_grader.grade(question, working_chunks)
        acceptable_grades, needs_fallback = self._retrieval_grader.filter_relevant(grades)
        trace.retrieval_grades = [
            {
                "chunk_id": g.chunk_id,
                "score": g.score,
                "label": g.label.value,
                "preview": g.text_preview,
            }
            for g in grades
        ]

        # Web fallback if retrieval quality is low
        if needs_fallback and self.config.enable_web_fallback:
            logger.info("[STEP 4/6] Result: Retrieval quality insufficient. Triggering web fallback.")
            web_chunks = self._web_search_fallback(question)
            trace.web_fallback_triggered = True
            trace.web_snippets_fetched = len(web_chunks)
            # Augment with any acceptable local chunks
            acceptable_chunk_ids = {g.chunk_id for g in acceptable_grades}
            local_good = [c for c in working_chunks if c["chunk_id"] in acceptable_chunk_ids]
            working_chunks = local_good + web_chunks
            logger.info(
                "[STEP 4/6] Context: %d local + %d web chunks after fallback.",
                len(local_good), len(web_chunks),
            )
        elif not needs_fallback:
            logger.info("[STEP 4/6] Result: Retrieval quality acceptable. No fallback needed.")
            # Keep only acceptable chunks
            acceptable_ids = {g.chunk_id for g in acceptable_grades}
            working_chunks = [c for c in working_chunks if c["chunk_id"] in acceptable_ids] \
                             or working_chunks  # safety: fall back to all if filter empties

        # Step 5: Self-correction loop — generate and grade the answer
        logger.info("[STEP 5/6] Generating answer with LLM...")
        answer = ""
        answer_grade_score = 0.0
        answer_is_supported = False

        for attempt in range(1, self.config.max_correction_attempts + 1):
            trace.correction_attempts = attempt
            logger.info("[STEP 5/6] Generation attempt %d/%d...", attempt, self.config.max_correction_attempts)

            answer = self._generator.generate(question, working_chunks)
            logger.debug("Generated answer (attempt %d): %s...", attempt, answer[:100])

            if not self.config.enable_answer_grading:
                answer_is_supported = True
                break

            answer_grade = self._answer_grader.grade(question, answer, working_chunks)
            answer_grade_score = answer_grade.score
            answer_is_supported = answer_grade.is_supported

            logger.info(
                "Answer grade (attempt %d): score=%.4f supported=%s — %s",
                attempt, answer_grade.score, answer_grade.is_supported, answer_grade.reason,
            )

            if answer_is_supported:
                break

            if attempt < self.config.max_correction_attempts:
                logger.info("Answer not supported — refining context for retry.")
                # On retry, filter to the highest-scored local chunks
                working_chunks = self._refine_context_for_retry(
                    question, working_chunks, attempt
                )

        trace.answer_grade_score = answer_grade_score
        trace.answer_is_supported = answer_is_supported

        return CRAGResult(
            query=question,
            answer=answer,
            source_chunks=working_chunks,
            trace=trace,
        )

    def _retrieve_vector(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve from the vector store."""
        logger.info("[STEP 2/6] Result: Using raw text query for vector search.")
        return self._rag.retrieve_documents(query, top_k=self.config.initial_top_k)

    def _retrieve_with_embedding(self, embedding: List[float]) -> List[Dict[str, Any]]:
        """Retrieve using a pre-computed embedding vector (for HyDE)."""
        store = self._rag.vector_store
        # Both ChromaDB and Qdrant stores accept raw embedding queries
        if hasattr(store, "collection"):
            # ChromaDB path
            results = store.collection.query(
                query_embeddings=[embedding],
                n_results=self.config.initial_top_k,
            )
            retrieved = []
            for i in range(len(results["ids"][0])):
                retrieved.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
            return retrieved
        elif hasattr(store, "client"):
            # Qdrant path
            search_result = store.client.search(
                collection_name=store.collection_name,
                query_vector=embedding,
                limit=self.config.initial_top_k,
                with_payload=True,
            )
            retrieved = []
            for point in search_result:
                metadata = {k: v for k, v in point.payload.items() if k != "text"}
                retrieved.append({
                    "chunk_id": point.payload.get("chunk_id", str(point.id)),
                    "text": point.payload.get("text", ""),
                    "metadata": metadata,
                    "distance": point.score,
                })
            return retrieved
        else:
            raise NotImplementedError("Unknown vector store type for direct embedding retrieval.")

    def _check_intent(self, query: str) -> bool:
        """Use the LLM to classify if the query requires knowledge base retrieval."""
        prompt = (
            "You are a strict intent router for a document retrieval system.\n"
            "Task: Classify if the user input requires searching documents for facts/context.\n\n"
            "Output 'YES' if the user is asking a question, requesting information, or referring to any documents, facts, or data.\n"
            "Output 'NO' ONLY if the user is making a casual greeting or social small talk (like 'hello' or 'thanks').\n\n"
            f"User input: '{query}'\n"
            "Classification (YES or NO):"
        )
        logger.debug("Intent check Prompt: %s", prompt)
        try:
            import requests
            url = f"{self.config.generator_config.base_url}/api/generate"
            payload = {
                "model": self.config.generator_config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10}
            }
            logger.info("[Intent Classifier] Requesting classification from LLM...")
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                result_text = resp.json().get("response", "").strip().upper()
                requires_search = "YES" in result_text or "NO" not in result_text
                logger.info("[Intent Classifier] LLM responded with: '%s' -> Requires Search: %s", result_text, requires_search)
                return requires_search
            else:
                logger.warning("[Intent Classifier] LLM returned status %s", resp.status_code)
        except Exception as exc:
            logger.warning("[Intent Classifier] Request failed: %s. Defaulting to retrieval=True", exc)
        return True

    def _web_search_fallback(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch DuckDuckGo search results and format them as chunk dicts.
        DuckDuckGo requires no API key and is open-source.
        """
        logger.info("Web fallback: searching DuckDuckGo for '%s'", query)
        web_chunks: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=self.config.web_search_max_results,
                ))
            for i, r in enumerate(results):
                snippet = f"{r.get('title', '')}\n{r.get('body', '')}"
                web_chunks.append({
                    "chunk_id": f"web_{i}",
                    "text": snippet.strip(),
                    "metadata": {
                        "source": "web",
                        "url": r.get("href", ""),
                        "title": r.get("title", ""),
                    },
                })
            logger.info("Web fallback fetched %d snippets.", len(web_chunks))
        except Exception as exc:
            logger.warning("Web fallback failed (%s) — proceeding without web context.", exc)
        return web_chunks

    def _refine_context_for_retry(
        self,
        query: str,
        current_chunks: List[Dict[str, Any]],
        attempt: int,
    ) -> List[Dict[str, Any]]:
        """
        On a correction retry, re-rank current context more aggressively
        and reduce chunk count to force the LLM to be more precise.
        """
        tighter_top_k = max(2, self.config.final_top_k - attempt)
        if self._reranker:
            return self._reranker.rerank_to_dicts(query, current_chunks, top_k=tighter_top_k)
        return current_chunks[:tighter_top_k]

    def _normalize_hybrid_chunks(
        self,
        hybrid_dicts: List[Dict[str, Any]],
        original_vector_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Map hybrid results back to the full chunk dicts from the vector store
        (since hybrid search results may have truncated metadata).
        """
        vector_map = {c["chunk_id"]: c for c in original_vector_chunks}
        result = []
        for hc in hybrid_dicts:
            cid = hc["chunk_id"]
            base = vector_map.get(cid, hc)
            merged = {**base, "rrf_score": hc.get("rrf_score"), "bm25_rank": hc.get("bm25_rank")}
            result.append(merged)
        return result

    def _get_corpus(self) -> List[Dict[str, Any]]:
        """
        Return cached corpus chunks for BM25 indexing.
        Populated lazily on first retrieval.
        """
        if not self._corpus_cache:
            # Fetch a broad set of chunks from the vector store using a generic query
            try:
                self._corpus_cache = self._rag.retrieve_documents("", top_k=500)
            except Exception as exc:
                logger.warning("Could not fetch corpus for BM25 (%s).", exc)
        return self._corpus_cache

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        """Initialise all pipeline components, logging any non-fatal failures."""
        cfg = self.config

        # Core RAG (vector store + chunker)
        extra_kwargs: Dict[str, Any] = {
            "collection_name": cfg.collection_name,
            "host": cfg.vector_store_host,
        }
        if cfg.vector_store_port:
            extra_kwargs["port"] = cfg.vector_store_port

        self._rag = RAG(
            vector_store_type=cfg.vector_store_type,
            embedding_model=cfg.embedding_model,
            chunking_strategy=cfg.chunking_strategy,
            chunk_size=cfg.chunk_size,
            overlap=cfg.overlap,
            **extra_kwargs,
        )

        # Generator (Ollama LLM)
        self._generator = OllamaGenerator(cfg.generator_config)

        # Graders
        self._retrieval_grader = RetrievalGrader(cfg.retrieval_grader_config)
        self._answer_grader = AnswerGrader(cfg.answer_grader_config)

        # Re-ranker (optional)
        self._reranker: Optional[CrossEncoderReranker] = None
        if cfg.enable_reranking:
            try:
                self._reranker = CrossEncoderReranker(cfg.reranker_config)
            except Exception as exc:
                logger.warning("Re-ranker init failed (%s); disabling re-ranking.", exc)

        # Hybrid searcher (optional)
        self._hybrid_searcher: Optional[HybridSearcher] = None
        if cfg.enable_hybrid_search:
            self._hybrid_searcher = HybridSearcher(cfg.hybrid_search_config)

        # HyDE (optional)
        self._hyde: Optional[HyDEExpander] = None
        if cfg.enable_hyde:
            hyde_config = HyDEConfig(enabled=True)
            try:
                self._hyde = HyDEExpander(
                    generator_fn=self._generator.generate_hypothetical_document,
                    config=hyde_config,
                )
            except Exception as exc:
                logger.warning("HyDE init failed (%s); disabling HyDE.", exc)

        # BM25 corpus cache
        self._corpus_cache: List[Dict[str, Any]] = []
