# Corrective RAG - Self-Correcting Retrieval-Augmented Generation

> **100% open-source | No API keys | Runs locally on CPU**

A production-grade RAG pipeline built on CRAG (Corrective RAG) research. It grades its own retrieved context, corrects bad retrievals via web fallback, and validates generated answers for hallucinations — all in a configurable, fully typed Python library.

---

## What Makes This Different

| Feature | Plain RAG | This Project |
|---|:---:|:---:|
| Vector similarity retrieval | Yes | Yes |
| Multiple vector stores (ChromaDB, Qdrant) | Yes | Yes |
| Hierarchical / sentence / paragraph chunking | Yes | Yes |
| LLM answer generation (Ollama, offline) | No | Yes |
| Self-correcting retrieval (CRAG) | No | Yes |
| Cross-encoder re-ranking | No | Yes |
| Hybrid BM25 + vector search (RRF) | No | Yes |
| HyDE query expansion | No | Yes |
| Retrieval quality grading | No | Yes |
| Answer hallucination grading | No | Yes |
| Web fallback (DuckDuckGo, no key) | No | Yes |
| PDF, TXT, MD, DOCX ingestion | No | Yes |

---

## Architecture

The pipeline runs in 6 sequential steps for every query:

1. **HyDE** - Generates a hypothetical answer paragraph and embeds it instead of the raw query. Improves retrieval recall on abstract or vague questions.

2. **Hybrid Search** - Combines BM25 keyword retrieval and vector semantic retrieval. Results are fused using Reciprocal Rank Fusion (RRF) for better overall ranking.

3. **Cross-Encoder Re-Ranking** - Re-scores all retrieved candidates by jointly encoding the query and each passage. Much more accurate than cosine similarity alone.

4. **Retrieval Grader** - Scores each chunk for relevance. If too few chunks pass the threshold, a DuckDuckGo web search is triggered as fallback to supplement context.

5. **LLM Generation** - Ollama runs qwen2.5:1.5b locally to generate a grounded, cited answer from the top-k chunks. No internet or API key required.

6. **Answer Grader + Self-Correction Loop** - The generated answer is scored for hallucinations. If unsupported, the pipeline retries with tighter context (up to 2 attempts).

---

## Project Structure

| File | Purpose |
|---|---|
| src/rag.py | Base vector store — ChromaDB and Qdrant backends |
| src/corrective_rag.py | Main CRAG pipeline — orchestrates all 6 steps |
| src/generator.py | Ollama LLM wrapper — streaming and non-streaming |
| src/grader.py | Retrieval grader and answer hallucination grader |
| src/reranker.py | Cross-encoder re-ranker |
| src/utils/chunker.py | Fixed / Sentence / Paragraph / Hierarchical chunking |
| src/utils/hybrid_search.py | BM25 + Vector search with RRF fusion |
| src/utils/hyde.py | HyDE query expansion |
| examples/demo.py | Interactive CLI demo using Rich |
| tests/test_corrective_rag.py | Unit tests (mocked) + integration tests |

---

## Prerequisites

### 1. Ollama (local LLM runtime)

Install Ollama from [ollama.com](https://ollama.com) or via the official install script.

Start the server with `ollama serve`, then pull the model:

- **Recommended (2-3s/query on CPU):** `ollama pull qwen2.5:1.5b` — 935MB
- **Faster but lighter:** `ollama pull qwen2.5:0.5b` — 397MB, ~1s/query

### 2. ChromaDB (vector store)

Install with `pip install chromadb`, then start with:

`chroma run --host localhost --port 8001`

---

## Installation

Clone the repository and install dependencies:

`pip install -r requirements.txt`

Download NLTK tokenizer data:

`python -c "import nltk; nltk.download('punkt')"`

---

## Usage

### Library

Import `CorrectiveRAG` and `CRAGConfig` from `src.corrective_rag`.

Create a config with your collection name and model, then call `ingest()` to load documents and `query()` to ask questions.

The `query()` method returns a `CRAGResult` object containing:
- `result.answer` — the final generated answer string
- `result.source_chunks` — list of chunks used as context
- `result.trace` — full pipeline trace including HyDE usage, web fallback status, correction attempts, answer grade score, and latency

### Interactive CLI Demo

Run `python -m examples.demo` with the following flags:

| Flag | Description |
|---|---|
| `--collection NAME` | Collection name to use (required) |
| `--file PATH ...` | One or more files to ingest at startup |
| `--model MODEL` | Ollama model name (default: qwen2.5:1.5b) |
| `--store chromadb/qdrant` | Vector store backend (default: chromadb) |
| `--no-hyde` | Disable HyDE query expansion |
| `--no-web` | Disable web fallback (fully offline mode) |
| `--verbose` | Enable debug logging |

### Batch Ingestion

`ingest_batch()` accepts a list of document dicts and returns a summary with success count, failed count, and per-document errors. Supported formats: PDF (via `pdf_path` key), plain text (via `text` key), and DOCX.

---

## Configuration

All behaviour is controlled via `CRAGConfig` (in `src/corrective_rag.py`). Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `collection_name` | crag_collection | Vector store collection name |
| `initial_top_k` | 8 | Candidates fetched before re-ranking |
| `final_top_k` | 4 | Chunks passed to the LLM |
| `max_correction_attempts` | 2 | Self-correction retry limit |
| `web_search_max_results` | 4 | DuckDuckGo results on fallback |
| `enable_hyde` | True | Toggle HyDE query expansion |
| `enable_hybrid_search` | True | Toggle BM25 + vector fusion |
| `enable_reranking` | True | Toggle cross-encoder re-ranking |
| `enable_web_fallback` | True | Toggle web search fallback |
| `enable_answer_grading` | True | Toggle hallucination checking |
| `chunk_size` | 400 | Characters per chunk |

Grading thresholds are controlled via `RetrievalGraderConfig` and `AnswerGraderConfig` in `src/grader.py`.

---

## Running Tests

Unit tests (no services required — all dependencies are mocked):

`pytest tests/ -m "not integration" -v`

Integration tests (requires Ollama and ChromaDB to be running):

`pytest tests/ -m integration -v`

---

## Open-Source Stack

| Component | Library | Details |
|---|---|---|
| Vector store | ChromaDB / Qdrant | Persistent, production-ready |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2, runs offline |
| LLM | Ollama + qwen2.5:1.5b | CPU-optimised, GQA architecture, ~935MB |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Best-in-class passage ranking |
| Keyword search | rank-bm25 | Classic BM25Okapi implementation |
| Web fallback | duckduckgo-search | No API key required |
| PDF parsing | pdfplumber | Accurate table and text extraction |
| CLI | rich | Terminal tables, progress bars, panels |

---

## References

- [Corrective RAG (Shi et al., 2024)](https://arxiv.org/abs/2401.15884)
- [HyDE (Gao et al., 2022)](https://arxiv.org/abs/2212.10496)
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [Cross-Encoders for Re-ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)

---

## License

MIT - see [LICENSE](LICENSE).
