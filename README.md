# RAG System

A Python-based Retrieval-Augmented Generation (RAG) system that processes text and PDF documents, supports ChromaDB and Qdrant vector stores, and uses Sentence Transformers for semantic search and embeddings.

---

## Features

- Supports ChromaDB and Qdrant for efficient vector storage and retrieval.
- Multiple chunking strategies: fixed-length, sentence-based, paragraph-based, and hierarchical.
- Text embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- PDF text extraction with `pdfplumber`.
- Modular architecture for easy extension and customization.

---

## Project Structure

```
RAG/
├── src/
│   ├── __init__.py
│   ├── rag.py
│   └── utils/
│       ├── __init__.py
│       └── chunker.py
├── examples/
│   └── example_usage.py
├── tests/
│   └── test_rag.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
