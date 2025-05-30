# RAG

A Python implementation of a Retrieval-Augmented Generation (RAG) system with support for ChromaDB and Qdrant vector stores, multiple chunking strategies, and text embedding using Sentence Transformers.

## Features
- Supports ChromaDB and Qdrant vector stores for efficient document retrieval.
- Multiple chunking strategies: fixed-length, sentence-based, paragraph-based, and hierarchical.
- Text embedding using Sentence Transformers (`all-MiniLM-L6-v2`).
- PDF text extraction using `pdfplumber` for processing PDF documents.
- Modular design with extensible architecture for easy customization.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhruvisen/RAG.git
   cd RAG
