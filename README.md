# Corrective RAG - Self-Correcting Retrieval-Augmented Generation

A production-grade RAG pipeline built on CRAG (Corrective RAG) research. It grades its own retrieved context, corrects bad retrievals via web fallback, and validates generated answers for hallucinations. It is a full-stack application with a React frontend and FastAPI backend.

## Tech Stack & Architecture Flow

This project is a full-stack web application designed for processing documents and answering questions locally:

- **Frontend:** React (Vite)
- **Backend API:** FastAPI (Python)
- **Database:** PostgreSQL (SQLAlchemy)
- **Vector Store:** ChromaDB (for fast similarity search)
- **Local LLM:** Ollama (running Qwen 2.5 7B model)
- **Package Manager:** `uv`

**How it flows:**
1. A user uploads a document (PDF, TXT, etc.) via the React frontend.
2. The FastAPI backend extracts text from the document, splits it into smaller chunks, and embeds them into ChromaDB.
3. The user asks a question via the chat interface.
4. The system searches ChromaDB for the most relevant document chunks.
5. A cross-encoder model reranks and grades the chunks for relevance, keeping only the best ones.
6. The chunks are passed to the Ollama LLM (`qwen2.5:7b`), which generates a concise answer completely offline.

## Prerequisites

1. **Ollama (local LLM runtime)**
   Install Ollama from [ollama.com](https://ollama.com).
   Start the server with `ollama serve`, then pull the model:
   `ollama pull qwen2.5:7b`

2. **uv (Python Package Manager)**
   Install `uv` to manage Python dependencies.

3. **Node.js**
   Required to run the React frontend.

## How to Run

### 1. Start the Backend Server
Navigate to the root directory and start the FastAPI backend using `uv`:
```bash
uv run main.py
```
*The backend will automatically start on http://localhost:8000.*

### 2. Start the Frontend Server
Open a new terminal, navigate to the `frontend` directory, and start the React app:
```bash
cd frontend
npm install
npm run dev
```
*The frontend will be available at http://localhost:5174 (or similar depending on Vite). You can access the UI to upload documents and ask questions!*
