# RAG

A Python implementation of a Retrieval-Augmented Generation (RAG) system with support for ChromaDB and Qdrant vector stores, multiple chunking strategies, and text embedding using Sentence Transformers.

## Features
- Supports ChromaDB and Qdrant vector stores for efficient document retrieval.
- Multiple chunking strategies: fixed-length, sentence-based, paragraph-based, and hierarchical.
- Text embedding using Sentence Transformers (\`all-MiniLM-L6-v2\`).
- PDF text extraction using \`pdfplumber\` for processing PDF documents.
- Modular design with extensible architecture for easy customization.

## Installation
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/Dhruvisen/RAG.git
   cd RAG
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
3. Ensure ChromaDB or Qdrant server is running:
   - For ChromaDB: Run the server (default: \`localhost:8001\`).
     \`\`\`bash
     docker run -p 8001:8000 chromadb/chroma
     \`\`\`
   - For Qdrant: Run the server (default: \`localhost:6333\`).
     \`\`\`bash
     docker run -p 6333:6333 qdrant/qdrant
     \`\`\`

\`\`\`python
from src.rag import RAG, VectorStoreType, EmbeddingModel
from src.utils.chunker import ChunkingStrategy

# Initialize RAG with ChromaDB
rag = RAG(
    vector_store_type=VectorStoreType.CHROMADB,
    embedding_model=EmbeddingModel.MINI_LM_L6_V2,
    chunking_strategy=ChunkingStrategy.PARAGRAPH,
    collection_name="test_collection",
    chunk_size=50,
    overlap=10,
    host="localhost",
    port=8001
)

# Store a document
document = {"id": "1", "text": "This is a test document.", "category": "test"}
rag.store_document(document)

# Retrieve documents
results = rag.retrieve_documents("test document", top_k=2)
for result in results:
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Distance: {result['distance']}")
\`\`\`

## Project Structure
\`\`\`
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
\`\`\`

## Requirements
The project dependencies are listed in \`requirements.txt\`:
- \`chromadb>=0.4.0\`
- \`qdrant-client>=1.7.0\`
- \`sentence-transformers>=2.2.2\`
- \`pdfplumber>=0.10.0\`
- \`nltk>=3.8.1\`
- \`pytest>=7.4.0\`

Install them using:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Testing
Run tests using \`pytest\`:
\`\`\`bash
pytest tests/
\`\`\`

The \`tests/test_rag.py\` file includes basic tests for RAG initialization and document storage/retrieval.

## License
This project is licensed under the MIT License - see the \`LICENSE\` file for details.
EOF
