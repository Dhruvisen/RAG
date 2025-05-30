from src.rag import RAG, VectorStoreType, EmbeddingModel
from src.utils.chunker import ChunkingStrategy

def main():
    # Initialize RAG with ChromaDB
    rag_chroma = RAG(
        vector_store_type=VectorStoreType.CHROMADB,
        embedding_model=EmbeddingModel.MINI_LM_L6_V2,
        chunking_strategy=ChunkingStrategy.PARAGRAPH,
        collection_name="test_collection",
        chunk_size=50,
        overlap=10,
        host="localhost",
        port=8001
    )

    # Sample documents
    documents = [
        {"id": "1", "text": "This is document 1. It has multiple sentences. Another sentence here.", "category": "test"},
        {"id": "2", "text": "This is document 2. It contains different information.", "category": "test"}
    ]

    # Store documents
    for doc in documents:
        rag_chroma.store_document(doc)

    # Retrieve documents
    results = rag_chroma.retrieve_documents("document information", top_k=2)
    print("ChromaDB Results:")
    for result in results:
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']}")
        print("-" * 50)

if __name__ == "__main__":
    main()