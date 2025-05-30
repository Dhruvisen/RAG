import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from enum import Enum
import pdfplumber
import os
from .utils.chunker import TextChunker, ChunkingStrategy

class VectorStoreType(Enum):
    CHROMADB = "chromadb"
    QDRANT = "qdrant"

class EmbeddingModel(Enum):
    MINI_LM_L6_V2 = "all-MiniLM-L6-v2"

class VectorStore:
    """Base class for vector store operations using cosine similarity"""
    
    def __init__(self, collection_name: str, embedding_model: EmbeddingModel, 
                 chunking_strategy: ChunkingStrategy, chunk_size: int = 500, overlap: int = 50,
                 host: Optional[str] = None, port: Optional[int] = None, 
                 id: Optional[str] = None, password: Optional[str] = None):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model.value)
        self.chunker = TextChunker(chunking_strategy, chunk_size, overlap)
        self.host = host
        self.port = port
        self.id = id
        self.password = password
    
    def store(self, document: Dict[str, Any]) -> None:
        """Store a single document and its embedding"""
        raise NotImplementedError
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on query text using cosine similarity"""
        raise NotImplementedError
    
    def _extract_text(self, document: Dict[str, Any]) -> str:
        """Extract text from document, handling PDFs if present"""
        if "pdf_path" in document and os.path.exists(document["pdf_path"]):
            with pdfplumber.open(document["pdf_path"]) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            return text
        return document.get("text", "")

class ChromaDBStore(VectorStore):
    """ChromaDB implementation of vector store using cosine similarity"""
    
    def __init__(self, collection_name: str, embedding_model: EmbeddingModel, 
                 chunking_strategy: ChunkingStrategy, chunk_size: int = 500, overlap: int = 50,
                 host: Optional[str] = None, port: Optional[int] = None, 
                 id: Optional[str] = None, password: Optional[str] = None):
        super().__init__(collection_name, embedding_model, chunking_strategy, chunk_size, overlap,
                        host, port, id, password)
        host = host or "localhost"
        port = port or 8001
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def store(self, document: Dict[str, Any]) -> None:
        text = self._extract_text(document)
        document_id = document.get("id", "0")
        chunks = self.chunker.chunk_text(text, document_id)
        metadata = {k: v for k, v in document.items() if k not in ["text", "pdf_path"]}
        
        for chunk in chunks:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "level": chunk.get("level", "chunk")
            })
            
            self.collection.add(
                ids=[chunk["chunk_id"]],
                embeddings=[chunk["embedding"]],  
                metadatas=[chunk_metadata],
                documents=[chunk["text"]]
            )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        return retrieved

class QdrantStore(VectorStore):
    """Qdrant implementation of vector store using cosine similarity"""
    
    def __init__(self, collection_name: str, embedding_model: EmbeddingModel, 
                 chunking_strategy: ChunkingStrategy, chunk_size: int = 500, overlap: int = 50,
                 host: Optional[str] = None, port: Optional[int] = None, 
                 id: Optional[str] = None, password: Optional[str] = None):
        super().__init__(collection_name, embedding_model, chunking_strategy, chunk_size, overlap,
                        host, port, id, password)
        host = host or "localhost"
        port = port or 6333
        self.client = QdrantClient(host=host, port=port, api_key=password)
        self._init_collection()
    
    def _init_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except Exception as e:
            if "Connection refused" in str(e):
                raise ConnectionError(
                    f"Failed to connect to Qdrant server at {self.host}:{self.port}. "
                    "Ensure the Qdrant server is running and accessible."
                )
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            except Exception as create_e:
                raise RuntimeError(f"Failed to create Qdrant collection: {str(create_e)}")
    
    def store(self, document: Dict[str, Any]) -> None:
        text = self._extract_text(document)
        document_id = document.get("id", "0")
        chunks = self.chunker.chunk_text(text, document_id)
        metadata = {k: v for k, v in document.items() if k not in ["text", "pdf_path"]}
        
        for chunk in chunks:
            point_id = int(chunk["chunk_id"].split("_")[-1], 16)
            payload = metadata.copy()
            payload.update({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "level": chunk.get("level", "chunk")
            })
            
            point = PointStruct(
                id=point_id,
                vector=chunk["embedding"],  
                payload=payload
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        retrieved = []
        for point in search_result:
            metadata = {k: v for k, v in point.payload.items() if k != "text"}
            if "id" in metadata and isinstance(metadata["id"], str) and metadata["id"].isdigit():
                metadata["id"] = int(metadata["id"])
            retrieved.append({
                "chunk_id": point.payload.get("chunk_id", str(point.id)),
                "text": point.payload.get("text", ""),
                "metadata": metadata,
                "distance": point.score
            })
        return retrieved

class RAG:
    """Generic RAG implementation using cosine similarity"""
    
    def __init__(self, vector_store_type: VectorStoreType, embedding_model: EmbeddingModel, 
                 chunking_strategy: ChunkingStrategy, chunk_size: int = 500, overlap: int = 50, **kwargs):
        self.vector_store = self._initialize_vector_store(
            vector_store_type, embedding_model, chunking_strategy, chunk_size, overlap, **kwargs
        )
    
    def _initialize_vector_store(self, store_type: VectorStoreType, embedding_model: EmbeddingModel, 
                                 chunking_strategy: ChunkingStrategy, chunk_size: int, overlap: int, 
                                 **kwargs) -> VectorStore:
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("collection_name", None)
        if store_type == VectorStoreType.CHROMADB:
            return ChromaDBStore(
                collection_name=kwargs.get("collection_name"), 
                embedding_model=embedding_model, 
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                **kwargs_copy
            )
        elif store_type == VectorStoreType.QDRANT:
            return QdrantStore(
                collection_name=kwargs.get("collection_name"), 
                embedding_model=embedding_model, 
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                **kwargs_copy
            )
        else:
            raise ValueError(f"Unsupported vector store: {store_type}")
    
    def store_document(self, document: Dict[str, Any]) -> None:
        """Store a single document"""
        self.vector_store.store(document)
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query using cosine similarity"""
        return self.vector_store.retrieve(query, top_k)