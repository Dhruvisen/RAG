import uuid
import nltk
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from enum import Enum

nltk.download('punkt')

class ChunkingStrategy(Enum):
    FIXED_LENGTH = "fixed_length"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HIERARCHICAL = "hierarchical"

class TextChunker:
    """Utility class for different text chunking strategies with embeddings"""
    
    def __init__(self, strategy: ChunkingStrategy, chunk_size: int = 500, overlap: int = 50):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk_text(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Chunk text based on the specified strategy and generate embeddings"""
        if self.strategy == ChunkingStrategy.FIXED_LENGTH:
            return self._fixed_length_chunking(text, document_id)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(text, document_id)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(text, document_id)
        elif self.strategy == ChunkingStrategy.HIERARCHICAL:
            return self._hierarchical_chunking(text, document_id)
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.strategy}")
    
    def _fixed_length_chunking(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Split text into fixed-length chunks with overlap and generate embeddings"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text.strip():
                embedding = self.embedding_model.encode([chunk_text]).tolist()[0]
                chunks.append({
                    "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                    "text": chunk_text,
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "embedding": embedding
                })
        return chunks
    
    def _sentence_chunking(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Split text into sentence-based chunks and generate embeddings"""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                embedding = self.embedding_model.encode([chunk_text]).tolist()[0]
                chunks.append({
                    "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                    "text": chunk_text,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "embedding": embedding
                })
                current_chunk = [sentence]
                current_length = sentence_length
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            embedding = self.embedding_model.encode([chunk_text]).tolist()[0]
            chunks.append({
                "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                "text": chunk_text,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "embedding": embedding
            })
        return chunks
    
    def _paragraph_chunking(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Split text into paragraph-based chunks and generate embeddings"""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            if current_length + paragraph_length > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                embedding = self.embedding_model.encode([chunk_text]).tolist()[0]
                chunks.append({
                    "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                    "text": chunk_text,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "embedding": embedding
                })
                current_chunk = [paragraph]
                current_length = paragraph_length
                chunk_index += 1
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            embedding = self.embedding_model.encode([chunk_text]).tolist()[0]
            chunks.append({
                "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                "text": chunk_text,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "embedding": embedding
            })
        return chunks
    
    def _hierarchical_chunking(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Hierarchical chunking: paragraphs, then sentences if needed, with embeddings"""
        paragraphs = text.split("\n\n")
        chunks = []
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                embedding = self.embedding_model.encode([paragraph]).tolist()[0]
                chunks.append({
                    "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                    "text": paragraph,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "level": "paragraph",
                    "embedding": embedding
                })
                chunk_index += 1
            else:
                sentences = nltk.sent_tokenize(paragraph)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    if current_length + sentence_length > self.chunk_size and current_chunk:
                        chunk_text = " ".join(current_chunk)
                        embedding = self.embedding_model.encode([chunk_text]).tolist()[0]
                        chunks.append({
                            "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                            "text": chunk_text,
                            "document_id": document_id,
                            "chunk_index": chunk_index,
                            "level": "sentence",
                            "embedding": embedding
                        })
                        current_chunk = [sentence]
                        current_length = sentence_length
                        chunk_index += 1
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    embedding = self.embedding_model.encode([chunk_text]).tolist()[0]
                    chunks.append({
                        "chunk_id": f"{document_id}_{uuid.uuid4().hex[:8]}",
                        "text": chunk_text,
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "level": "sentence",
                        "embedding": embedding
                    })
                    chunk_index += 1
        return chunks