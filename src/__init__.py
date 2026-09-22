"""
Public API surface for the RAG package.

Import examples:

    from src import CorrectiveRAG, CRAGConfig
    from src import RAG, VectorStoreType, EmbeddingModel
    from src.utils.chunker import ChunkingStrategy
"""

from .rag import RAG, VectorStoreType, EmbeddingModel
from .corrective_rag import CorrectiveRAG, CRAGConfig, CRAGResult, PipelineTrace
from .generator import OllamaGenerator, GeneratorConfig
from .grader import RetrievalGrader, AnswerGrader
from .reranker import CrossEncoderReranker

__all__ = [
    # Core RAG
    "RAG",
    "VectorStoreType",
    "EmbeddingModel",
    # CRAG pipeline
    "CorrectiveRAG",
    "CRAGConfig",
    "CRAGResult",
    "PipelineTrace",
    # Components
    "OllamaGenerator",
    "GeneratorConfig",
    "RetrievalGrader",
    "AnswerGrader",
    "CrossEncoderReranker",
]