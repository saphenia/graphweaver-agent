"""Embeddings module for text and knowledge graph embeddings."""

from .text_embeddings import TextEmbedder, EmbeddingResult, embed_all_metadata
from .kg_embeddings import KGEmbedder, generate_all_kg_embeddings
from .vector_indexes import VectorIndexManager, SemanticSearch
from .semantic_fk import SemanticFKDiscovery, SemanticFKCandidate

__all__ = [
    "TextEmbedder",
    "EmbeddingResult",
    "embed_all_metadata",
    "KGEmbedder",
    "generate_all_kg_embeddings",
    "VectorIndexManager",
    "SemanticSearch",
    "SemanticFKDiscovery",
    "SemanticFKCandidate",
]
