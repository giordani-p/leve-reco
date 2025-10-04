# reco/database/__init__.py
"""
Database module for recommendation system.

Provides PostgreSQL + pgvector integration for persistent embeddings storage.
"""

from .connection import DatabaseConnection, get_connection
from .schema import EmbeddingRecord, EmbeddingMetadata

__all__ = [
    "DatabaseConnection",
    "get_connection", 
    "EmbeddingRecord",
    "EmbeddingMetadata",
]
