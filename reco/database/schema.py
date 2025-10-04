# reco/database/schema.py
"""
Database schema definitions for the recommendation system.

Defines data structures and validation for embedding records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

import numpy as np


@dataclass
class EmbeddingMetadata:
    """Metadata associated with an embedding record."""
    
    status: str = ""
    difficulty: str = ""
    area: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "status": self.status,
            "difficulty": self.difficulty,
            "area": self.area,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingMetadata":
        """Create from dictionary."""
        return cls(
            status=data.get("status", ""),
            difficulty=data.get("difficulty", ""),
            area=data.get("area", ""),
        )
    
    def __post_init__(self):
        """Normalize string values."""
        self.difficulty = self.difficulty.lower() if self.difficulty else ""


@dataclass
class EmbeddingRecord:
    """Complete embedding record with all metadata."""
    
    public_id: str
    embedding: np.ndarray
    model_version: str
    content_hash: str
    metadata: EmbeddingMetadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and normalize the record."""
        # Ensure embedding is the right shape and type
        if self.embedding.ndim != 1:
            raise ValueError(f"Embedding must be 1D, got {self.embedding.ndim}D")
        
        if len(self.embedding) != 768:
            raise ValueError(f"Embedding must be 768-dimensional, got {len(self.embedding)}")
        
        # Ensure embedding is float32
        if self.embedding.dtype != np.float32:
            self.embedding = self.embedding.astype(np.float32)
        
        # Normalize metadata
        if isinstance(self.metadata, dict):
            self.metadata = EmbeddingMetadata.from_dict(self.metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "public_id": self.public_id,
            "embedding": self.embedding.tolist(),
            "model_version": self.model_version,
            "content_hash": self.content_hash,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingRecord":
        """Create from dictionary (e.g., from database)."""
        return cls(
            public_id=data["public_id"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            model_version=data["model_version"],
            content_hash=data["content_hash"],
            metadata=EmbeddingMetadata.from_dict(data["metadata"]),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )
    
    def is_outdated(self, current_content_hash: str) -> bool:
        """Check if this embedding is outdated based on content hash."""
        return self.content_hash != current_content_hash
    
    def similarity_to(self, other_embedding: np.ndarray) -> float:
        """Calculate cosine similarity to another embedding."""
        # Ensure both embeddings are normalized
        norm1 = np.linalg.norm(self.embedding)
        norm2 = np.linalg.norm(other_embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(self.embedding, other_embedding)
        return dot_product / (norm1 * norm2)


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    
    public_id: str
    similarity_score: float
    metadata: EmbeddingMetadata
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Normalize metadata if it's a dict."""
        if isinstance(self.metadata, dict):
            self.metadata = EmbeddingMetadata.from_dict(self.metadata)


@dataclass
class EmbeddingStats:
    """Statistics about the embedding database."""
    
    total_embeddings: int
    model_versions: list
    last_updated: Optional[datetime]
    avg_dimension: Optional[float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingStats":
        """Create from database result."""
        return cls(
            total_embeddings=data.get("total_embeddings", 0),
            model_versions=data.get("model_versions", []),
            last_updated=data.get("last_updated"),
            avg_dimension=data.get("avg_dimension"),
        )
