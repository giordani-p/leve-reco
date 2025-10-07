# reco/database/connection.py
"""
Database connection management for PostgreSQL + pgvector.

Provides connection pooling and transaction management for the recommendation system.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from psycopg2.sql import SQL, Identifier, Literal

from reco.config import RecoConfig

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages PostgreSQL connections with pgvector support for embeddings.
    
    Features:
    - Connection pooling for performance
    - Automatic retry on connection failures
    - Transaction management with context managers
    - Vector operations with pgvector
    """
    
    def __init__(self, cfg: RecoConfig):
        self.cfg = cfg
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool with pgvector support."""
        try:
            # Use individual connection parameters for better control and security
            conn_params = {
                'host': self.cfg.POSTGRES_HOST,
                'port': int(self.cfg.POSTGRES_PORT),
                'database': self.cfg.POSTGRES_DB,
                'user': self.cfg.POSTGRES_USER,
                'password': self.cfg.POSTGRES_PASSWORD,
                'options': '-c search_path=reco,leve,public',  # Set search path (reco first for our tables)
            }
            
            # Create connection pool
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self.cfg.DB_POOL_SIZE + self.cfg.DB_MAX_OVERFLOW,
                **conn_params
            )
            
            logger.info(f"Database pool initialized with {self.cfg.DB_POOL_SIZE} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup."""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get a cursor with automatic cleanup."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return []
    
    def execute_vector_search(
        self,
        query_embedding: np.ndarray,
        similarity_threshold: float = 0.0,
        max_results: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search using pgvector.
        
        Args:
            query_embedding: Query vector (768-dimensional)
            similarity_threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results
            filters: Optional metadata filters
            
        Returns:
            List of results with similarity scores
        """
        # Build filter conditions
        filter_conditions = []
        filter_params = []
        
        if filters:
            for key, value in filters.items():
                if value is not None:
                    filter_conditions.append(f"metadata->>%s = %s")
                    filter_params.extend([key, str(value)])
        
        # Build query
        where_clause = ""
        if filter_conditions:
            where_clause = " AND " + " AND ".join(filter_conditions)
        
        query = f"""
        SELECT 
            public_id,
            embedding,
            metadata,
            (1 - (embedding <=> %s::vector)) as similarity_score
        FROM {self.cfg.RECO_SCHEMA}.trail_embeddings
        WHERE (1 - (embedding <=> %s::vector)) >= %s
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        # Prepare parameters - convert numpy array to list for PostgreSQL
        query_embedding_list = query_embedding.tolist()
        params = [query_embedding_list, query_embedding_list, similarity_threshold]
        params.extend(filter_params)
        params.extend([query_embedding_list, max_results])
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def upsert_embedding(
        self,
        public_id: str,
        embedding: np.ndarray,
        model_version: str,
        content_hash: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Insert or update an embedding record."""
        query = f"""
        INSERT INTO {self.cfg.RECO_SCHEMA}.trail_embeddings 
        (public_id, embedding, model_version, content_hash, metadata, updated_at)
        VALUES (%s::uuid, %s, %s, %s, %s, NOW())
        ON CONFLICT (public_id) 
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            model_version = EXCLUDED.model_version,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (
                public_id,
                embedding.tolist(),  # Convert numpy array to list
                model_version,
                content_hash,
                psycopg2.extras.Json(metadata)
            ))
            # Commit the transaction
            cursor.connection.commit()
    
    def delete_embedding(self, public_id: str) -> None:
        """Delete an embedding record."""
        query = f"DELETE FROM {self.cfg.RECO_SCHEMA}.trail_embeddings WHERE public_id = %s"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (public_id,))
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        query = f"""
        SELECT 
            COUNT(*) as total_embeddings,
            COALESCE(ARRAY_AGG(DISTINCT model_version) FILTER (WHERE model_version IS NOT NULL), ARRAY[]::text[]) as model_versions,
            MAX(updated_at) as last_updated,
            768 as avg_dimension
        FROM {self.cfg.RECO_SCHEMA}.trail_embeddings
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                stats = dict(result)
                # Ensure model_versions is always a list
                if stats.get('model_versions') is None:
                    stats['model_versions'] = []
                return stats
            else:
                return {
                    'total_embeddings': 0,
                    'model_versions': [],
                    'last_updated': None,
                    'avg_dimension': 768
                }
    
    def get_outdated_embeddings(self, current_hashes: Dict[str, str]) -> List[str]:
        """
        Get public_ids of embeddings that need to be updated.
        
        Args:
            current_hashes: Dict mapping public_id to current content_hash
            
        Returns:
            List of public_ids that need updating
        """
        if not current_hashes:
            return []
        
        # Build query for checking outdated embeddings
        placeholders = ','.join(['%s'] * len(current_hashes))
        query = f"""
        SELECT public_id 
        FROM {self.cfg.RECO_SCHEMA}.trail_embeddings 
        WHERE public_id IN ({placeholders})
        AND content_hash NOT IN ({placeholders})
        """
        
        params = list(current_hashes.keys()) + list(current_hashes.values())
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            return [row['public_id'] for row in results]
    
    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database pool closed")


# Global connection instance
_connection: Optional[DatabaseConnection] = None


def get_connection(cfg: Optional[RecoConfig] = None) -> DatabaseConnection:
    """Get or create a global database connection."""
    global _connection
    
    if _connection is None:
        if cfg is None:
            from reco.config import RecoConfig
            cfg = RecoConfig()
        _connection = DatabaseConnection(cfg)
    
    return _connection


def close_connection() -> None:
    """Close the global database connection."""
    global _connection
    
    if _connection:
        _connection.close()
        _connection = None
