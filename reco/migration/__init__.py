# reco/migration/__init__.py
"""
Migration utilities for the recommendation system.

Provides tools for migrating data to PostgreSQL + pgvector.
"""

from .migrator import DataMigrator, MigrationStats

__all__ = [
    "DataMigrator",
    "MigrationStats",
]
