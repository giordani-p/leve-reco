# reco/migration/migrator.py
"""
Data migration utilities for the recommendation system.

Provides tools for migrating existing data to PostgreSQL + pgvector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

from reco.config import RecoConfig
from reco.data_loader import load_trails as load_trails_file
from reco.data_loader_api import fetch_trails as fetch_trails_api
from reco.normalizer import to_candidates, dedupe_by_public_id, filter_by_status
from reco.index.persistent_indexer import PersistentIndexer

logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    """Statistics from a migration operation."""
    
    total_trails: int = 0
    processed_trails: int = 0
    new_embeddings: int = 0
    updated_embeddings: int = 0
    skipped_trails: int = 0
    error_trails: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of migration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_trails == 0:
            return 0.0
        return (self.processed_trails / self.total_trails) * 100


class DataMigrator:
    """
    Migrates data from files/API to PostgreSQL + pgvector.
    
    Features:
    - Migrate from files or API
    - Batch processing for large datasets
    - Progress tracking and logging
    - Error handling and recovery
    - Validation and statistics
    """
    
    def __init__(self, cfg: RecoConfig, batch_size: Optional[int] = None):
        self.cfg = cfg
        self.batch_size = batch_size or cfg.SYNC_BATCH_SIZE
        self.indexer = PersistentIndexer(cfg)
    
    def migrate_from_files(self, trails_path: str) -> MigrationStats:
        """
        Migrate trails from local files to PostgreSQL.
        
        Args:
            trails_path: Path to trails JSON file
            
        Returns:
            Migration statistics
        """
        logger.info(f"Starting migration from files: {trails_path}")
        
        stats = MigrationStats(start_time=datetime.now())
        
        try:
            # 1. Load trails from files
            raw_trails = load_trails_file(trails_path)
            stats.total_trails = len(raw_trails)
            
            # 2. Normalize and filter
            candidates_all = to_candidates(raw_trails)
            candidates_all = dedupe_by_public_id(candidates_all)
            candidates = filter_by_status(candidates_all, self.cfg.ALLOWED_STATUS)
            
            logger.info(f"Loaded {len(candidates)} valid trails for migration")
            
            # 3. Migrate in batches
            batch_stats = self._migrate_batches(candidates)
            
            # 4. Update statistics
            stats.processed_trails = batch_stats['total_processed']
            stats.new_embeddings = batch_stats['new_embeddings']
            stats.updated_embeddings = batch_stats['updated_embeddings']
            stats.skipped_trails = batch_stats['skipped']
            stats.error_trails = batch_stats['errors']
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            stats.error_trails = stats.total_trails
        
        finally:
            stats.end_time = datetime.now()
            logger.info(f"Migration completed: {stats}")
        
        return stats
    
    def migrate_from_api(self) -> MigrationStats:
        """
        Migrate trails from API to PostgreSQL.
        
        Returns:
            Migration statistics
        """
        logger.info("Starting migration from API")
        
        stats = MigrationStats(start_time=datetime.now())
        
        try:
            # 1. Load trails from API
            raw_trails = fetch_trails_api(self.cfg)
            stats.total_trails = len(raw_trails)
            
            # 2. Normalize and filter
            candidates_all = to_candidates(raw_trails)
            candidates_all = dedupe_by_public_id(candidates_all)
            candidates = filter_by_status(candidates_all, self.cfg.ALLOWED_STATUS)
            
            logger.info(f"Loaded {len(candidates)} valid trails for migration")
            
            # 3. Migrate in batches
            batch_stats = self._migrate_batches(candidates)
            
            # 4. Update statistics
            stats.processed_trails = batch_stats['total_processed']
            stats.new_embeddings = batch_stats['new_embeddings']
            stats.updated_embeddings = batch_stats['updated_embeddings']
            stats.skipped_trails = batch_stats['skipped']
            stats.error_trails = batch_stats['errors']
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            stats.error_trails = stats.total_trails
        
        finally:
            stats.end_time = datetime.now()
            logger.info(f"Migration completed: {stats}")
        
        return stats
    
    def _migrate_batches(self, candidates: List) -> Dict[str, int]:
        """Migrate candidates in batches."""
        batch_size = self.batch_size
        total_batches = (len(candidates) + batch_size - 1) // batch_size
        
        total_stats = {
            'total_processed': 0,
            'new_embeddings': 0,
            'updated_embeddings': 0,
            'skipped': 0,
            'errors': 0
        }
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} trails)")
            
            try:
                # Sync this batch
                batch_stats = self.indexer.sync_trails(batch)
                
                # Accumulate statistics
                for key in total_stats:
                    total_stats[key] += batch_stats.get(key, 0)
                
                logger.info(f"Batch {batch_num} completed: {batch_stats}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                total_stats['errors'] += len(batch)
        
        return total_stats
    
    def validate_migration(self, expected_trail_ids: Optional[Set[str]] = None) -> Dict[str, any]:
        """
        Validate the migration by checking database state.
        
        Args:
            expected_trail_ids: Set of expected trail IDs (optional)
            
        Returns:
            Validation results
        """
        logger.info("Validating migration")
        
        # Get database statistics
        db_stats = self.indexer.get_stats()
        
        validation = {
            'total_embeddings': db_stats['total_embeddings'],
            'model_versions': db_stats['model_versions'],
            'last_updated': db_stats['last_updated'],
            'validation_passed': True,
            'issues': []
        }
        
        # Check if we have embeddings
        if db_stats['total_embeddings'] == 0:
            validation['validation_passed'] = False
            validation['issues'].append("No embeddings found in database")
        
        # Check model versions
        expected_model = self.cfg.MODEL_VERSION
        if expected_model not in db_stats['model_versions']:
            validation['validation_passed'] = False
            validation['issues'].append(f"Expected model {expected_model} not found")
        
        # Check expected trail IDs if provided
        if expected_trail_ids:
            # This would require additional database queries
            # For now, just log the check
            logger.info(f"Expected {len(expected_trail_ids)} trail IDs")
        
        logger.info(f"Validation completed: {validation}")
        return validation
    
    def cleanup_orphans(self, valid_trail_ids: Set[str]) -> int:
        """
        Clean up orphaned embeddings.
        
        Args:
            valid_trail_ids: Set of valid trail IDs
            
        Returns:
            Number of orphaned embeddings removed
        """
        logger.info(f"Cleaning up orphaned embeddings (valid IDs: {len(valid_trail_ids)})")
        
        removed_count = self.indexer.cleanup_orphaned_embeddings(valid_trail_ids)
        
        logger.info(f"Removed {removed_count} orphaned embeddings")
        return removed_count
    
    def get_migration_status(self) -> Dict[str, any]:
        """Get current migration status."""
        return self.indexer.get_stats()
    
    
    def create_schema_if_missing(self) -> bool:
        """
        Create the required schema and tables if they don't exist.
        
        Returns:
            True if schema was created or already exists, False if failed
        """
        logger.info("Ensuring database schema exists")
        
        try:
            # Read and execute schema SQL
            schema_path = Path(__file__).parent.parent.parent / "helpers" / "reco_schema.sql"
            
            if not schema_path.exists():
                logger.error(f"Schema file not found: {schema_path}")
                return False
            
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            # Execute schema creation
            with self.indexer.db.get_cursor() as cursor:
                cursor.execute(schema_sql)
                cursor.connection.commit()
            
            logger.info("Database schema created/verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False
    
    def close(self) -> None:
        """Close connections and cleanup resources."""
        if self.indexer:
            self.indexer.close()
