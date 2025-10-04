# reco/index/persistent_indexer.py
"""
Persistent Indexer - Sistema de indexação que sincroniza com PostgreSQL.

Responsável por:
- Manter embeddings sincronizados com o banco de dados
- Detectar mudanças no conteúdo das trilhas
- Gerenciar cache de embeddings
- Sincronização incremental e em lote
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

import numpy as np

from reco.config import RecoConfig
from reco.database.connection import DatabaseConnection
from reco.embeddings.embedding_provider import EmbeddingProvider
from schemas.trail_candidate import TrailCandidate

logger = logging.getLogger(__name__)


class PersistentIndexer:
    """
    Sistema de indexação persistente que mantém embeddings sincronizados.
    
    Funcionalidades:
    - Sincronização incremental com trilhas da API
    - Detecção de mudanças no conteúdo
    - Geração de embeddings em lote
    - Cache inteligente para performance
    """
    
    def __init__(self, cfg: RecoConfig):
        self.cfg = cfg
        self.db = DatabaseConnection(cfg)
        self.embedding_provider = EmbeddingProvider.from_config(cfg)
        
        # Cache para evitar recomputação desnecessária
        self._content_hashes: Dict[str, str] = {}
        self._last_sync: Optional[datetime] = None
    
    def sync_trails(self, candidates: List[TrailCandidate]) -> Dict[str, int]:
        """
        Sincroniza trilhas com o banco de dados.
        
        Args:
            candidates: Lista de TrailCandidate para sincronizar
            
        Returns:
            Dict com estatísticas da sincronização
        """
        logger.info(f"Starting sync for {len(candidates)} trails")
        
        # 1. Calcular hashes de conteúdo
        current_hashes = self._calculate_content_hashes(candidates)
        
        # 2. Identificar trilhas que precisam de atualização
        outdated_ids = self._identify_outdated_embeddings(current_hashes)
        new_ids = self._identify_new_embeddings(current_hashes)
        
        logger.info(f"Found {len(outdated_ids)} outdated and {len(new_ids)} new embeddings")
        
        # 3. Processar em lotes
        stats = {
            'total_processed': 0,
            'new_embeddings': 0,
            'updated_embeddings': 0,
            'skipped': 0,
            'errors': 0
        }
        
        # Processar trilhas novas e desatualizadas
        to_process = list(set(outdated_ids + new_ids))
        if to_process:
            candidates_to_process = [c for c in candidates if str(c.publicId) in to_process]
            batch_stats = self._process_batch(candidates_to_process, current_hashes)
            stats.update(batch_stats)
        
        # 4. Atualizar cache
        self._content_hashes.update(current_hashes)
        self._last_sync = datetime.now()
        
        logger.info(f"Sync completed: {stats}")
        return stats
    
    def _calculate_content_hashes(self, candidates: List[TrailCandidate]) -> Dict[str, str]:
        """Calcula hashes SHA256 do conteúdo de cada trilha."""
        hashes = {}
        
        for candidate in candidates:
            # Usar combined_text para detectar mudanças
            content = self._get_trail_content(candidate)
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            hashes[str(candidate.publicId)] = content_hash
        
        return hashes
    
    def _get_trail_content(self, candidate: TrailCandidate) -> str:
        """Extrai conteúdo textual da trilha para hash."""
        parts = [
            candidate.title or "",
            candidate.subtitle or "",
            candidate.description or "",
            " ".join(candidate.topics or []),
            " ".join(candidate.tags or []),
            candidate.combined_text or "",
        ]
        return " | ".join(p for p in parts if p)
    
    def _identify_outdated_embeddings(self, current_hashes: Dict[str, str]) -> List[str]:
        """Identifica embeddings que precisam ser atualizados."""
        return self.db.get_outdated_embeddings(current_hashes)
    
    def _identify_new_embeddings(self, current_hashes: Dict[str, str]) -> List[str]:
        """Identifica trilhas que não têm embeddings ainda."""
        # Buscar IDs existentes no banco
        existing_ids = self._get_existing_embedding_ids()
        
        # Retornar IDs que não existem
        return [tid for tid in current_hashes.keys() if tid not in existing_ids]
    
    def _get_existing_embedding_ids(self) -> Set[str]:
        """Obtém IDs de embeddings existentes no banco."""
        query = "SELECT public_id FROM reco.trail_embeddings"
        results = self.db.execute_query(query)
        return {row['public_id'] for row in results}
    
    def _process_batch(
        self, 
        candidates: List[TrailCandidate], 
        content_hashes: Dict[str, str]
    ) -> Dict[str, int]:
        """Processa um lote de trilhas para gerar embeddings."""
        stats = {
            'total_processed': 0,
            'new_embeddings': 0,
            'updated_embeddings': 0,
            'skipped': 0,
            'errors': 0
        }
        
        if not candidates:
            return stats
        
        try:
            # 1. Preparar textos para embedding
            texts = [self._get_trail_content(c) for c in candidates]
            
            # 2. Gerar embeddings em lote
            embeddings = self.embedding_provider.embed_texts(
                texts, 
                normalize=True, 
                batch_size=self.cfg.EMBEDDING_BATCH_SIZE
            )
            
            # 3. Salvar no banco
            for candidate, embedding in zip(candidates, embeddings):
                try:
                    public_id = str(candidate.publicId)
                    content_hash = content_hashes[public_id]
                    
                    # Preparar metadados
                    metadata = {
                        "status": candidate.status or "",
                        "difficulty": (candidate.difficulty or "").lower(),
                        "area": candidate.area or "",
                    }
                    
                    # Verificar se é novo ou atualização
                    is_new = public_id not in self._get_existing_embedding_ids()
                    
                    # Salvar no banco
                    self.db.upsert_embedding(
                        public_id=public_id,
                        embedding=embedding,
                        model_version=self.cfg.MODEL_VERSION,
                        content_hash=content_hash,
                        metadata=metadata
                    )
                    
                    # Atualizar estatísticas
                    stats['total_processed'] += 1
                    if is_new:
                        stats['new_embeddings'] += 1
                    else:
                        stats['updated_embeddings'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing trail {candidate.publicId}: {e}")
                    stats['errors'] += 1
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            stats['errors'] += len(candidates)
        
        return stats
    
    def get_vector_index(self) -> 'VectorIndex':
        """Obtém um VectorIndex configurado para usar o backend PostgreSQL."""
        from reco.index.vector_index import VectorIndex
        
        return VectorIndex.from_config(
            cfg=self.cfg,
            backend="postgresql",
            index_name=self.cfg.INDEX_TRILHAS
        )
    
    def get_stats(self) -> Dict[str, any]:
        """Obtém estatísticas do sistema de indexação."""
        db_stats = self.db.get_embedding_stats()
        
        # Ensure model_versions is a list
        model_versions = db_stats.get('model_versions', [])
        if not isinstance(model_versions, list):
            model_versions = []
        
        return {
            'total_embeddings': db_stats.get('total_embeddings', 0),
            'model_versions': model_versions,
            'last_updated': db_stats.get('last_updated'),
            'last_sync': self._last_sync,
            'cached_hashes': len(self._content_hashes),
        }
    
    def cleanup_orphaned_embeddings(self, valid_public_ids: Set[str]) -> int:
        """
        Remove embeddings órfãos (sem trilha correspondente).
        
        Args:
            valid_public_ids: Set de IDs válidos de trilhas
            
        Returns:
            Número de embeddings removidos
        """
        # Buscar todos os embeddings
        all_embeddings = self._get_existing_embedding_ids()
        
        # Identificar órfãos
        orphaned_ids = all_embeddings - valid_public_ids
        
        if not orphaned_ids:
            return 0
        
        # Remover órfãos
        for orphan_id in orphaned_ids:
            self.db.delete_embedding(orphan_id)
        
        logger.info(f"Removed {len(orphaned_ids)} orphaned embeddings")
        return len(orphaned_ids)
    
    def close(self) -> None:
        """Fecha conexões e limpa recursos."""
        if self.db:
            self.db.close()
