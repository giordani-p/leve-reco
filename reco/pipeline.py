# reco/pipeline.py
"""
Pipeline Principal do Sistema de Recomendação - Leve Agents

Este módulo implementa o pipeline completo de recomendação de trilhas educacionais,
processando perfis de usuários e retornando recomendações personalizadas baseadas
em busca híbrida (semântica + textual).

Fluxo de Processamento:
1. Carregamento: Snapshot do usuário + catálogo de trilhas via API
2. Normalização: Conversão para TrailCandidate com deduplicação e filtros
3. Construção de consulta: Pergunta + pistas do perfil + contexto adicional
4. Indexação: Criação de índices vetoriais (MPNet) e BM25 persistentes
5. Retrieval: Busca híbrida combinando semântica e textual
6. Ranking: Aplicação de regras de negócio e boosts personalizados
7. Output: Geração de explicações e validação final

Arquitetura do Sistema:
- Índices persistentes com PostgreSQL + pgvector
- Integração robusta com API da Leve
- Sistema de logging completo para observabilidade
- Validação rigorosa com schemas Pydantic
- Filtros de segurança para consultas inadequadas
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from reco.config import RecoConfig
from reco.data_loader import load_snapshot
from reco.data_loader_api import fetch_trails as fetch_trails_api
from reco.normalizer import to_candidates, dedupe_by_public_id, filter_by_status
from reco.query_builder import build as build_query

from reco.indexer import Indexer as BM25Indexer

# Denso / Híbrido
from reco.embeddings.embedding_provider import EmbeddingProvider
from reco.index.vector_index import VectorIndex, VectorItem
from reco.retriever.dense_retriever import DenseRetriever
from reco.retriever.hybrid_retriever import HybridRetriever, HybridResult

from reco.ranker import rank as rank_candidates, ScoredCandidate
from reco.explainer import make_reason
from reco.output_builder import build_output

from schemas.trail_input import TrailInput
from schemas.trail_output import TrailOutput
from schemas.trail_candidate import TrailCandidate

from validators.trail_output_checks import apply_business_rules
from reco.safety_filters import check_query_safety, create_safety_response


# -----------------------------
# Funções Auxiliares Internas
# -----------------------------
def _item_vector_text(cand: TrailCandidate) -> str:
    """
    Constrói o texto base para geração de embeddings do item (trilha).
    Mantém coerência com o query_builder para consistência no pipeline.
    """
    parts = [
        cand.title or "",
        cand.subtitle or "",
        cand.description or "",
        " ".join(cand.topics or []),
        " ".join(cand.tags or []),
        cand.combined_text or "",
    ]
    return " | ".join(p for p in parts if p)




def _build_persistent_vector_index(
    candidates: List[TrailCandidate],
    cfg: RecoConfig,
) -> Tuple[VectorIndex, EmbeddingProvider]:
    """
    Constrói um índice vetorial persistente usando PostgreSQL + pgvector.
    Utiliza embeddings já persistidos no banco de dados para otimizar performance.
    """
    from reco.index.persistent_indexer import PersistentIndexer
    
    # Cria indexer persistente para PostgreSQL
    indexer = PersistentIndexer(cfg)
    
    # Obtém índice configurado para PostgreSQL (sem sincronização)
    index = indexer.get_vector_index()
    
    # Cria provider de embeddings para consultas
    provider = EmbeddingProvider.from_config(cfg)
    
    return index, provider


def sync_trails_to_database(
    candidates: List[TrailCandidate],
    cfg: RecoConfig,
) -> Dict[str, int]:
    """
    Sincroniza trilhas com o banco de dados PostgreSQL.
    Esta operação deve ser executada separadamente, não a cada consulta.
    
    Args:
        candidates: Lista de TrailCandidate para sincronizar
        cfg: Configuração do sistema
        
    Returns:
        Estatísticas da sincronização (inseridos, atualizados, etc.)
    """
    from reco.index.persistent_indexer import PersistentIndexer
    
    indexer = PersistentIndexer(cfg)
    try:
        sync_stats = indexer.sync_trails(candidates)
        return sync_stats
    finally:
        indexer.close()


def _bm25_callable_for_trilhas(
    candidates: List[TrailCandidate],
    cfg: RecoConfig,
) -> tuple:
    """
    Prepara o indexer BM25 e retorna um callable para busca.
    
    Returns:
        Tupla contendo (indexer_bm25, função_de_busca)
        onde a função de busca tem assinatura: bm25_search_topk(query_text, k) -> List[(id, score_bm25, meta)]
    """
    # Cria indexer BM25 com configuração fornecida
    bm25 = BM25Indexer(cfg)
    bm25.fit_items(candidates)

    def _searcher(query_text: str, k: int) -> List[tuple]:
        # Adapta formato de retorno do indexer para o HybridRetriever
        # Suporta formatos: [(TrailCandidate, score)] ou [(id, score, meta)]
        raw = bm25.search_topk(query_text, k)
        out: List[tuple] = []
        
        if raw and isinstance(raw[0], tuple) and len(raw[0]) == 2 and isinstance(raw[0][0], TrailCandidate):
            # Formato [(TrailCandidate, score)] - converte para formato esperado
            for cand, score in raw:
                meta = {
                    "status": cand.status or "", 
                    "difficulty": (cand.difficulty or "").lower(), 
                    "area": getattr(cand, "area", "") or ""
                }
                out.append((str(cand.publicId), float(score), meta))
        else:
            # Formato já compatível: [(id, score, meta)]
            out = [(str(r[0]), float(r[1]), r[2] if len(r) > 2 else {}) for r in raw]
        return out

    return bm25, _searcher


def _map_reasons(
    ranked: List[ScoredCandidate],
    query_text: str,
    limit: int,
) -> Dict[str, str]:
    """
    Gera dicionário de razões (why_match) por publicId para os primeiros 'limit' itens.
    
    Args:
        ranked: Lista de candidatos ranqueados
        query_text: Texto da consulta original
        limit: Número máximo de itens para processar
        
    Returns:
        Dicionário mapeando publicId -> explicação do match
    """
    reasons: Dict[str, str] = {}
    topk = ranked[: max(1, min(limit, len(ranked)))]
    for sc in topk:
        pid = str(sc.candidate.publicId)
        reasons[pid] = make_reason(sc, query_text)
    return reasons


# -----------------------------
# Pipeline Principal de Recomendação
# -----------------------------
def run(
    user_input: TrailInput,
    snapshot_path: Optional[str],
    trails_path: str,
    cfg: Optional[RecoConfig] = None,
) -> TrailOutput:
    """
    Executa o pipeline completo de recomendação e retorna um TrailOutput validado.
    
    Args:
        user_input: Entrada do usuário com pergunta e contexto
        snapshot_path: Caminho para o snapshot do usuário (opcional)
        trails_path: Caminho para o catálogo de trilhas
        cfg: Configuração do sistema (usa padrão se não fornecida)
        
    Returns:
        TrailOutput com recomendações e explicações
    """
    cfg = cfg or RecoConfig()

    # Etapa 0: Verificação de segurança da consulta
    safety_result = check_query_safety(user_input.user_question)
    if not safety_result.is_safe:
        return create_safety_response(safety_result, user_input.user_question)

    # Etapa 1: Carregamento de dados
    snapshot = load_snapshot(snapshot_path)
    raw_trails = fetch_trails_api(cfg)

    # Etapa 2: Normalização, deduplicação e filtros
    candidates_all = to_candidates(raw_trails)
    candidates_all = dedupe_by_public_id(candidates_all)
    candidates = filter_by_status(candidates_all, cfg.ALLOWED_STATUS)

    # Verifica se há candidatos após filtros
    if not candidates:
        empty = build_output(
            ranked=[],
            query_text=user_input.user_question,
            max_results=user_input.max_results,
            reasons_by_id={},
        )
        return apply_business_rules(empty)

    # Etapa 3: Construção da consulta enriquecida
    query_text = build_query(
        user_question=user_input.user_question,
        snapshot=snapshot,
        cfg=cfg,
        contexto_extra=user_input.contexto_extra,
    )

    # Etapa 4: Construção do índice vetorial persistente
    vindex, prov = _build_persistent_vector_index(candidates, cfg)

    # Etapa 5: Configuração do sistema de retrieval híbrido
    dense = DenseRetriever.from_config(cfg, vector_index=vindex, embedding_provider=prov)
    
    # Prepara BM25 para o catálogo atual
    bm25, bm25_search_topk = _bm25_callable_for_trilhas(candidates, cfg)

    # Combina BM25 + Busca Densa com normalização e blending
    hybrid = HybridRetriever.from_config(cfg, dense_retriever=dense, bm25_search_topk=bm25_search_topk)
    # Executa busca híbrida
    hybrid_res: List[HybridResult] = hybrid.search(
        query_text=query_text,
        k=cfg.TOP_K_DEFAULT,
        filters={"status": "Published"} if cfg.ENFORCE_PUBLISHED else None,
    )

    # Etapa 6: Adaptação dos resultados para o sistema de ranking
    scored_candidates = []
    for hr in hybrid_res:
        try:
            candidate = next(c for c in candidates if str(c.publicId) == hr.id)
            scored_candidates.append({
                "candidate": candidate,
                "score_combined": hr.score_combined,
                "score_semantic": hr.score_semantic,
                "score_bm25": hr.score_bm25,
            })
        except StopIteration:
            # Candidato não encontrado - pode ter sido filtrado ou removido
            print(f"[AVISO] Trilha {hr.id} encontrada pelo sistema híbrido mas removida durante normalização/filtros (status, duplicatas, validação)", file=sys.stderr)
            continue

    # Etapa 7: Aplicação de ranking com regras de negócio
    ranked = rank_candidates(
        scored_candidates=scored_candidates,
        query_text=query_text,
        cfg=cfg,
        collection="trilhas",
        max_results=user_input.max_results,
    )

    # Etapa 8: Geração de explicações (why_match) para o top-K
    reasons_by_id = _map_reasons(ranked, query_text, limit=user_input.max_results)

    # Etapa 9: Construção da saída final
    output = build_output(
        ranked=ranked,
        query_text=query_text,
        max_results=user_input.max_results,
        reasons_by_id=reasons_by_id,
    )

    # Etapa 10: Aplicação de regras de negócio e validação final
    final_output = apply_business_rules(output)
    return final_output
