# reco/normalizer.py
"""
Sistema de Normalização de Dados - Leve Agents

Este módulo é responsável pela normalização e limpeza de dados do catálogo de trilhas
educacionais, convertendo dados brutos em formatos padronizados e aplicando filtros
de qualidade para garantir consistência no pipeline de recomendação.

Funcionalidades Principais:
- Conversão de dados brutos para TrailCandidate (schema padronizado)
- Preenchimento automático de campos ausentes (combined_text)
- Filtragem por status permitido (case-insensitive)
- Deduplicação por publicId mantendo maior completude
- Ordem estável para evitar flutuações no pipeline
- Validação de dados obrigatórios
- Logging de operações para observabilidade
- Compatibilidade com schemas Pydantic v1 e v2
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from uuid import UUID
import logging

from schemas.trail_candidate import TrailCandidate
from reco.config import RecoConfig

logger = logging.getLogger(__name__)


# -----------------------------
# Conversão e Validação de Dados
# -----------------------------
def to_candidates(raw_items: List[Dict]) -> List[TrailCandidate]:
    """
    Converte a lista bruta do catálogo em TrailCandidate.
    Ignora entradas inválidas (registra aviso); lança erro apenas se nenhum item válido for encontrado.
    
    Args:
        raw_items: Lista de dicionários com dados brutos das trilhas
        
    Returns:
        Lista de TrailCandidate válidos
        
    Raises:
        ValueError: Se nenhuma trilha válida for encontrada após conversão
    """
    candidates: List[TrailCandidate] = []
    for i, item in enumerate(raw_items or []):
        try:
            # Validação e conversão usando schema Pydantic
            cand = TrailCandidate.from_source(item)
            # Aplicação de higienização leve (não altera semântica)
            cand = _sanitize_candidate(cand)
            candidates.append(cand)
        except Exception as e:
            logger.warning("Normalizer: ignorando item inválido na posição %s (%s)", i, e)
            continue

    if not candidates:
        raise ValueError("Nenhuma trilha válida encontrada após normalização.")
    return candidates


def _sanitize_candidate(c: TrailCandidate) -> TrailCandidate:
    """
    Aplica limpeza e normalização de campos para garantir consistência no pipeline.
    
    Args:
        c: TrailCandidate a ser sanitizado
        
    Returns:
        TrailCandidate com campos limpos e normalizados
    """
    try:
        title = (c.title or "").strip()
        slug = (c.slug or "").strip()
        description = (c.description or "").strip()
        subtitle = (getattr(c, "subtitle", None) or "").strip()
        difficulty = (c.difficulty or "").strip()
        status = (c.status or "").strip()

        # Garante que tags e tópicos sejam listas válidas
        tags = list(c.tags or []) if isinstance(c.tags, list) else []
        topics = list(c.topics or []) if isinstance(c.topics, list) else []

        # Preserva combined_text existente (pode ser preenchido posteriormente se vazio)
        combined_text = c.combined_text

        return c.model_copy(
            update={
                "title": title,
                "slug": slug,
                "subtitle": subtitle or None,
                "description": description or None,
                "difficulty": difficulty.lower() or None,
                "status": status or None,
                "tags": tags,
                "topics": topics,
                "combined_text": combined_text,
            }
        )
    except Exception:
        # Fallback para schemas Pydantic v1 ou modelos não-imutáveis
        try:
            c.title = (c.title or "").strip()
            c.slug = (c.slug or "").strip()
            c.description = (c.description or "").strip()
            if hasattr(c, "subtitle"):
                c.subtitle = (getattr(c, "subtitle", "") or "").strip() or None
            if c.difficulty:
                c.difficulty = (c.difficulty or "").strip().lower() or None
            if c.status:
                c.status = (c.status or "").strip() or None
            if not isinstance(c.tags, list):
                c.tags = list(c.tags or [])
            if not isinstance(c.topics, list):
                c.topics = list(c.topics or [])
            return c
        except Exception:
            # Retorna candidato original se todas as tentativas falharem
            return c


# -----------------------------
# Filtros por Status
# -----------------------------
def _norm_status(s: Optional[str]) -> Optional[str]:
    """
    Normaliza string de status removendo espaços e convertendo para lowercase.
    
    Args:
        s: String de status a ser normalizada
        
    Returns:
        String normalizada ou None se entrada for None
    """
    if s is None:
        return None
    return s.strip().casefold()


def filter_by_status(candidates: List[TrailCandidate], allowed_status: Tuple[str, ...]) -> List[TrailCandidate]:
    """
    Filtra candidatos mantendo apenas aqueles com status permitido.
    
    Args:
        candidates: Lista de TrailCandidate para filtrar
        allowed_status: Tupla com status permitidos (case-insensitive)
        
    Returns:
        Lista filtrada de candidatos com status permitido
    """
    allowed_norm = {(_norm_status(s) or "") for s in (allowed_status or ())}
    out: List[TrailCandidate] = []
    for c in candidates:
        st = _norm_status(getattr(c, "status", None))
        if st and st in allowed_norm:
            out.append(c)
    if not out:
        logger.warning("Normalizer: nenhum candidato restante após filtro de status (%s).", allowed_status)
    return out


# -----------------------------
# Sistema de Deduplicação e Completude
# -----------------------------
def _completeness_score(c: TrailCandidate) -> int:
    """
    Calcula heurística de completude do candidato para deduplicação.
    
    Critérios de pontuação:
    - +1 para slug presente
    - +1 para título presente
    - +1 para dificuldade definida
    - +1 para descrição presente
    - +1 se houver pelo menos 1 tag
    - +1 se houver combined_text
    
    Args:
        c: TrailCandidate a ser avaliado
        
    Returns:
        Score de completude (0-6)
    """
    score = 0
    if getattr(c, "slug", None):
        score += 1
    if getattr(c, "title", None):
        score += 1
    if getattr(c, "difficulty", None):
        score += 1
    if getattr(c, "description", None):
        score += 1
    tags = getattr(c, "tags", None) or []
    if isinstance(tags, list) and len(tags) > 0:
        score += 1
    if getattr(c, "combined_text", None):
        score += 1
    return score


def _public_id_as_uuid(pid) -> Optional[UUID]:
    """
    Converte publicId para UUID quando possível.
    
    Args:
        pid: publicId a ser convertido (pode ser UUID, string ou outro tipo)
        
    Returns:
        UUID convertido ou None se conversão não for possível
    """
    if isinstance(pid, UUID):
        return pid
    try:
        return UUID(str(pid))
    except Exception:
        return None


def dedupe_by_public_id(candidates: List[TrailCandidate]) -> List[TrailCandidate]:
    """
    Remove duplicatas por publicId mantendo o candidato de maior completude.
    
    Em caso de empate na completude, mantém o primeiro candidato encontrado
    para garantir ordem estável. Retorna lista ordenada por título, slug e publicId.
    
    Args:
        candidates: Lista de TrailCandidate para deduplicar
        
    Returns:
        Lista deduplicada e ordenada de candidatos
    """
    best_by_id: Dict[str, TrailCandidate] = {}
    for c in candidates:
        pid_raw = getattr(c, "publicId", None)
        if pid_raw is None:
            logger.warning(
                "Normalizer: candidato sem publicId descartado (slug=%s, title=%s)",
                getattr(c, "slug", None),
                getattr(c, "title", None),
            )
            continue

        pid_uuid = _public_id_as_uuid(pid_raw)
        key = str(pid_uuid) if pid_uuid is not None else str(pid_raw)

        prev = best_by_id.get(key)
        if prev is None:
            best_by_id[key] = c
        else:
            if _completeness_score(c) > _completeness_score(prev):
                best_by_id[key] = c

    deduped = list(best_by_id.values())

    def _sort_key(x: TrailCandidate):
        title = (getattr(x, "title", "") or "").casefold()
        slug = (getattr(x, "slug", "") or "").casefold()
        pid_raw = getattr(x, "publicId", None)
        pid_uuid = _public_id_as_uuid(pid_raw)
        pid_str = str(pid_uuid) if pid_uuid is not None else str(pid_raw or "")
        return (title, slug, pid_str)

    deduped.sort(key=_sort_key)
    return deduped


# -----------------------------
# Construção de Texto Combinado
# -----------------------------
def build_combined_text(c: TrailCandidate) -> str:
    """
    Constrói texto combinado consistente com o pipeline de busca densa/BM25.
    
    Utilizado quando o campo combined_text está vazio, criando um texto
    padronizado a partir dos campos principais da trilha.
    
    Args:
        c: TrailCandidate para construir o texto
        
    Returns:
        String com texto combinado dos campos principais
    """
    parts = [
        c.title or "",
        getattr(c, "subtitle", None) or "",
        c.description or "",
        " ".join(c.topics or []),
        " ".join(c.tags or []),
        c.combined_text or "",
    ]
    return " | ".join(p for p in parts if p)


def fill_missing_combined_text(candidates: List[TrailCandidate]) -> List[TrailCandidate]:
    """
    Preenche combined_text ausente com texto canônico construído a partir dos campos principais.
    
    Útil para catálogos legados onde o backend ainda não preenche o campo combined_text.
    Cria uma nova lista com candidatos atualizados.
    
    Args:
        candidates: Lista de TrailCandidate para processar
        
    Returns:
        Nova lista com combined_text preenchido onde necessário
    """
    out: List[TrailCandidate] = []
    changed = 0
    for c in candidates:
        if (c.combined_text or "").strip():
            out.append(c)
            continue
        new_ct = build_combined_text(c)
        try:
            out.append(c.model_copy(update={"combined_text": new_ct}))
        except Exception:
            c.combined_text = new_ct  # fallback para modelos não-imutáveis
            out.append(c)
        changed += 1
    if changed:
        logger.info("Normalizer: preenchido combined_text em %d item(ns).", changed)
    return out
