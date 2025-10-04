# reco/ranker.py
"""
Sistema Inteligente de Ranking - Leve Agents

Este módulo implementa um sistema avançado de ranking que combina scores de diferentes
fontes (BM25 + MPNet) e aplica regras de negócio para ordenar e filtrar recomendações
de trilhas educacionais de forma otimizada e explicável.

Funcionalidades Principais:
- Combinação inteligente de scores híbridos (BM25 + MPNet) e formatos legados
- Aplicação de boosts explicáveis (título, descrição, tags, nível, relevância educacional)
- Normalização e limitação de scores (0..1)
- Thresholds específicos por tipo de conteúdo
- Deduplicação por publicId
- Fallback de dominância para garantir resultados
- Tokenização acento-insensível
- Compatibilidade com formatos legados e híbridos
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re
import unicodedata as _ud

from reco.config import RecoConfig
from schemas.trail_candidate import TrailCandidate


# Regex para identificação de palavras; cobre letras/dígitos em ASCII e Unicode
_WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+")


def _strip_accents(text: str) -> str:
    """
    Remove acentos de uma string usando normalização Unicode NFKD.
    
    Args:
        text: String a ser processada
        
    Returns:
        String sem acentos
    """
    if not text:
        return ""
    return "".join(ch for ch in _ud.normalize("NFKD", text) if not _ud.combining(ch))


def _tokenize(text: str) -> List[str]:
    """
    Tokenização simples, acento-insensível e case-insensível.
    
    Args:
        text: Texto a ser tokenizado
        
    Returns:
        Lista de tokens normalizados
    """
    if not text:
        return []
    norm = _strip_accents(text).casefold()
    return [m.group(0) for m in _WORD_RE.finditer(norm)]


@dataclass
class ScoredCandidate:
    candidate: TrailCandidate
    match_score: float
    applied_boosts: List[str] = field(default_factory=list)
    # Campos extras (debug/observabilidade)
    base_score: Optional[float] = None
    score_semantic: Optional[float] = None
    score_bm25: Optional[float] = None
    score_combined: Optional[float] = None


# -----------------------------
# Sistema de Boosts de Negócio
# -----------------------------
def _has_title_desc_keyword(cand: TrailCandidate, query_tokens: Iterable[str]) -> bool:
    """
    Verifica se alguma palavra-chave da consulta aparece no título/descrição/conteúdo.
    
    Args:
        cand: Candidato a ser verificado
        query_tokens: Tokens da consulta
        
    Returns:
        True se houver match de palavras-chave
    """
    q = {t for t in query_tokens if len(t) > 2}
    if not q:
        return False
    hay = f"{cand.title or ''} | {cand.description or ''} | {cand.combined_text or ''}"
    hay_norm = _strip_accents(hay).casefold()
    return any(tok in hay_norm for tok in q)


def _has_educational_relevance(cand: TrailCandidate, query_tokens: Iterable[str]) -> bool:
    """
    Verifica se a trilha tem relevância educacional baseada em palavras-chave.
    
    Args:
        cand: Candidato a ser verificado
        query_tokens: Tokens da consulta
        
    Returns:
        True se a trilha tem relevância educacional
    """
    educational_keywords = [
        'aprender', 'estudar', 'curso', 'trilha', 'carreira', 'profissão',
        'trabalho', 'habilidade', 'competência', 'desenvolvimento',
        'programação', 'tecnologia', 'negócio', 'gestão', 'marketing',
        'design', 'vendas', 'atendimento', 'líder', 'equipe', 'projeto'
    ]
    
    # Verifica se a query tem palavras-chave educacionais
    query_lower = ' '.join(query_tokens).lower()
    has_educational_query = any(keyword in query_lower for keyword in educational_keywords)
    
    if not has_educational_query:
        return True  # Se a query não é educacional, não penaliza
    
    # Verifica se a trilha tem conteúdo educacional
    content = f"{cand.title or ''} | {cand.description or ''} | {cand.combined_text or ''}"
    content_lower = _strip_accents(content).casefold()
    
    return any(keyword in content_lower for keyword in educational_keywords)


def _apply_boosts(
    base_score: float,
    cand: TrailCandidate,
    query_tokens: Iterable[str],
    cfg: RecoConfig,
) -> Tuple[float, List[str]]:
    """
    Aplica boosts de negócio baseados em critérios específicos.
    
    Regras de Boost:
    - TITLE_DESC_BOOST: se palavras-chave aparecem no título/descrição/conteúdo
    - TAG_BOOST: se tokens de tags aparecem na consulta
    - BEGINNER_BOOST: se difficulty == Beginner (ajuda fase de descoberta)
    - EDUCATIONAL_RELEVANCE_BOOST: se trilha tem relevância educacional
    
    Args:
        base_score: Score base antes dos boosts
        cand: Candidato a ser processado
        query_tokens: Tokens da consulta
        cfg: Configuração do sistema
        
    Returns:
        Tupla com (score_final, lista_de_boosts_aplicados)
    """
    boosts_applied: List[str] = []
    score = float(base_score)

    # Boost 1: Título/Descrição - palavras-chave da consulta no conteúdo
    if _has_title_desc_keyword(cand, query_tokens):
        score += cfg.TITLE_DESC_BOOST
        boosts_applied.append("title_desc_match")

    # Boost 2: Tags/Temas - tokens de tags aparecem na consulta
    tag_tokens: List[str] = []
    for t in cand.tags or []:
        tag_tokens.extend(_tokenize(str(t)))
    qset = set(query_tokens)
    if tag_tokens and qset.intersection(tag_tokens):
        score += cfg.TAG_BOOST
        boosts_applied.append("tag_match")

    # Boost 3: Nível Iniciante - ajuda na fase de descoberta
    if (cand.difficulty or "").lower() == "beginner":
        score += cfg.BEGINNER_BOOST
        boosts_applied.append("beginner_boost")

    # Boost 4: Relevância Educacional - trilha tem conteúdo educacional
    if _has_educational_relevance(cand, query_tokens):
        score += cfg.EDUCATIONAL_RELEVANCE_BOOST
        boosts_applied.append("educational_relevance")

    # Aplicação de limites de score
    if score > cfg.SCORE_CAP:
        score = cfg.SCORE_CAP
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return score, boosts_applied


# -----------------------------
# Sistema de Entrada Flexível (Legado e Híbrido)
# -----------------------------
# Formato legado: List[Tuple[TrailCandidate, float]]
LegacyInput = List[Tuple[TrailCandidate, float]]

# Formato híbrido: List[Dict[str, Any]] com estrutura:
# {
#   "candidate": TrailCandidate,
#   "score_combined": float,   # opcional
#   "score_semantic": float,   # opcional
#   "score_bm25": float,       # opcional
# }
HybridInputItem = Dict[str, Any]
HybridInput = List[HybridInputItem]

RankInput = Union[LegacyInput, HybridInput]


def _coerce_inputs(
    items: RankInput,
) -> List[Tuple[TrailCandidate, Optional[float], Optional[float], Optional[float]]]:
    """
    Converte entrada (legado ou híbrido) para formato homogêneo.
    
    Args:
        items: Lista de entrada em formato legado ou híbrido
        
    Returns:
        Lista homogênea de tuplas: (candidate, score_combined, score_semantic, score_bm25)
    """
    out: List[Tuple[TrailCandidate, Optional[float], Optional[float], Optional[float]]] = []
    if not items:
        return out

    # Caso 1: legado
    if isinstance(items, list) and items and isinstance(items[0], tuple):
        for cand, content_score in items:  # type: ignore
            out.append((cand, None, None, float(content_score)))
        return out

    # Caso 2: híbrido (dicts com chaves conhecidas)
    for it in items:  # type: ignore
        cand = it.get("candidate") or it.get("item") or it.get("trail")  # tolerante a nomes
        if not isinstance(cand, TrailCandidate):
            continue
        out.append(
            (
                cand,
                _safe_float(it.get("score_combined")),
                _safe_float(it.get("score_semantic")),
                _safe_float(it.get("score_bm25")),
            )
        )
    return out


def _safe_float(x: Any) -> Optional[float]:
    """
    Converte valor para float de forma segura.
    
    Args:
        x: Valor a ser convertido
        
    Returns:
        Float convertido ou None se conversão falhar
    """
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


# -----------------------------
# Núcleo do Sistema de Ranking
# -----------------------------
def rank(
    scored_candidates: RankInput,
    query_text: str,
    cfg: RecoConfig,
    *,
    collection: str = "trilhas",
    max_results: Optional[int] = None,
) -> List[ScoredCandidate]:
    """
    Aplica sistema completo de ranking com boosts, threshold, ordenação e deduplicação.
    
    Args:
        scored_candidates: Lista de candidatos (formato legado ou híbrido)
        query_text: Texto da consulta original
        cfg: Configuração do sistema
        collection: Tipo de coleção ("trilhas" ou "vagas") - define threshold
        max_results: Número máximo de resultados (usa padrão se None)
        
    Returns:
        Lista de ScoredCandidate ranqueados e filtrados
    """
    coerced = _coerce_inputs(scored_candidates)
    if not coerced:
        return []

    # Define threshold baseado no tipo de coleção
    if collection == "vagas":
        threshold = cfg.MATCH_THRESHOLD_VAGAS
    else:
        threshold = cfg.MATCH_THRESHOLD_TRILHAS

    max_n = max_results if isinstance(max_results, int) and max_results > 0 else cfg.MAX_SUGGESTIONS
    q_tokens = _tokenize(query_text)

    # Aplicação de boosts e enriquecimento dos candidatos
    enriched: List[ScoredCandidate] = []
    for cand, score_combined, score_sem, score_bm25 in coerced:
        # Score base: prioriza híbrido sobre legado
        base_score = score_combined
        if base_score is None:
            # Formato legado: content_score estava no slot "bm25"
            base_score = score_bm25 if score_bm25 is not None else 0.0

        final_score, boosts = _apply_boosts(base_score, cand, q_tokens, cfg)
        enriched.append(
            ScoredCandidate(
                candidate=cand,
                match_score=final_score,
                applied_boosts=boosts,
                base_score=base_score,
                score_semantic=score_sem,
                score_bm25=score_bm25,
                score_combined=score_combined,
            )
        )

    # Filtro por threshold
    filtered = [sc for sc in enriched if sc.match_score >= threshold]

    # Função de deduplicação por publicId (mantém maior score)
    def _dedup(items: List[ScoredCandidate]) -> List[ScoredCandidate]:
        best_by_id: Dict[str, ScoredCandidate] = {}
        for sc in items:
            pid = str(sc.candidate.publicId)
            prev = best_by_id.get(pid)
            if (prev is None) or (sc.match_score > prev.match_score):
                best_by_id[pid] = sc
        out = list(best_by_id.values())
        out.sort(key=lambda x: (-(x.match_score), (x.candidate.title or "").casefold(), str(x.candidate.publicId)))
        return out

    # Fallback de dominância se nenhum resultado passar no threshold
    if not filtered:
        dedup_all = _dedup(enriched)
        if dedup_all and dedup_all[0].match_score >= cfg.DOMINANCE_MIN_ACCEPT:
            return dedup_all[:1]
        return []

    # Deduplicação e retorno dos resultados finais
    deduped = _dedup(filtered)
    return deduped[:max_n]
