# validators/trail_output_checks.py
"""
Regras de negócio do Sistema de Recomendação (CLI) — V4 / P1 Híbrido.
Aplica threshold de match por coleção, garante coerência de status/textos e normaliza o output.

Uso no fluxo:
1) Validar com schemas.TrailOutput
2) Chamar apply_business_rules(output, cfg, collection="trilhas") antes de retornar ao CLI
"""

from __future__ import annotations

from typing import Optional, List, Dict
from copy import deepcopy

from pydantic import ValidationError

# Schemas
from schemas.trail_output import TrailOutput, SuggestedTrail, WebFallback  # compat
from reco.config import RecoConfig


# ---------------------------------------------------------------------
# Defaults (fallback quando cfg não vier)
# ---------------------------------------------------------------------
DEFAULT_MATCH_THRESHOLD: float = 0.75
DEFAULT_MAX_SUGGESTIONS: int = 3
DEFAULT_FALLBACK_MESSAGE: str = (
    "No momento não encontrei trilhas publicadas que combinem com a sua dúvida. "
    "Você pode tentar reformular com uma palavra-chave (ex.: 'programação', 'carreira')."
)
DEFAULT_SHORT_ANSWER_EMPTY: str = "Ainda não encontrei trilhas publicadas que combinem com a sua dúvida."
DEFAULT_CTA_EMPTY: str = "Tentar de novo"
DEFAULT_WHY_MATCH: str = "Combina com o que você buscou."


# ---------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------
def _strip_text(s: Optional[str]) -> Optional[str]:
    return s.strip() if isinstance(s, str) else s


def _strip_text_fields(o: TrailOutput) -> None:
    """Higieniza campos de texto (trim simples)."""
    o.short_answer = _strip_text(o.short_answer)
    o.cta = _strip_text(o.cta)
    if o.mensagem_padrao is not None:
        o.mensagem_padrao = _strip_text(o.mensagem_padrao)

    if getattr(o, "suggested_trails", None):
        for s in o.suggested_trails or []:
            s.why_match = _strip_text(s.why_match) or ""


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _dedupe_keep_best(items: List[SuggestedTrail]) -> List[SuggestedTrail]:
    """Dedup por publicId (preferível) e, se ausente, por slug; mantém maior score."""
    best_by_key: Dict[str, SuggestedTrail] = {}
    for s in items:
        key = f"pid:{s.publicId}" if s.publicId is not None else (f"slug:{(s.slug or '').lower()}" or f"title:{(s.title or '').lower()}")
        prev = best_by_key.get(key)
        if (prev is None) or (float(s.match_score) > float(prev.match_score)):
            best_by_key[key] = s
    out = list(best_by_key.values())
    out.sort(key=lambda x: (-(float(x.match_score)), (x.title or "").casefold(), str(x.publicId)))
    return out


def _enforce_threshold(items: List[SuggestedTrail], threshold: float) -> List[SuggestedTrail]:
    return [s for s in items if float(s.match_score) >= threshold]


def _ensure_suggestions_list(o: TrailOutput) -> List[SuggestedTrail]:
    """Normaliza acesso às sugestões como lista (compat legada)."""
    suggestions = getattr(o, "suggested_trails", None)
    if suggestions is None and hasattr(o, "suggested_trail"):
        one = getattr(o, "suggested_trail")
        suggestions = [one] if one is not None else []
    return suggestions or []


def _write_suggestions_list(o: TrailOutput, items: Optional[List[SuggestedTrail]]) -> None:
    if hasattr(o, "suggested_trails"):
        o.suggested_trails = items if items else None
    if hasattr(o, "suggested_trail"):
        o.suggested_trail = None  # zera legado


def _fill_missing_why_match(items: List[SuggestedTrail]) -> None:
    """Garante why_match com >= 5 chars para cada sugestão."""
    for s in items:
        wm = (s.why_match or "").strip()
        if len(wm) < 5:
            s.why_match = DEFAULT_WHY_MATCH


def _ensure_status_coherence(
    o: TrailOutput,
    fallback_message: str,
    short_answer_empty: str,
    cta_empty: str,
) -> None:
    """
    Coerência entre status e payload:
    - status="ok": precisa ter >=1 sugestão (ou web_fallback, que não usamos aqui).
    - status="fora_do_escopo": zera sugestões, define textos padrão.
    """
    suggestions = _ensure_suggestions_list(o)
    has_trails = bool(suggestions)
    has_fallback = bool(getattr(o, "web_fallback", None))

    if o.status == "ok":
        if not has_trails and not has_fallback:
            o.status = "fora_do_escopo"
            o.mensagem_padrao = o.mensagem_padrao or fallback_message
            _write_suggestions_list(o, None)
            o.web_fallback = None
            o.short_answer = short_answer_empty
            o.cta = cta_empty
    elif o.status == "fora_do_escopo":
        _write_suggestions_list(o, None)
        o.web_fallback = None
        o.mensagem_padrao = o.mensagem_padrao or fallback_message
        o.short_answer = short_answer_empty
        o.cta = cta_empty


def _resolve_threshold(cfg: RecoConfig, collection: str) -> float:
    """
    Seleciona o threshold por coleção.
    - trilhas -> MATCH_THRESHOLD_TRILHAS (fallback para DEFAULT_MATCH_THRESHOLD)
    - vagas   -> MATCH_THRESHOLD_VAGAS (fallback para DEFAULT_MATCH_THRESHOLD)
    """
    if collection == "vagas":
        return float(getattr(cfg, "MATCH_THRESHOLD_VAGAS", DEFAULT_MATCH_THRESHOLD))
    return float(getattr(cfg, "MATCH_THRESHOLD_TRILHAS", DEFAULT_MATCH_THRESHOLD))


# ---------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------
def apply_business_rules(
    output: TrailOutput,
    cfg: Optional[RecoConfig] = None,
    *,
    collection: str = "trilhas",
) -> TrailOutput:
    """
    Aplica regras de negócio sobre um TrailOutput já validado pelo schema:
    - Limpeza/higienização de textos.
    - Threshold por coleção (≥ MATCH_THRESHOLD_*).
    - Dedup; ordenação; cap por MAX_SUGGESTIONS.
    - Clamp de match_score em [0,1] e why_match mínimo.
    - Coerência de status e textos.
    - Revalidação final (Pydantic).
    """
    cfg = cfg or RecoConfig()

    match_threshold = _resolve_threshold(cfg, collection)
    max_suggestions = int(getattr(cfg, "MAX_SUGGESTIONS", DEFAULT_MAX_SUGGESTIONS))
    fallback_message = DEFAULT_FALLBACK_MESSAGE
    short_answer_empty = DEFAULT_SHORT_ANSWER_EMPTY
    cta_empty = DEFAULT_CTA_EMPTY

    o = deepcopy(output)

    # 1) Higieniza textos
    _strip_text_fields(o)

    # 2) Normaliza/filtra sugestões (quando houver)
    suggestions = _ensure_suggestions_list(o)
    if suggestions:
        # Clamp de score
        for s in suggestions:
            s.match_score = _clamp01(float(s.match_score))

        # Threshold por item
        suggestions = _enforce_threshold(suggestions, match_threshold)

        # Dedup + ordenação
        suggestions = _dedupe_keep_best(suggestions)

        # Cap por MAX_SUGGESTIONS
        suggestions = suggestions[: max(1, min(max_suggestions, len(suggestions)))]

        # Why_match mínimo
        _fill_missing_why_match(suggestions)

        _write_suggestions_list(o, suggestions if suggestions else None)

    # 3) Coerência status vs payload
    _ensure_status_coherence(
        o,
        fallback_message=fallback_message,
        short_answer_empty=short_answer_empty,
        cta_empty=cta_empty,
    )

    # 4) Revalida contra o schema
    try:
        o = TrailOutput.model_validate(o.model_dump())
    except ValidationError as ve:
        raise ve

    return o
