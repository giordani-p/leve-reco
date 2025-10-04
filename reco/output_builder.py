# reco/output_builder.py
"""
Constrói o objeto TrailOutput a partir dos candidatos ranqueados.
- 1..N sugestões => status="ok", suggested_trails preenchido.
- 0 sugestões   => status="fora_do_escopo", mensagem padrão e nenhuma sugestão.

Aprimoramentos (P1):
- Pluralização da short_answer conforme a quantidade encontrada.
- Clamp defensivo de 'max_results' (≥1).
- Higienização extra em reasons_by_id.
- Robustez quando publicId vier vazio.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import re

from schemas.trail_output import (
    TrailOutput,
    SuggestedTrail,
    QueryUnderstanding,
)
from reco.ranker import ScoredCandidate


_WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+")
DEFAULT_EMPTY_MESSAGE = (
    "No momento não encontrei trilhas publicadas que combinem com a sua dúvida. "
    "Você pode tentar reformular com uma palavra-chave (ex.: 'programação', 'carreira')."
)


# ----------------------------
# Helpers
# ----------------------------
def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    # casefold para acento/maiúsculas; limita tokens muito curtos
    return [m.group(0).casefold() for m in _WORD_RE.finditer(text) if len(m.group(0)) >= 2]


def _infer_theme(query_text: str) -> str:
    """
    Heurística simples para tema principal: pega a palavra 'significativa' mais longa.
    Fica legível para logs/telemetria (QueryUnderstanding.tema).
    """
    tokens = _tokenize(query_text)
    if not tokens:
        return "não informado"
    tokens_sorted = sorted(tokens, key=len, reverse=True)
    for t in tokens_sorted:
        if len(t) >= 4:
            return t
    return tokens_sorted[0]


def _keywords(query_text: str, limit: int = 5, max_token_len: int = 24) -> List[str]:
    """
    Extrai até 'limit' palavras-chave distintas da consulta, em ordem de ocorrência.
    Limita o tamanho de cada token para evitar ruído em telemetria/UI.
    """
    seen = set()
    out: List[str] = []
    for tok in _tokenize(query_text):
        if tok not in seen:
            seen.add(tok)
            out.append(tok[:max_token_len])
        if len(out) >= limit:
            break
    return out


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _safe_text(s: Optional[str], fallback: str = "") -> str:
    s = (s or "").strip()
    return s if s else fallback


def _safe_reason(reason: Optional[str]) -> str:
    r = (reason or "").strip()
    return r if len(r) >= 5 else "Combina com o que você buscou."


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


# ----------------------------
# Builder principal
# ----------------------------
def build_output(
    ranked: List[ScoredCandidate],
    query_text: str,
    max_results: int = 3,
    reasons_by_id: Optional[Dict[str, str]] = None,
) -> TrailOutput:
    """
    Cria o TrailOutput final (sem aplicar o validator de regras — isso é feito fora).

    Parâmetros:
      - ranked: lista ranqueada de ScoredCandidate (já com match_score final).
      - query_text: pergunta do jovem (+ pistas), usada para entendimento/telemetria.
      - max_results: limite superior de itens a exibir (1..3).
      - reasons_by_id: dicionário {publicId(str) -> why_match(str)}.
    """
    reasons_by_id = {str(k): (v or "").strip() for k, v in (reasons_by_id or {}).items()}
    theme = _infer_theme(query_text)
    kws = _keywords(query_text, limit=5)

    # Sem sugestões => fora_do_escopo
    if not ranked:
        return TrailOutput(
            status="fora_do_escopo",
            mensagem_padrao=DEFAULT_EMPTY_MESSAGE,
            short_answer="Ainda não encontrei trilhas publicadas que combinem com a sua dúvida.",
            suggested_trails=None,
            web_fallback=None,
            cta="Tentar de novo",
            query_understanding=QueryUnderstanding(tema=theme, palavras_chave=kws),
        )

    # Clamp defensivo de max_results
    nmax = max(1, int(max_results))
    n = min(nmax, len(ranked))
    topk = ranked[:n]

    suggestions: List[SuggestedTrail] = []
    for sc in topk:
        cand = sc.candidate
        pid_str = str(cand.publicId) if cand.publicId is not None else ""
        reason = _safe_reason(reasons_by_id.get(pid_str))

        suggestions.append(
            SuggestedTrail(
                publicId=cand.publicId,  # mantém tipo do schema
                slug=_safe_text(cand.slug, fallback=""),
                title=_safe_text(cand.title, fallback="Trilha"),
                description=_safe_text(cand.description, fallback=None),
                why_match=reason,
                match_score=_clamp01(float(sc.match_score)),
            )
        )

    # Short answer com pluralização
    n_found = len(suggestions)
    texto_opcoes = _plural(n_found, "1 opção", f"{n_found} opções")
    short_answer = f"Boa! Encontrei {texto_opcoes} que combinam com o que você buscou. Dá para começar pelo básico e ir testando o ritmo."

    return TrailOutput(
        status="ok",
        mensagem_padrao=None,
        short_answer=short_answer,
        suggested_trails=suggestions,
        web_fallback=None,
        cta="Começar trilha",
        query_understanding=QueryUnderstanding(tema=theme, palavras_chave=kws),
    )
