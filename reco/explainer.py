# reco/explainer.py
"""
Gerador de Explicações - Leve Agents

Gera explicações claras e objetivas sobre por que uma trilha foi recomendada.
Cria textos explicativos baseados em sinais identificáveis e relevantes para o usuário.

Funcionalidades:
- Geração de explicações em tom cordial e jovem
- Identificação de palavras-âncora em títulos e descrições
- Matching acento-insensível para melhor cobertura
- Baseado em sinais explicáveis (tags, tema, nível, conteúdo)
- Ajuste de confiança baseado no score combinado
- Fallback inteligente quando não há tags disponíveis
"""

from __future__ import annotations

from typing import Iterable, List, Optional
import re
import unicodedata as _ud

from reco.ranker import ScoredCandidate
from schemas.trail_candidate import TrailCandidate


_WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+")


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    return "".join(ch for ch in _ud.normalize("NFKD", text) if not _ud.combining(ch))


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    norm = _strip_accents(text).casefold()
    return [m.group(0) for m in _WORD_RE.finditer(norm)]


def _best_tag_match(cand: TrailCandidate, query_tokens: Iterable[str]) -> Optional[str]:
    """
    Retorna uma tag representativa do candidato que apareceu na consulta (se houver).
    A comparação é acento-insensível; o retorno preserva a grafia original.
    """
    qset = set(query_tokens)
    for t in (cand.tags or []):
        toks = _tokenize(str(t))
        if set(toks) & qset:
            return str(t)
    return None


def _best_title_desc_keyword(cand: TrailCandidate, query_tokens: Iterable[str]) -> Optional[str]:
    """
    Se não houve tag correspondente, tenta achar uma palavra-âncora no título/descrição/combined_text
    que também esteja na consulta. Retorna a forma original do título/descrição quando possível.
    """
    if not query_tokens:
        return None
    # Texto normalizado para busca
    hay_raw = f"{cand.title or ''} | {cand.description or ''} | {cand.combined_text or ''}"
    hay_norm = _strip_accents(hay_raw).casefold()
    # Ordena tokens por tamanho para priorizar âncoras mais “ricas”
    for tok in sorted(set(t for t in query_tokens if len(t) > 2), key=len, reverse=True):
        if tok in hay_norm:
            # tenta recuperar a palavra original (best-effort)
            # varre palavras do título/descrição buscando uma que normalize igual a tok
            for m in _WORD_RE.finditer(hay_raw):
                w = m.group(0)
                if _strip_accents(w).casefold() == tok:
                    return w
            return tok
    return None


def _content_cues(cand: TrailCandidate) -> List[str]:
    """
    Infere 'cues' (pistas de formato/estilo) a partir da descrição/combined_text.
    Ex.: aulas curtas, conteúdos práticos, tem vídeos, tem quizzes.
    """
    text_raw = f"{cand.description or ''} | {cand.combined_text or ''}"
    text = _strip_accents(text_raw).casefold()

    cues: List[str] = []
    if "curt" in text:  # cobre 'curto', 'curtas'
        cues.append("aulas curtas")
    if "exerc" in text or "pratic" in text:  # 'exercícios', 'prática'
        cues.append("conteúdos práticos")
    if "video" in text:
        cues.append("tem vídeos")
    if "quiz" in text:
        cues.append("tem quizzes")

    # Dedup e limite a 2
    seen = set()
    out: List[str] = []
    for c in cues:
        if c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= 2:
            break
    return out


def _confidence_suffix(scored: ScoredCandidate) -> Optional[str]:
    """
    Adiciona um leve reforço de confiança quando o score combinado é alto.
    Evita termos técnicos; apenas 'bem alinhado' quando (aprox.) >= 0.85.
    """
    s = scored.score_combined if scored.score_combined is not None else scored.match_score
    try:
        if s is not None and float(s) >= 0.85:
            return " — bem alinhado ao que você quer"
    except Exception:
        pass
    return None


def make_reason(scored: ScoredCandidate, query_text: str) -> str:
    """
    Monta a frase final de why_match (≈ até 180 caracteres).
    Ex.: "Conecta com JavaScript e é nível iniciante — aulas curtas."
    """
    cand = scored.candidate
    q_tokens = _tokenize(query_text)

    parts: List[str] = []

    tag = _best_tag_match(cand, q_tokens)
    beginner = (cand.difficulty or "").lower() == "beginner"

    # Bloco principal (tema + nível)
    if tag and beginner:
        parts.append(f"Conecta com {tag} e é nível iniciante")
    elif tag:
        parts.append(f"Conecta com {tag}")
    else:
        # Sem tag: tenta palavra-âncora do título/descrição
        key = _best_title_desc_keyword(cand, q_tokens)
        if key and beginner:
            parts.append(f"Faz match com {key} e é nível iniciante")
        elif key:
            parts.append(f"Faz match com {key}")
        elif beginner:
            parts.append("Boa porta de entrada (nível iniciante)")
        else:
            parts.append("Combina com o que você buscou")

    # Cues de formato/estilo
    cues = _content_cues(cand)
    if cues:
        parts.append(" — " + ", ".join(cues))

    # Sufixo suave de confiança (quando score alto)
    conf = _confidence_suffix(scored)
    if conf:
        parts.append(conf)

    reason = ". ".join([p for p in parts if p]).replace(".  —", " —").replace(". —", " —")

    if len(reason) > 180:
        reason = reason[:177].rstrip() + "..."

    return reason
