# reco/indexer.py
"""
Indexer (BM25/TF-IDF) — V4 / P1 Híbrido

Responsável pelo caminho de busca por texto (BM25 ou TF-IDF como fallback).
Interface:
  - fit_items(candidates: List[TrailCandidate]) -> None
  - search_topk(query_text: str, k: int) -> List[Tuple[str, float, Dict[str, Any]]]

Notas:
- Expansão leve de sinônimos **apenas** neste caminho (ver config.USE_QUERY_SYNONYMS_BM25).
- O caminho denso (MPNet) é tratado por EmbeddingProvider + VectorIndex + DenseRetriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re
import unicodedata as _ud

import numpy as np

try:
    # BM25 opcional (melhor para busca lexical). Se não houver, caímos para TF-IDF.
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _HAS_SK = True
except Exception:
    _HAS_SK = False

from reco.config import RecoConfig
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


def _canonical_doc(c: TrailCandidate) -> str:
    # Mesmos campos que você usa para combined_text (evita “duas visões” do item)
    parts = [
        c.title or "",
        c.subtitle or "",
        c.description or "",
        " ".join(c.topics or []),
        " ".join(c.tags or []),
        c.combined_text or "",
    ]
    return " | ".join(p for p in parts if p)


@dataclass
class _Doc:
    id: str
    text: str
    meta: Dict[str, str]


class Indexer:
    """
    Indexador lexical para o caminho BM25/TF-IDF do híbrido.
    """

    def __init__(self, cfg: RecoConfig) -> None:
        self.cfg = cfg
        self._docs: List[_Doc] = []

        # BM25
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[List[str]] = []

        # TF-IDF fallback
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[np.ndarray] = None

    # -----------------------------
    # Público
    # -----------------------------
    def fit_items(self, candidates: List[TrailCandidate]) -> None:
        """Prepara o índice lexical a partir dos candidatos publicados."""
        self._docs = []
        for c in candidates:
            text = _canonical_doc(c)
            if not text.strip():
                continue
            meta = {
                "status": c.status or "",
                "difficulty": (c.difficulty or "").lower(),
                "area": c.area or "",
            }
            self._docs.append(_Doc(id=str(c.publicId), text=text, meta=meta))

        if not self._docs:
            # Sem documentos; zera estruturas
            self._bm25 = None
            self._bm25_corpus = []
            self._tfidf = None
            self._tfidf_matrix = None
            return

        # BM25 se disponível; senão TF-IDF
        if _HAS_BM25:
            corpus_tokens = [_tokenize(d.text) for d in self._docs]
            self._bm25 = BM25Okapi(corpus_tokens)
            self._bm25_corpus = corpus_tokens
            self._tfidf = None
            self._tfidf_matrix = None
        else:
            if not _HAS_SK:
                raise RuntimeError(
                    "Nem rank_bm25 nem scikit-learn disponíveis. "
                    "Instale um deles: pip install rank-bm25  OU  pip install scikit-learn"
                )
            self._tfidf = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=1,
                max_features=None,
                sublinear_tf=True,
            )
            self._tfidf_matrix = self._tfidf.fit_transform([d.text for d in self._docs]).astype(np.float32)
            self._bm25 = None
            self._bm25_corpus = []

    def search_topk(self, query_text: str, k: int) -> List[Tuple[str, float, Dict[str, str]]]:
        """
        Retorna [(id, score, meta)] ordenado por score desc.
        - Aplica expansão leve de sinônimos se USE_QUERY_SYNONYMS_BM25=True.
        - Não aplica filtros/threshold (isso é papel do HybridRetriever/Ranker).
        """
        if not query_text or not self._docs:
            return []

        q_text = self._expand_query_if_needed(query_text)

        if self._bm25 is not None:
            q_tokens = _tokenize(q_text)
            scores = np.array(self._bm25.get_scores(q_tokens), dtype=np.float32)  # shape: (N,)
        else:
            assert self._tfidf is not None and self._tfidf_matrix is not None
            q_vec = self._tfidf.transform([q_text])
            sim = cosine_similarity(self._tfidf_matrix, q_vec)  # (N, 1)
            scores = sim[:, 0].astype(np.float32)

        k = max(1, min(k, len(self._docs)))
        # Top-K mais altos
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        results: List[Tuple[str, float, Dict[str, str]]] = []
        for i in idx:
            d = self._docs[int(i)]
            results.append((d.id, float(scores[int(i)]), d.meta))
        return results

    # -----------------------------
    # Internos
    # -----------------------------
    def _expand_query_if_needed(self, query_text: str) -> str:
        """Expande leve a consulta com sinônimos apenas no caminho BM25/TF-IDF."""
        if not getattr(self.cfg, "USE_QUERY_SYNONYMS_BM25", True):
            return query_text

        synonyms = getattr(self.cfg, "QUERY_SYNONYMS", {}) or {}
        tokens = _tokenize(query_text)
        extra: List[str] = []
        for t in tokens:
            # chaves da tabela já estão em minúsculas sem acento
            if t in synonyms:
                extra.extend(synonyms[t])
        if not extra:
            return query_text
        # evita poluir demais; limite 6 termos extras
        extra = list(dict.fromkeys(extra))[:6]
        return query_text + " " + " ".join(extra)
