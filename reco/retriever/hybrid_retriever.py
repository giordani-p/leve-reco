# reco/retriever/hybrid_retriever.py
"""
HybridRetriever — V4 / P1 Híbrido (BM25 + MPNet)

Responsável por:
- Consultar BM25 (indexer) e Denso (MPNet) em paralelo.
- Deduplicar resultados por id.
- Normalizar escores (min-max ou z-score) para mesma escala.
- Aplicar blending com pesos configuráveis (ex.: semantic 0.65 / bm25 0.35).
- Retornar Top-K final com score combinado e metadados.

Dependências internas:
- DenseRetriever (caminho denso)
- Uma função/callable BM25: bm25_search_topk(query_text, k) -> List[Tuple[id, score_bm25, metadata]]

Observações:
- Filtros por metadados (ex.: status="Published") são aplicados antes do blending.
- O threshold NÃO é aplicado aqui; fica a cargo do ranker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np

from reco.retriever.dense_retriever import DenseRetriever, DenseResult


@dataclass
class HybridResult:
    id: str
    score_combined: float
    score_semantic: float
    score_bm25: float
    metadata: Dict[str, Any]


def _minmax(values: np.ndarray, eps: float) -> np.ndarray:
    if values.size == 0:
        return values
    vmin, vmax = float(values.min()), float(values.max())
    if math.isclose(vmax, vmin, rel_tol=0.0, abs_tol=eps):
        return np.zeros_like(values, dtype=np.float32)
    
    # Normalização MinMax com clamp para evitar scores muito altos
    normalized = (values - vmin) / (vmax - vmin + eps)
    
    # Clamp para [0, 1] para evitar valores extremos
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)


def _percentile_normalize(values: np.ndarray, eps: float) -> np.ndarray:
    """
    Normalização baseada em percentis para ser mais robusta a outliers.
    Usa percentil 10 e 90 em vez de min/max, com fallback para min/max.
    """
    if values.size == 0:
        return values
    
    # Usar percentis 10 e 90 para ser mais robusto a outliers
    p10, p90 = np.percentile(values, [10, 90])
    
    # Se os percentis são muito próximos, usar min/max
    if math.isclose(p90, p10, rel_tol=0.0, abs_tol=eps):
        vmin, vmax = float(values.min()), float(values.max())
        if math.isclose(vmax, vmin, rel_tol=0.0, abs_tol=eps):
            return np.zeros_like(values, dtype=np.float32)
        normalized = (values - vmin) / (vmax - vmin + eps)
    else:
        # Normalizar usando percentis
        normalized = (values - p10) / (p90 - p10 + eps)
    
    # Clamp para [0, 1]
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)


def _robust_normalize(values: np.ndarray, eps: float) -> np.ndarray:
    """
    Normalização robusta com clamp agressivo para evitar scores muito altos.
    """
    if values.size == 0:
        return values
    
    vmin, vmax = float(values.min()), float(values.max())
    if math.isclose(vmax, vmin, rel_tol=0.0, abs_tol=eps):
        return np.zeros_like(values, dtype=np.float32)
    
    # Normalização padrão
    normalized = (values - vmin) / (vmax - vmin + eps)
    
    # Clamp agressivo para evitar scores muito altos
    # Limitar a 0.8 para deixar espaço para boosts
    normalized = np.clip(normalized, 0.0, 0.8)
    
    return normalized.astype(np.float32)


def _zscore(values: np.ndarray, eps: float) -> np.ndarray:
    if values.size == 0:
        return values
    mean = float(values.mean())
    std = float(values.std())
    if math.isclose(std, 0.0, rel_tol=0.0, abs_tol=eps):
        return np.zeros_like(values, dtype=np.float32)
    z = (values - mean) / (std + eps)
    # opcional: mapear z para [0,1] de forma estável
    # aqui usamos CDF aproximada via erf para suavizar extremos
    return (0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))).astype(np.float32)


class HybridRetriever:
    """
    Uso:
        hybrid = HybridRetriever.from_config(
            cfg,
            dense_retriever=dense,
            bm25_search_topk=indexer.search_topk,  # callable (query_text, k) -> [(id, score_bm25, meta)]
        )
        results = hybrid.search(query_text, k=50, filters={"status": "Published"})
    """

    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever,
        bm25_search_topk: Callable[[str, int], List[Tuple[str, float, Dict[str, Any]]]],
        weights: Dict[str, float],
        normalization: str,
        normalization_eps: float,
        top_k_default: int,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_search_topk
        self._w_sem = float(weights.get("semantic", 0.65))
        self._w_bm25 = float(weights.get("bm25", 0.35))
        self._norm = normalization  # "minmax" | "zscore"
        self._eps = float(normalization_eps)
        self._k_default = int(top_k_default)

    # --------- fábrica ---------
    @classmethod
    def from_config(
        cls,
        cfg,
        *,
        dense_retriever: DenseRetriever,
        bm25_search_topk: Callable[[str, int], List[Tuple[str, float, Dict[str, Any]]]],
    ) -> "HybridRetriever":
        return cls(
            dense_retriever=dense_retriever,
            bm25_search_topk=bm25_search_topk,
            weights=getattr(cfg, "WEIGHTS", {"semantic": 0.65, "bm25": 0.35}),
            normalization=getattr(cfg, "NORMALIZATION", "minmax"),
            normalization_eps=getattr(cfg, "NORMALIZATION_EPS", 1e-6),
            top_k_default=getattr(cfg, "TOP_K_DEFAULT", 50),
        )

    # --------- API ---------
    def search(
        self,
        query_text: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[HybridResult]:
        """
        Retorna Top-K híbrido:
          - Deduplica ids vindos de ambos os caminhos.
          - Normaliza escores por caminho.
          - Aplica blending configurado.
        """
        k_eff = int(k or self._k_default)
        if not query_text or k_eff <= 0:
            return []

        # 1) Buscar em ambos os caminhos
        dense_res: List[DenseResult] = self._dense.search(query_text, k=k_eff, filters=filters or None)
        bm25_raw: List[Tuple[str, float, Dict[str, Any]]] = self._bm25(query_text, k_eff)

        # 2) Aplicar filtros (se bm25 não aplicar internamente)
        if filters:
            bm25_raw = [
                (id_, sc, meta) for (id_, sc, meta) in bm25_raw
                if self._match_filters(meta or {}, filters)
            ]

        # 3) Deduplicar ids e montar tabelas de lookup
        ids_dense = [r.id for r in dense_res]
        ids_bm25 = [r[0] for r in bm25_raw]
        all_ids = list(dict.fromkeys(ids_dense + ids_bm25))  # preserva ordem de aparição

        sem_by_id: Dict[str, float] = {r.id: r.score_semantic for r in dense_res}
        bm25_by_id: Dict[str, float] = {id_: sc for (id_, sc, _) in bm25_raw}
        meta_by_id: Dict[str, Dict[str, Any]] = {r.id: r.metadata for r in dense_res}
        for (id_, _, meta) in bm25_raw:
            if id_ not in meta_by_id:
                meta_by_id[id_] = meta or {}

        # 4) Vetores para normalização
        sem_scores = np.array([sem_by_id.get(i, 0.0) for i in all_ids], dtype=np.float32)
        bm25_scores = np.array([bm25_by_id.get(i, 0.0) for i in all_ids], dtype=np.float32)

        # 5) Normalização
        if self._norm == "zscore":
            sem_n = _zscore(sem_scores, self._eps)
            bm25_n = _zscore(bm25_scores, self._eps)
        elif self._norm == "percentile":
            sem_n = _percentile_normalize(sem_scores, self._eps)
            bm25_n = _percentile_normalize(bm25_scores, self._eps)
        elif self._norm == "robust":
            sem_n = _robust_normalize(sem_scores, self._eps)
            bm25_n = _robust_normalize(bm25_scores, self._eps)
        else:
            sem_n = _minmax(sem_scores, self._eps)
            bm25_n = _minmax(bm25_scores, self._eps)

        # 6) Blending
        combined = (self._w_sem * sem_n) + (self._w_bm25 * bm25_n)

        # 7) Ordenar e cortar Top-K
        order = np.argsort(-combined)[:k_eff]
        results: List[HybridResult] = []
        for idx in order:
            id_ = all_ids[idx]
            results.append(
                HybridResult(
                    id=id_,
                    score_combined=float(combined[idx]),
                    score_semantic=float(sem_scores[idx]),
                    score_bm25=float(bm25_scores[idx]),
                    metadata=meta_by_id.get(id_, {}),
                )
            )
        return results

    # --------- util ---------
    @staticmethod
    def _match_filters(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            val = meta.get(key)
            if isinstance(expected, (list, tuple, set)):
                if val not in expected:
                    return False
            else:
                if val != expected:
                    return False
        return True
