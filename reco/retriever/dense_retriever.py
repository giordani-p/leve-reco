# reco/retriever/dense_retriever.py
"""
DenseRetriever — V4 / P1 Híbrido (BM25 + MPNet)

Responsável por:
- Receber um texto de consulta já canônico (vindo do query_builder).
- Gerar embedding da consulta (MPNet via EmbeddingProvider).
- Consultar o índice vetorial (VectorIndex) e retornar Top-K por similaridade de cosseno.
- Aplicar filtros por metadados quando necessário (ex.: status="Published").

Observações:
- Espera que os embeddings de catálogo estejam L2-normalizados (L2 = norma de vetor igual a 1).
- O embedding da consulta também é normalizado L2.
- Não faz blending (isso é papel do hybrid_retriever/ranker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco.embeddings.embedding_provider import EmbeddingProvider
from reco.index.vector_index import VectorIndex


@dataclass
class DenseResult:
    id: str
    score_semantic: float
    metadata: Dict[str, Any]


class DenseRetriever:
    """
    Uso típico:
        retr = DenseRetriever.from_config(cfg, vector_index=index_trilhas, embedding_provider=prov)
        results = retr.search(query_text=texto, k=50, filters={"status": "Published"})
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        top_k_default: int,
    ) -> None:
        self._prov = embedding_provider
        self._index = vector_index
        self._k_default = int(top_k_default)

    # ---------- fábricas ----------
    @classmethod
    def from_config(
        cls,
        cfg,
        *,
        vector_index: VectorIndex,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> "DenseRetriever":
        prov = embedding_provider or EmbeddingProvider.from_config(cfg)
        return cls(
            embedding_provider=prov,
            vector_index=vector_index,
            top_k_default=getattr(cfg, "TOP_K_DEFAULT", 50),
        )

    # ---------- API ----------
    def search(
        self,
        query_text: str,
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        show_progress_bar: bool = False,
    ) -> List[DenseResult]:
        """
        Retorna Top-K resultados denso (MPNet) como lista de DenseResult.
        - query_text: texto canônico vindo do query_builder (pergunta + pistas do snapshot).
        - k: se None, usa TOP_K_DEFAULT.
        - filters: filtros por metadados do item (ex.: {"status": "Published"}).
        """
        k_eff = int(k or self._k_default)
        if not query_text or k_eff <= 0:
            return []

        qvec: np.ndarray = self._prov.embed_text(query_text, normalize=True)
        # VectorIndex espera vetor 1D float32 na mesma dimensão de catálogo
        results = self._index.search(qvec, k=k_eff, filters=filters or None)

        dense: List[DenseResult] = [
            DenseResult(id=item_id, score_semantic=float(score), metadata=meta or {})
            for (item_id, score, meta) in results
        ]
        return dense

    # ---------- utilitários ----------
    def metadata(self) -> Dict[str, Any]:
        """Retorna metadados úteis do provedor (modelo, versão, dim, device) e do índice."""
        prov_meta = self._prov.get_metadata()
        return {
            "embedding_model": prov_meta.model_name,
            "embedding_version": prov_meta.model_version,
            "dim": prov_meta.dim,
            "device": prov_meta.device,
            "index_name": getattr(self._index, "_name", "unknown"),
            "index_backend": getattr(self._index, "_backend", "unknown"),
            "index_size": self._index.size(),
        }
