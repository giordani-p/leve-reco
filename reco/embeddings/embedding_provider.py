# reco/embeddings/embedding_provider.py
"""
EmbeddingProvider — V4 / P1 Híbrido (BM25 + MPNet)

Responsável por:
- Carregar o modelo de embeddings (MPNet multilíngue) de forma preguiçosa (lazy).
- Gerar embeddings para consultas e documentos (trilhas/vagas) em batch.
- Normalizar vetores (L2) para uso com similaridade por cosseno.
- Expor metadados úteis (model_name, model_version, dim, device).

Dependências:
- sentence-transformers
- torch
- numpy

Observações:
- A dimensão (dim) deve bater com config.EMBED_DIM (ex.: 768 para mpnet-base).
- Normalização L2 habilitada por padrão para compatibilidade com índices k-NN/cosseno.
- Dispositivo: respeita config.EMBEDDING_DEVICE, senão tenta "cuda" e cai para "cpu".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception as e:
    # Mantemos a exceção explícita para facilitar diagnóstico em ambientes sem deps.
    raise ImportError(
        "Faltam dependências para embeddings. Instale: 'sentence-transformers' e 'torch'. "
        "Ex.: pip install sentence-transformers torch"
    ) from e


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza L2 linha a linha (cada vetor), retornando float32."""
    if mat.ndim == 1:
        denom = np.linalg.norm(mat) + eps
        return (mat / denom).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return (mat / norms).astype(np.float32)


@dataclass
class EmbeddingMetadata:
    model_name: str
    model_version: str
    dim: int
    device: str


class EmbeddingProvider:
    """
    Provedor de embeddings plugável.

    Interface pública:
      - embed_text(text: str, normalize: bool = True) -> np.ndarray  # (dim,)
      - embed_texts(texts: Sequence[str], normalize: bool = True, batch_size: Optional[int] = None) -> np.ndarray  # (N, dim)
      - get_metadata() -> EmbeddingMetadata

    Uso típico:
        provider = EmbeddingProvider.from_config(cfg)
        vecs = provider.embed_texts(docs, normalize=True)
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        dim: int,
        device: Optional[str] = None,
        default_batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._model_version = model_version
        self._dim = dim
        self._user_device = device  # preferência do usuário/config
        self._default_batch = default_batch_size

        self._resolved_device = self._resolve_device(device)
        self._model: Optional[SentenceTransformer] = None

    # --------- Fábrica a partir da RecoConfig ---------
    @classmethod
    def from_config(cls, cfg) -> "EmbeddingProvider":
        return cls(
            model_name=cfg.EMBEDDING_MODEL,
            model_version=getattr(cfg, "MODEL_VERSION", "unspecified"),
            dim=getattr(cfg, "EMBED_DIM", 768),
            device=getattr(cfg, "EMBEDDING_DEVICE", None),
            default_batch_size=getattr(cfg, "EMBEDDING_BATCH_SIZE", 64),
        )

    # --------- API pública ---------
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        if not text:
            # vetor zero normalizado (evita exceções em entradas vazias)
            z = np.zeros(self._dim, dtype=np.float32)
            return _l2_normalize(z) if normalize else z

        model = self._get_model()
        emb = model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalizamos aqui para manter controle
            device=self._resolved_device,
            show_progress_bar=False,
        )[0]

        if emb.shape[-1] != self._dim:
            raise ValueError(
                f"Dimensão de embedding inesperada: {emb.shape[-1]} (esperado {self._dim}). "
                f"Verifique config.EMBED_DIM e o modelo '{self._model_name}'."
            )
        return _l2_normalize(emb) if normalize else emb.astype(np.float32)

    def embed_texts(
        self,
        texts: Sequence[str],
        normalize: bool = True,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        model = self._get_model()
        bs = int(batch_size or self._default_batch)

        # Sanitiza entradas nulas
        safe_texts = [t if t is not None else "" for t in texts]

        embs = model.encode(
            safe_texts,
            batch_size=bs,
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=self._resolved_device,
            show_progress_bar=show_progress_bar,
        )

        if embs.shape[-1] != self._dim:
            raise ValueError(
                f"Dimensão de embedding inesperada: {embs.shape[-1]} (esperado {self._dim}). "
                f"Verifique config.EMBED_DIM e o modelo '{self._model_name}'."
            )

        return _l2_normalize(embs) if normalize else embs.astype(np.float32)

    def get_metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            model_name=self._model_name,
            model_version=self._model_version,
            dim=self._dim,
            device=self._resolved_device,
        )

    # --------- Internos ---------
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            # load_in_8bit/4bit não são necessários aqui; mantemos simples para CPU/GPU padrão.
            self._model = SentenceTransformer(self._model_name, device=self._resolved_device)
            # Valida dimensão informada vs. real
            test = self._model.encode(["_probe_"], convert_to_numpy=True, normalize_embeddings=False)
            real_dim = int(test.shape[-1])
            if real_dim != self._dim:
                raise ValueError(
                    f"Config.EMBED_DIM={self._dim} não bate com a dimensão real do modelo ({real_dim}). "
                    f"Ajuste EMBED_DIM ou troque o modelo."
                )
        return self._model

    @staticmethod
    def _resolve_device(pref: Optional[str]) -> str:
        if pref in ("cpu", "cuda"):
            return pref
        return "cuda" if torch.cuda.is_available() else "cpu"
