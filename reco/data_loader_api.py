# reco/data_loader_api.py
"""
Cliente HTTP para leitura do catálogo de trilhas a partir do backend da Leve.

Responsabilidades:
- Buscar trilhas publicadas via /api/trails (com suporte a ?status=Published).
- (Opcional) Buscar detalhe de uma trilha via /api/trails/{publicId}.
- Implementar timeouts granulares, retry com backoff (com jitter) e tratamento de erros.
- Retornar dados *brutos* (list[dict] / dict), deixando a normalização para TrailCandidate.from_source().

Observações:
- Mesmo pedindo Published, o filtro final por status é feito mais à frente (config.ALLOWED_STATUS).
- Se /api/trails paginar, há suporte básico a iteração até API_MAX_PAGES (items/nextPageToken).
"""

from __future__ import annotations

import time
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import UUID

import httpx

from .config import RecoConfig


# -----------------------------
# Helpers internos
# -----------------------------
def _build_timeout(cfg: RecoConfig) -> httpx.Timeout:
    """Monta timeouts granulares a partir do RecoConfig."""
    return httpx.Timeout(
        connect=cfg.HTTP_TIMEOUT_CONNECT,
        read=cfg.HTTP_TIMEOUT_READ,
        write=cfg.HTTP_TIMEOUT_WRITE,
        pool=cfg.HTTP_TIMEOUT_POOL,
    )


def _retry_delay(attempt_idx: int, base: float, retry_after: Optional[float]) -> float:
    """
    Calcula o delay de retry:
    - Se houver Retry-After (segundos), prioriza.
    - Senão: backoff exponencial com jitter (±20%).
    """
    if retry_after is not None and retry_after > 0:
        return float(retry_after)
    delay = base * (2 ** attempt_idx)
    jitter = delay * random.uniform(-0.2, 0.2)
    return max(0.0, delay + jitter)


def _should_retry(status_code: Optional[int], exc: Optional[BaseException]) -> bool:
    """
    Define se vale tentar novamente:
    - Erros de rede/transientes do httpx
    - HTTP 408, 429 e 5xx
    """
    if exc is not None:
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                httpx.PoolTimeout,
                httpx.WriteError,
            ),
        )
    if status_code is None:
        return False
    if status_code in (408, 429):
        return True
    if 500 <= status_code <= 599:
        return True
    return False


def _join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"


def _ensure_list_of_dicts(data: Iterable[Any]) -> List[Dict[str, Any]]:
    """Valida que 'data' é uma sequência de dicts e retorna lista nova."""
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entrada inválida na lista de trilhas (índice {i}): esperado objeto JSON (dict).")
        out.append(item)
    return out


# -----------------------------
# Cliente HTTP (context manager)
# -----------------------------
class TrailsApiClient:
    """
    Encapsula o httpx.Client com timeout, headers e retry/backoff.
    Reutiliza conexões (pool).
    """

    def __init__(self, cfg: RecoConfig) -> None:
        self._cfg = cfg

        # Cabeçalhos padrão (User-Agent útil p/ observabilidade no backend)
        headers = {
            "User-Agent": f"LeveReco/1.0 ({getattr(cfg, 'MODEL_VERSION', 'unspecified')}; {getattr(cfg, 'INDEX_TRILHAS', 'trilhas_mpnet_v1')})",
            "Accept": "application/json",
        }
        # Header opcional de API key, se existir na config
        api_key = getattr(cfg, "API_KEY", None)
        if api_key:
            headers["X-API-Key"] = str(api_key)

        self._client = httpx.Client(
            base_url=cfg.TRAILS_API_BASE.rstrip("/"),
            timeout=_build_timeout(cfg),
            headers=headers,
            http2=True,
        )

    def close(self) -> None:
        self._client.close()

    # -----------------------------
    # Baixo nível (request + retry)
    # -----------------------------
    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """
        Faz uma requisição com tentativas adicionais em caso de erro transitório.
        Respeita Retry-After quando presente.
        """
        attempts = 1 + max(0, self._cfg.HTTP_RETRIES)
        last_exc: Optional[BaseException] = None
        last_status: Optional[int] = None

        url = _join_url(self._cfg.TRAILS_API_BASE, path)

        for i in range(attempts):
            try:
                resp = self._client.request(method, url, params=params)
                last_status = resp.status_code

                if _should_retry(last_status, None) and i < attempts - 1:
                    retry_after_hdr = resp.headers.get("Retry-After")
                    retry_after = None
                    if retry_after_hdr:
                        try:
                            retry_after = float(retry_after_hdr)
                        except Exception:
                            retry_after = None
                    time.sleep(_retry_delay(i, self._cfg.HTTP_BACKOFF_BASE, retry_after))
                    continue

                # Fora dos casos de retry, dispara erro se 4xx/5xx
                resp.raise_for_status()
                return resp

            except httpx.HTTPStatusError as e:
                last_exc = e
                last_status = e.response.status_code
                if _should_retry(last_status, None) and i < attempts - 1:
                    retry_after_hdr = e.response.headers.get("Retry-After")
                    retry_after = None
                    if retry_after_hdr:
                        try:
                            retry_after = float(retry_after_hdr)
                        except Exception:
                            retry_after = None
                    time.sleep(_retry_delay(i, self._cfg.HTTP_BACKOFF_BASE, retry_after))
                    continue
                raise

            except (httpx.ConnectError, httpx.ReadError, httpx.NetworkError,
                    httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.WriteError) as e:
                last_exc = e
                if _should_retry(None, e) and i < attempts - 1:
                    time.sleep(_retry_delay(i, self._cfg.HTTP_BACKOFF_BASE, None))
                    continue
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Falha desconhecida ao executar requisição HTTP.")

    # -----------------------------
    # Endpoints
    # -----------------------------
    def fetch_trails(self) -> List[Dict[str, Any]]:
        """
        Busca lista de trilhas em /api/trails.
        - Se API_FILTER_PUBLISHED=True, envia ?status=Published.
        - Usa API_PAGE_SIZE_HINT como 'limit' (se configurado).
        - Suporte básico a paginação: { items, nextPageToken }.
        - Retorna SEM normalizar (lista de dicts).
        """
        path = "/api/trails"
        params: Dict[str, Any] = {}
        if self._cfg.API_FILTER_PUBLISHED:
            params["status"] = "Published"
        if getattr(self._cfg, "API_PAGE_SIZE_HINT", None):
            params["limit"] = int(self._cfg.API_PAGE_SIZE_HINT)

        resp = self._request_with_retry("GET", path, params=params)
        data = _ensure_json(resp)

        # Caso tradicional: lista direta
        if isinstance(data, list):
            return _ensure_list_of_dicts(data)

        # Paginação futura: { "items": [...], "nextPageToken": "..." }
        if isinstance(data, dict) and "items" in data:
            items: List[Dict[str, Any]] = _ensure_list_of_dicts(data.get("items", []))
            next_token = data.get("nextPageToken")
            pages = 1

            while next_token and pages < self._cfg.API_MAX_PAGES:
                page_params = dict(params)
                page_params["pageToken"] = next_token
                resp = self._request_with_retry("GET", path, params=page_params)
                page_data = _ensure_json(resp)

                if isinstance(page_data, list):
                    items.extend(_ensure_list_of_dicts(page_data))
                    next_token = None
                elif isinstance(page_data, dict):
                    items.extend(_ensure_list_of_dicts(page_data.get("items", [])))
                    next_token = page_data.get("nextPageToken")
                else:
                    next_token = None

                pages += 1

            return items

        raise ValueError("Resposta inesperada de /api/trails: esperado list ou dict com chave 'items'.")

    def fetch_trail_detail(self, public_id: UUID | str) -> Optional[Dict[str, Any]]:
        """
        Busca o detalhe de uma trilha via /api/trails/{publicId}.
        Retorna dict ou None se 404.
        """
        pid = str(public_id)
        path = f"/api/trails/{pid}"

        try:
            resp = self._request_with_retry("GET", path)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

        data = _ensure_json(resp)
        if not isinstance(data, dict):
            raise ValueError("Resposta inesperada de /api/trails/{publicId}: esperado objeto JSON (dict).")
        return data


# -----------------------------
# API de alto nível (funções)
# -----------------------------
def fetch_trails(cfg: RecoConfig) -> List[Dict[str, Any]]:
    """Convenience para uso fora da classe."""
    client = TrailsApiClient(cfg)
    try:
        return client.fetch_trails()
    finally:
        client.close()


def fetch_trail_detail(cfg: RecoConfig, public_id: UUID | str) -> Optional[Dict[str, Any]]:
    client = TrailsApiClient(cfg)
    try:
        return client.fetch_trail_detail(public_id)
    finally:
        client.close()


# -----------------------------
# Utilidades
# -----------------------------
def _ensure_json(resp: httpx.Response) -> Any:
    """Valida Content-Type e faz resp.json() com mensagem de erro clara."""
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" not in ctype and "+json" not in ctype:
        # Ainda assim tenta decodificar para capturar payload de erro útil
        try:
            return resp.json()
        except Exception:
            text = (resp.text or "")[:500]
            raise ValueError(f"Resposta não-JSON (status {resp.status_code}). Content-Type='{ctype}'. Trecho: {text!r}")
    try:
        return resp.json()
    except Exception as e:
        text = (resp.text or "")[:500]
        raise ValueError(f"Falha ao decodificar JSON (status {resp.status_code}). Trecho: {text!r}") from e
