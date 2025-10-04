# reco/data_loader.py
"""
Leitura dos dados mockados a partir de arquivos JSON em 'files/'.

Responsabilidades:
- Carregar o snapshot do perfil (dict).
- Carregar o catálogo de trilhas (lista de dicts).
- Validar estrutura mínima e emitir erros claros quando algo estiver fora do esperado.

Observações:
- A validação de campos específicos das trilhas é feita na normalização (TrailCandidate).
- Aqui garantimos apenas os tipos de topo: snapshot=dict, trilhas=list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido em {p}: {e}") from e


def load_snapshot(snapshot_path: str | Path | None) -> Dict[str, Any]:
    """
    Lê e retorna o snapshot do perfil.
    Espera um objeto JSON (dict).
    Se snapshot_path for None, retorna um dicionário vazio.
    """
    if snapshot_path is None:
        return {}
    
    data = _read_json(snapshot_path)
    if not isinstance(data, dict):
        raise ValueError(f"O snapshot deve ser um objeto JSON (dict). Arquivo: {snapshot_path}")
    return data


def load_trails(trails_path: str | Path) -> List[Dict[str, Any]]:
    """
    Lê e retorna a lista bruta de trilhas.
    Espera um array JSON (list) de objetos.
    """
    data = _read_json(trails_path)
    if not isinstance(data, list):
        raise ValueError(f"O catálogo de trilhas deve ser uma lista JSON. Arquivo: {trails_path}")
    # Checagem leve: garantir que cada item seja dict
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entrada de trilha inválida no índice {i}: esperado objeto JSON (dict).")
    return data
