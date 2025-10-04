import json
import re
from typing import Any, Dict, Optional


def try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Tenta extrair um JSON válido de um texto bruto vindo do LLM.
    Estratégias:
      1) Procurar bloco cercado por ```json ... ```
      2) Procurar do primeiro '{' ao último '}' e tentar json.loads
      Retorna dict ou None.
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Bloco cercado por ```json ... ```
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Primeiro '{' até o último '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            cleaned = snippet.replace("\u00A0", " ").replace("\u200b", "")
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    return None
