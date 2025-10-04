# schemas/trail_candidate.py
"""
Schema de Candidato de Trilha - Leve Agents

Este módulo define o schema para representar trilhas educacionais normalizadas
no sistema de recomendação, estabelecendo a estrutura de dados utilizada
internamente pelo pipeline de processamento.
"""

from typing import List, Literal, Optional, Iterable
from uuid import UUID
from pydantic import BaseModel, Field

Difficulty = Literal["Beginner", "Intermediate", "Advanced"]
Status = Literal["Published", "Draft", "Archived"]


class TrailCandidate(BaseModel):
    """
    Representa um item do catálogo de trilhas normalizado para uso no Sistema de Recomendação.
    
    Este schema define a estrutura padronizada de uma trilha educacional após normalização,
    incluindo todos os campos necessários para busca semântica e textual.
    
    Campos Essenciais:
    - publicId: Identificador único obrigatório (UUID)
    - title: Título da trilha (obrigatório)
    - combined_text: Texto combinado para embeddings (MPNet)
    
    Observações:
    - Se title estiver ausente, usa subtitle como fallback
    - combined_text é construído automaticamente para otimizar busca semântica
    - Validação rigorosa de tipos e formatos
    """
    publicId: UUID = Field(..., description="Identificador público obrigatório (UUID).")
    slug: Optional[str] = Field(default=None, description="Slug da trilha (opcional).")
    title: str
    subtitle: Optional[str] = Field(default=None, description="Subtítulo da trilha (opcional).")
    tags: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list, description="Tópicos/temas da trilha.")
    area: Optional[str] = Field(default=None, description="Área/domínio da trilha.")
    difficulty: Optional[Difficulty] = Field(default=None)
    description: str = Field(default="")
    status: Optional[Status] = Field(default=None, description="Status original no catálogo (auditoria).")
    combined_text: str = Field(
        default="",
        description="Concatenação de campos textuais (título, subtítulo, tags, descrição) para similaridade.",
    )

    @classmethod
    def from_source(cls, item: dict) -> "TrailCandidate":
        """
        Normaliza um item bruto do JSON de origem para o contrato TrailCandidate.
        - Dedupe/limpeza de tags e campos textuais.
        - Mapeamento robusto de difficulty/status (PT/EN).
        - Montagem controlada do combined_text.
        """
        # --- Identificador essencial (UUID) ---
        public_id_raw = item.get("publicId") or item.get("id")
        if not public_id_raw:
            raise ValueError("publicId é obrigatório para construir TrailCandidate.")
        public_id = UUID(str(public_id_raw))

        # --- Campos básicos ---
        slug = _clean_str(item.get("slug"))
        title = _clean_str(item.get("title"))
        subtitle = _clean_str(item.get("subtitle"))

        if not title and subtitle:
            # fallback: usa subtitle como título se title estiver vazio
            title = subtitle

        # --- Tags: aceita lista heterogênea, remove vazios, dedupe preservando ordem ---
        tags_raw = item.get("tags")
        tags_list = _normalize_tags(tags_raw)
        
        # --- Topics: similar às tags, mas campo separado ---
        topics_raw = item.get("topics")
        topics_list = _normalize_tags(topics_raw)
        
        # --- Area/domínio ---
        area = _clean_str(item.get("area"))

        # --- Difficulty normalizada (PT/EN) ---
        difficulty = _map_difficulty(item.get("difficulty"))

        # --- Descrição e status (para auditoria) ---
        description = _clean_str(item.get("description") or item.get("summary"))
        status = _map_status(item.get("status"))

        # --- Texto combinado (título | subtítulo | #tags | descrição) ---
        tags_str = " ".join(f"#{t}" for t in tags_list) if tags_list else ""
        combined_parts = [title, subtitle if subtitle and subtitle != title else "", tags_str, description]
        combined_text = " | ".join([p for p in combined_parts if p])

        return cls(
            publicId=public_id,
            slug=slug or None,
            title=title or "",
            subtitle=subtitle or None,
            tags=tags_list,
            topics=topics_list,
            area=area or None,
            difficulty=difficulty,
            description=description or "",
            status=status,
            combined_text=combined_text,
        )


# ----------------------------
# Helpers de normalização
# ----------------------------
def _clean_str(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    # colapsa espaços múltiplos
    return " ".join(s.split())


def _normalize_tags(val) -> List[str]:
    if not val:
        return []
    out: List[str] = []
    seen = set()
    # aceita lista heterogênea
    iterable: Iterable = val if isinstance(val, (list, tuple)) else [val]
    for x in iterable:
        t = _clean_str(x)
        if not t:
            continue
        # chave de dedupe case-insensitive
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)  # preserva grafia original para UI
    return out


def _map_difficulty(raw) -> Optional[Difficulty]:
    if not isinstance(raw, str):
        return None
    norm = raw.strip().casefold()
    if norm in {"beginner", "iniciante", "iniciantes", "basico", "básico", "do zero"}:
        return "Beginner"
    if norm in {"intermediate", "intermediario", "intermediário", "medio", "médio"}:
        return "Intermediate"
    if norm in {"advanced", "avancado", "avançado"}:
        return "Advanced"
    return None


def _map_status(raw) -> Optional[Status]:
    if not isinstance(raw, str):
        return None
    s = raw.strip().casefold()
    if s == "published":
        return "Published"
    if s == "draft":
        return "Draft"
    if s == "archived":
        return "Archived"
    return None
