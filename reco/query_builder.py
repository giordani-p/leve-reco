# reco/query_builder.py
"""
Sistema de Construção de Consultas - Leve Agents

Este módulo é responsável pela construção inteligente do texto de consulta (query_text)
que será utilizado tanto pelo sistema de busca densa (MPNet) quanto pelo BM25.

Princípios de Construção:
- Base sempre é a pergunta do usuário (user_question)
- Adiciona até SNAPSHOT_MAX_HINTS pistas do perfil psicológico do usuário
- Expansão opcional por sinônimos (apenas se USE_QUERY_SYNONYMS_DENSE=True)
- Blocos separados por ' || ' para melhor legibilidade

Estratégia de Enriquecimento:
- Heurística não intrusiva que evita termos genéricos
- Prioriza objetivos de carreira e dificuldades do usuário
- O mesmo query_text é usado por ambos os sistemas de busca
- BM25 faz sua própria expansão de sinônimos internamente
"""

from __future__ import annotations

from typing import Dict, List, Optional, Iterable, Any
import re
import unicodedata as _ud

from reco.config import RecoConfig

GENERIC_MARKERS = {
    "não informado",
    "nao informado",
    "n/a",
    "nenhum",
    "none",
    "desconhecido",
    "indefinido",
}

_WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+")


# ----------------------------
# Funções Auxiliares de Processamento de Texto
# ----------------------------
def _strip_accents(text: str) -> str:
    """
    Remove acentos de uma string usando normalização Unicode.
    
    Args:
        text: String a ser processada
        
    Returns:
        String sem acentos
    """
    if not text:
        return ""
    return "".join(ch for ch in _ud.normalize("NFKD", text) if not _ud.combining(ch))


def _clean(s: str) -> str:
    """
    Limpa e normaliza string removendo espaços extras.
    
    Args:
        s: String a ser limpa
        
    Returns:
        String normalizada
    """
    return " ".join((s or "").strip().split())


def _tokenize_norm(text: str) -> List[str]:
    """
    Tokenização acento-insensível e case-insensível.
    
    Args:
        text: Texto a ser tokenizado
        
    Returns:
        Lista de tokens normalizados
    """
    if not text:
        return []
    norm = _strip_accents(text).casefold()
    return [m.group(0) for m in _WORD_RE.finditer(norm)]


def _valid_hint(text: str) -> bool:
    """
    Verifica se um texto é uma pista válida para enriquecimento da consulta.
    
    Args:
        text: Texto a ser validado
        
    Returns:
        True se o texto for uma pista válida
    """
    if not text:
        return False
    t = _clean(text).casefold()
    if len(t) < 3:
        return False
    return t not in GENERIC_MARKERS


def _apply_length_cap(text: str, cfg: RecoConfig) -> str:
    """
    Aplica limite de tamanho na consulta se configurado.
    
    Args:
        text: Texto da consulta
        cfg: Configuração do sistema
        
    Returns:
        Texto limitado se necessário
    """
    max_chars = int(getattr(cfg, "QUERY_MAX_CHARS", 0) or 0)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _add_educational_context(user_question: str) -> str:
    """
    Adiciona contexto educacional à consulta quando necessário.
    
    Melhora a qualidade semântica direcionando a busca para educação e carreira,
    especialmente útil para consultas que não contêm palavras-chave educacionais explícitas.
    
    Args:
        user_question: Pergunta original do usuário
        
    Returns:
        Consulta enriquecida com contexto educacional
    """
    query_lower = user_question.lower()
    
    # Palavras-chave educacionais que indicam contexto educacional
    educational_keywords = [
        'aprender', 'estudar', 'curso', 'trilha', 'carreira', 'profissão',
        'trabalho', 'habilidade', 'competência', 'desenvolvimento',
        'programação', 'tecnologia', 'negócio', 'gestão', 'marketing',
        'design', 'vendas', 'atendimento', 'líder', 'equipe', 'projeto',
        'capacitação', 'formação', 'treinamento', 'especialização',
        'certificação', 'qualificação', 'profissionalização',
        # Mercado de trabalho
        'emprego', 'vaga', 'entrevista', 'currículo', 'cv', 'candidato',
        'recrutamento', 'seleção', 'contratação', 'admissão',
        'primeiro emprego', 'busca por emprego', 'oportunidade',
        'mercado de trabalho', 'inserção profissional', 'início de carreira'
    ]
    
    # Verifica se já tem contexto educacional
    has_educational_context = any(keyword in query_lower for keyword in educational_keywords)
    
    if has_educational_context:
        return user_question
    
    # Adiciona contexto educacional baseado na intenção da query
    if query_lower.startswith('como'):
        # Para queries "como fazer X", direciona para aprendizado
        return f"{user_question} educação carreira desenvolvimento profissional"
    elif query_lower.startswith('quero') or query_lower.startswith('preciso'):
        # Para queries "quero/preciso X", direciona para capacitação
        return f"{user_question} capacitação formação profissional"
    elif any(word in query_lower for word in ['fazer', 'criar', 'desenvolver', 'construir']):
        # Para queries sobre fazer algo, direciona para aprendizado
        return f"{user_question} aprendizado educação"
    else:
        # Para outras queries, adiciona contexto educacional geral
        return f"{user_question} educação carreira desenvolvimento"


# ----------------------------
# Acesso Seguro ao Snapshot do Usuário
# ----------------------------
def _get(d: Dict[str, Any], path: str, default=None):
    """
    Acesso seguro a campos aninhados do snapshot usando notação de ponto.
    
    Args:
        d: Dicionário do snapshot
        path: Caminho no formato 'a.b.c'
        default: Valor padrão se caminho não existir
        
    Returns:
        Valor encontrado ou default
    """
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _as_list(val: Any) -> List[str]:
    """
    Converte valor para lista de strings.
    
    Args:
        val: Valor a ser convertido
        
    Returns:
        Lista de strings
    """
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val if isinstance(x, (str, int, float))]
    return []


def _collect_hints(snapshot: Dict[str, Any], max_hints: int) -> List[str]:
    """
    Extrai pistas do perfil psicológico do usuário para enriquecer a consulta.
    
    Prioriza informações mais relevantes para recomendação educacional:
    1) Objetivos de carreira (principal, específicos, curto prazo)
    2) Dificuldades e barreiras
    3) Preferências de aprendizado
    4) Interesses e direção profissional
    5) Perfil e talentos
    6) Aspirações profissionais
    7) Contexto socioeconômico (apenas se sobrar espaço)
    
    Args:
        snapshot: Dicionário com perfil psicológico do usuário
        max_hints: Número máximo de pistas a extrair
        
    Returns:
        Lista de pistas extraídas
    """
    hints: List[str] = []

    def _add(items: Iterable[str]):
        nonlocal hints
        for it in items:
            if len(hints) >= max_hints:
                return
            if isinstance(it, str) and _valid_hint(it):
                hints.append(_clean(it))

    # 1) Objetivos
    _add(_as_list(_get(snapshot, "objetivos_carreira.objetivo_principal")))
    _add(_as_list(_get(snapshot, "objetivos_carreira.objetivos_especificos")))
    _add(_as_list(_get(snapshot, "objetivos_carreira.metas_temporais.curto_prazo")))

    if len(hints) >= max_hints:
        return hints

    # 2) Dificuldades / Barreiras
    _add(_as_list(_get(snapshot, "preferencias_aprendizado.dificuldades_aprendizado")))
    _add(_as_list(_get(snapshot, "situacao_academica.materias_dificuldade")))
    _add(_as_list(_get(snapshot, "barreiras_desafios.tecnicas")))
    _add(_as_list(_get(snapshot, "barreiras_desafios.sociais")))
    _add(_as_list(_get(snapshot, "barreiras_desafios.financeiras")))
    _add(_as_list(_get(snapshot, "barreiras_desafios.geograficas")))

    if len(hints) >= max_hints:
        return hints

    # 3) Preferências de aprendizado (compacta para frases curtas)
    modalidade = _get(snapshot, "preferencias_aprendizado.modalidade_preferida")
    ritmo = _get(snapshot, "preferencias_aprendizado.ritmo_aprendizado")
    horario = _get(snapshot, "preferencias_aprendizado.horario_estudo")
    recursos = _as_list(_get(snapshot, "preferencias_aprendizado.recursos_preferidos"))
    pref_parts = []
    if _valid_hint(str(modalidade or "")):
        pref_parts.append(f"Preferência: {modalidade}")
    if _valid_hint(str(ritmo or "")):
        pref_parts.append(f"Ritmo: {ritmo}")
    if _valid_hint(str(horario or "")):
        pref_parts.append(f"Horário: {horario}")
    if recursos:
        pref_parts.append("Recursos: " + ", ".join([_clean(r) for r in recursos[:3]]))
    if pref_parts and len(hints) < max_hints:
        _add(["; ".join(pref_parts)])

    if len(hints) >= max_hints:
        return hints

    # 4) Interesses & direção
    _add(_as_list(_get(snapshot, "interesses_pessoais.areas_curiosidade")))
    _add(_as_list(_get(snapshot, "interesses_pessoais.atividades_extracurriculares")))
    _add(_as_list(_get(snapshot, "interesses_pessoais.consumo_midia")))
    _add(_as_list(_get(snapshot, "situacao_academica.materias_favoritas")))

    if len(hints) >= max_hints:
        return hints

    # 5) Perfil & talentos (curtos)
    _add(_as_list(_get(snapshot, "perfil_disc.caracteristicas_chave")))
    _add(_as_list(_get(snapshot, "perfil_disc.areas_compatibilidade")))
    _add(_as_list(_get(snapshot, "perfil_disc.pontos_atencao")))
    _add(_as_list(_get(snapshot, "talentos_cliftonstrengths.talentos_dominantes")))
    if len(hints) < max_hints:
        _add(_as_list(_get(snapshot, "talentos_cliftonstrengths.talentos_secundarios")))

    if len(hints) >= max_hints:
        return hints

    # 6) Aspirações (campos curtos)
    _add(_as_list(_get(snapshot, "aspiracoes_profissionais.tipo_empresa")))
    _add(_as_list(_get(snapshot, "aspiracoes_profissionais.tamanho_equipe")))
    _add(_as_list(_get(snapshot, "aspiracoes_profissionais.disponibilidade_tempo")))
    _add(_as_list(_get(snapshot, "aspiracoes_profissionais.impacto_desejado")))

    if len(hints) >= max_hints:
        return hints

    # 7) Contexto (apenas se sobrar espaço)
    _add(_as_list(_get(snapshot, "dados_pessoais.localizacao")))
    _add(_as_list(_get(snapshot, "contexto_socioeconomico.origem")))

    return hints[:max_hints]


def _expand_synonyms_dense(
    user_question: str,
    cfg: RecoConfig,
    max_synonyms: int = 6,
) -> List[str]:
    """
    Expansão leve apenas para o caminho DENSO (se habilitada em config).
    *Na P1, essa flag deve ficar False; BM25 expande por conta própria.
    """
    if not (getattr(cfg, "USE_QUERY_SYNONYMS_DENSE", False) and getattr(cfg, "QUERY_SYNONYMS", None)):
        return []

    toks = _tokenize_norm(user_question)
    keys = set(t for t in toks if len(t) > 3)

    collected: List[str] = []
    seen = set()

    synonyms_map = getattr(cfg, "QUERY_SYNONYMS", {}) or {}
    for key in keys:
        syns = synonyms_map.get(key)
        if not syns:
            continue
        for s in syns:
            s_norm = _strip_accents(s).casefold().strip()
            if not s_norm or s_norm in seen:
                continue
            seen.add(s_norm)
            collected.append(_clean(s))
            if len(collected) >= max_synonyms:
                return collected
    return collected


# ----------------------------
# API Principal de Construção de Consulta
# ----------------------------
def build(
    user_question: str,
    snapshot: Optional[Dict],
    cfg: RecoConfig,
    contexto_extra: Optional[str] = None,
) -> str:
    """
    Constrói o texto final de consulta enriquecido para busca híbrida.
    
    Combina a pergunta do usuário com pistas do perfil psicológico e contexto adicional,
    criando uma consulta otimizada para ambos os sistemas de busca (denso e BM25).
    
    Args:
        user_question: Pergunta original do usuário
        snapshot: Perfil psicológico do usuário (opcional)
        cfg: Configuração do sistema
        contexto_extra: Contexto adicional fornecido pelo usuário
        
    Returns:
        Texto de consulta enriquecido
    """
    blocks: List[str] = []

    # Etapa 1: Base da consulta com contexto educacional
    uq = _clean(user_question)
    if uq:
        # Adiciona contexto educacional se necessário
        educational_context = _add_educational_context(uq)
        blocks.append(educational_context)

    # Etapa 2: Pistas do perfil psicológico do usuário
    if getattr(cfg, "USE_SNAPSHOT_HINTS", True) and isinstance(snapshot, dict):
        max_hints = int(getattr(cfg, "SNAPSHOT_MAX_HINTS", 2) or 2)
        snap_hints = _collect_hints(snapshot, max_hints=max_hints)
        if snap_hints:
            blocks.append("; ".join(snap_hints))

    # Etapa 3: Contexto adicional fornecido pelo usuário
    cx = _clean(contexto_extra or "")
    if cx:
        blocks.append(cx)

    # Etapa 4: Expansão de sinônimos (apenas se habilitada para busca densa)
    syns: List[str] = _expand_synonyms_dense(user_question, cfg)
    if syns:
        blocks.append("; ".join(syns))

    # Montagem final da consulta
    blocks = [b for b in (blocks or []) if b]
    if not blocks:
        blocks = [""]

    query_text = " || ".join(blocks)
    query_text = _apply_length_cap(query_text, cfg)
    return query_text


# Alias para compatibilidade com código legado
def build_query(
    user_question: str,
    snapshot: Optional[Dict],
    cfg: RecoConfig,
    contexto_extra: Optional[str] = None,
) -> str:
    """
    Alias para a função build() para manter compatibilidade com código legado.
    
    Args:
        user_question: Pergunta original do usuário
        snapshot: Perfil psicológico do usuário (opcional)
        cfg: Configuração do sistema
        contexto_extra: Contexto adicional fornecido pelo usuário
        
    Returns:
        Texto de consulta enriquecido
    """
    return build(user_question=user_question, snapshot=snapshot, cfg=cfg, contexto_extra=contexto_extra)
