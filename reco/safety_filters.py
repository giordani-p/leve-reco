# reco/safety_filters.py
"""
Safety Filters - Proteção contra conteúdo inadequado

Sistema focado em proteger contra:
- Conteúdo violento ou perigoso
- Linguagem inadequada ou ofensiva
- Spam ou queries irrelevantes

NÃO bloqueia contextos educacionais válidos.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class SafetyCheckResult:
    is_safe: bool
    reason: str = None
    suggested_response: str = None


class SafetyFilters:
    """
    Filtros de segurança focados em proteção essencial.
    """
    
    def __init__(self):
        # Padrões de conteúdo violento ou perigoso
        self.violent_patterns = [
            r'\b(matar|assassinar|homic[íi]dio|viol[eê]ncia|agredir|agress[ãa]o)\b',
            r'\b(suic[íi]dio|morrer|morte|morrendo|se matar)\b',
            r'\b(ódio|raiva|vingança|vingar|vingança)\b',
            r'\b(perigo|perigoso|ameaça|ameaçar|ameaçando)\b',
            r'\b(destruir|quebrar|danificar|estragar)\b',
        ]
        
        # Padrões de linguagem inadequada
        self.inappropriate_patterns = [
            r'\b(puta|puto|caralho|porra|merda|foda|foder)\b',
            r'\b(viado|bicha|gay|lesbica|traveco)\b',  # Termos ofensivos
            r'\b(otario|idiota|burro|estupido|imbecil)\b',
            r'\b(desgraça|desgraçado|filho da puta)\b',
        ]
        
        # Padrões de spam ou conteúdo irrelevante
        self.spam_patterns = [
            r'^[0-9\s]+$',  # Apenas números
            r'^[a-z]{1,2}$',  # Apenas 1-2 letras
            r'^(teste|test|oi|ola|hello|hi|tchau|bye)$',  # Saudações genéricas
            r'^.{1,3}$',  # Texto muito curto
            r'^(asdf|qwerty|1234|abcd)$',  # Sequências aleatórias
        ]
        
        # Compilar regex para performance
        self.violent_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.violent_patterns]
        self.inappropriate_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.inappropriate_patterns]
        self.spam_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.spam_patterns]
    
    def check_query_safety(self, query: str) -> SafetyCheckResult:
        """
        Verifica se a query é segura e adequada.
        Foca apenas em proteção essencial.
        """
        query_clean = query.strip()
        
        if not query_clean:
            return SafetyCheckResult(
                is_safe=False, 
                reason="query_vazia", 
                suggested_response="Por favor, digite uma pergunta."
            )
        
        # Verificar spam
        for regex in self.spam_regex:
            if regex.search(query_clean):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="spam_ou_curta",
                    suggested_response="Sua pergunta é muito curta ou genérica. Por favor, seja mais específico sobre o que você quer aprender ou desenvolver."
                )
        
        # Verificar linguagem inadequada
        for regex in self.inappropriate_regex:
            if regex.search(query_clean):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="linguagem_inadequada",
                    suggested_response="Por favor, use uma linguagem respeitosa. Nossa plataforma é focada em educação e desenvolvimento profissional."
                )
        
        # Verificar conteúdo violento
        for regex in self.violent_regex:
            if regex.search(query_clean):
                return SafetyCheckResult(
                    is_safe=False,
                    reason="conteudo_violento",
                    suggested_response="Nossa plataforma é focada em educação e desenvolvimento profissional. Posso te ajudar com questões sobre carreira, estudos ou habilidades técnicas?"
                )
        
        return SafetyCheckResult(is_safe=True)


def check_query_safety(query: str) -> SafetyCheckResult:
    """
    Função de conveniência para verificar segurança da query.
    """
    filters = SafetyFilters()
    return filters.check_query_safety(query)


def create_safety_response(safety_result: SafetyCheckResult, original_query: str) -> dict:
    """
    Cria resposta de segurança quando a query não é adequada.
    """
    return {
        "status": "fora_do_escopo",
        "mensagem_padrao": safety_result.suggested_response,
        "suggested_trails": None,
        "web_fallback": None,
        "cta": None,
        "query_understanding": {
            "tema": "inadequado",
            "palavras_chave": original_query.split()[:5]  # Primeiras 5 palavras
        }
    }
