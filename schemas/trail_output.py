# schemas/trail_output.py
"""
Schema de Saída do Sistema de Recomendação - Leve Agents

Este módulo define os schemas de saída para o sistema de recomendação de trilhas educacionais,
estabelecendo a estrutura de dados retornada ao usuário após o processamento da consulta.
"""

from typing import List, Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, ConfigDict, model_validator


class SuggestedTrail(BaseModel):
    """
    Representa uma trilha educacional sugerida pelo sistema.
    
    Contém todas as informações necessárias para apresentar a recomendação ao usuário,
    incluindo identificação, descrição, explicação do match e score de relevância.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    publicId: Optional[UUID] = Field(
        default=None,
        description="Identificador público da trilha (UUID)."
    )
    slug: Optional[str] = Field(
        default=None,
        description="Slug da trilha (obrigatório se não houver publicId).",
        max_length=200,
    )
    title: str = Field(
        ...,
        min_length=3,
        max_length=140,
        description="Título da trilha."
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Descrição da trilha (opcional)."
    )
    why_match: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Breve explicação (tom cordial e jovem) do porquê essa trilha foi escolhida."
    )
    match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Pontuação de aderência no intervalo [0.0, 1.0]."
    )

    @model_validator(mode="after")
    def _at_least_one_identifier(self):
        """Garante que pelo menos um identificador (publicId ou slug) exista."""
        if self.publicId is None and (self.slug is None or self.slug.strip() == ""):
            raise ValueError("É obrigatório informar ao menos um identificador: publicId ou slug.")
        return self


class WebFallback(BaseModel):
    """
    Conteúdo externo sugerido como fallback.
    Observação: no contexto atual (CLI de recomendações com filtro rígido por Published),
    este recurso não é utilizado e deve permanecer None em TrailOutput.
    Mantido por compatibilidade.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(
        ...,
        min_length=3,
        max_length=140,
        description="Título curto do conteúdo."
    )
    url: HttpUrl = Field(
        ...,
        description="URL do conteúdo sugerido."
    )
    why_useful: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Por que esse conteúdo é útil para o jovem."
    )


class QueryUnderstanding(BaseModel):
    """Entendimento resumido da questão do usuário (telemetria/logs)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    tema: str = Field(
        ...,
        min_length=3,
        max_length=48,
        description="Tema principal identificado na questão."
    )
    palavras_chave: List[str] = Field(
        default_factory=list,
        description="Lista de palavras-chave relevantes extraídas da questão."
    )


class TrailOutput(BaseModel):
    """
    Saída do Sistema de Recomendação (CLI).
    Sempre retorna short_answer + CTA. Quando houver aderência, retorna 1..3 suggested_trails.
    Não utilizamos web_fallback no contexto atual (deve permanecer None).
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    status: Literal["ok", "fora_do_escopo"] = Field(
        ...,
        description='Status do atendimento: "ok" quando há orientação útil; '
                    '"fora_do_escopo" quando não houver trilhas publicadas aderentes.'
    )
    mensagem_padrao: Optional[str] = Field(
        default=None,
        min_length=10,
        max_length=500,
        description="Mensagem padrão usada quando não houver sugestão aderente."
    )

    short_answer: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Resposta curta (2–4 linhas) à questão do jovem."
    )
    suggested_trails: Optional[List[SuggestedTrail]] = Field(
        default=None,
        description="Lista de trilhas sugeridas (1 a 3 itens)."
    )
    # Mantido por compatibilidade; não utilizado no fluxo atual (deve ser None).
    web_fallback: Optional[List[WebFallback]] = Field(
        default=None,
        description="Não utilizado no contexto atual (manter None)."
    )
    cta: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Chamada para ação, ex.: 'Começar trilha'."
    )
    query_understanding: Optional[QueryUnderstanding] = Field(
        default=None,
        description="Entendimento da questão para logs/monitoramento (opcional)."
    )

    @model_validator(mode="after")
    def _limits_and_consistency(self):
        """
        Regras estruturais:
        - status="ok": exige suggested_trails com 1..3 itens; web_fallback deve ser None.
        - status="fora_do_escopo": suggested_trails deve estar vazio/None; exige mensagem_padrao; web_fallback None.
        - Limita web_fallback a None no contexto atual.
        """
        if self.web_fallback is not None and len(self.web_fallback) > 0:
            raise ValueError("web_fallback não deve ser utilizado no contexto atual (mantenha None).")

        if self.status == "ok":
            if not self.suggested_trails or len(self.suggested_trails) == 0:
                raise ValueError("Quando status='ok', é obrigatório retornar 1 a 3 sugestões em suggested_trails.")
            if len(self.suggested_trails) > 3:
                raise ValueError("suggested_trails deve conter no máximo 3 itens.")
        elif self.status == "fora_do_escopo":
            if self.suggested_trails and len(self.suggested_trails) > 0:
                raise ValueError("Quando status='fora_do_escopo', não deve haver suggested_trails.")
            if self.mensagem_padrao is None or self.mensagem_padrao.strip() == "":
                raise ValueError("mensagem_padrao é obrigatória quando status='fora_do_escopo'.")

        return self
