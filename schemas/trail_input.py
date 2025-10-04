# schemas/trail_input.py
"""
Schema de Entrada do Sistema de Recomendação - Leve Agents

Este módulo define o schema de entrada para o sistema de recomendação de trilhas educacionais,
estabelecendo a estrutura de dados que o usuário deve fornecer para obter recomendações personalizadas.
"""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class TrailInput(BaseModel):
    """
    Schema de entrada para o Sistema de Recomendação de Trilhas Educacionais.

    Define a estrutura de dados necessária para processar uma consulta de recomendação,
    incluindo a pergunta do usuário, contexto adicional e configurações de retorno.

    Campos Principais:
    - user_question: Pergunta/dúvida do jovem em linguagem natural (obrigatório)
    - user_id: Identificador único do usuário para personalização (opcional)
    - contexto_extra: Detalhes adicionais relevantes (área de interesse, nível, objetivo)
    - max_results: Quantidade máxima de sugestões a retornar (padrão: 3)

    Observações Importantes:
    - A fonte de dados vem de arquivos JSON localizados em 'files/'
    - O sistema aplica filtro rigoroso por trilhas com status 'Published'
    - Validação automática de tamanhos e formatos de entrada
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    user_question: str = Field(
        ...,
        min_length=8,
        max_length=500,
        description="Pergunta/dúvida do jovem em linguagem natural.",
        examples=["Quero aprender lógica de programação do zero, por onde começo?"],
    )
    user_id: Optional[UUID] = Field(
        default=None,
        description="UUID do usuário, se disponível, para personalização baseada no perfil/progresso.",
        examples=["3f1a9c3e-8b4c-4cb7-9c31-2cf3f0a2a8d3"],
    )
    contexto_extra: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=500,
        description="Contexto adicional livre (ex.: área de interesse, objetivo, nível atual).",
        examples=["Interesse em front-end e JavaScript; nível iniciante; 30 min/dia."],
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Número máximo de trilhas sugeridas a retornar (1 a 3)."
    )
