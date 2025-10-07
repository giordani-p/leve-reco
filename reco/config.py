# reco/config.py
"""
Configuração Central do Sistema de Recomendação - Leve Agents

Este módulo centraliza todas as configurações do sistema de recomendação híbrido,
que combina busca semântica (MPNet) e busca textual (BM25) para recomendar
trilhas educacionais personalizadas para jovens brasileiros.

Arquitetura do Sistema:
- Sistema híbrido: combina BM25 (busca textual) com MPNet (busca semântica)
- Normalização inteligente de scores para otimizar a combinação
- Thresholds específicos por tipo de conteúdo (trilhas vs vagas)
- Expansão de consulta com sinônimos (apenas no BM25)
- Integração completa com API da Leve
- Armazenamento persistente usando PostgreSQL + pgvector

Funcionalidades Principais:
- Recomendação personalizada baseada no perfil psicológico do usuário
- Integração robusta com API da Leve
- Sistema de ranking com boosts e filtros de negócio
- Observabilidade completa com logs estruturados
- Configuração flexível para diferentes cenários de uso
- Sincronização automática de embeddings com banco de dados
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Literal, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




@dataclass(frozen=True)
class RecoConfig:
    # -----------------------------
    # Regras de Negócio e Sistema de Ranking
    # -----------------------------
    # Número máximo de sugestões retornadas ao usuário
    MAX_SUGGESTIONS: int = 3
    
    # Boosts aplicados durante o ranking para melhorar a relevância
    TAG_BOOST: float = 0.08  # Boost quando tags da trilha aparecem na consulta
    BEGINNER_BOOST: float = 0.05  # Boost para trilhas de nível iniciante
    TITLE_DESC_BOOST: float = 0.12  # Boost quando palavras-chave aparecem no título/descrição
    EDUCATIONAL_RELEVANCE_BOOST: float = 0.10  # Boost para relevância educacional
    
    # Limites de score para evitar valores extremos
    SCORE_CAP: float = 0.99  # Score máximo permitido
    DOMINANCE_MIN_ACCEPT: float = 0.70  # Score mínimo para fallback de dominância

    # Thresholds específicos por tipo de conteúdo (qualidade semântica)
    MATCH_THRESHOLD_TRILHAS: float = 0.80  # Threshold para trilhas educacionais
    MATCH_THRESHOLD_VAGAS: float = 0.85  # Threshold para vagas (futuro P2)

    # Filtro de status das trilhas (apenas Published na fase atual)
    ALLOWED_STATUS: Tuple[str, ...] = ("Published",)

    # Configurações para uso de pistas do perfil do usuário
    USE_SNAPSHOT_HINTS: bool = True  # Habilita uso de pistas do snapshot
    SNAPSHOT_MAX_HINTS: int = 3  # Número máximo de pistas a serem utilizadas

    # Limite opcional de tamanho da consulta construída
    QUERY_MAX_CHARS: int = 0  # 0 = sem limite de caracteres

    # -----------------------------
    # Sistema de Embeddings e Busca Densa (MPNet)
    # -----------------------------
    # Modelo de embeddings multilíngue para busca semântica
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBED_DIM: int = 768  # Dimensão dos vetores de embedding
    EMBEDDING_DEVICE: Optional[str] = None  # Dispositivo: "cpu" | "cuda" | None (auto)
    EMBEDDING_BATCH_SIZE: int = 64  # Tamanho do lote para processamento de embeddings
    TOP_K_DEFAULT: int = 50  # Número de resultados brutos retornados pela busca densa

    # -----------------------------
    # Sistema Híbrido BM25 + Busca Densa
    # -----------------------------
    USE_HYBRID: bool = True  # Habilita combinação de BM25 e busca densa
    FUSION_METHOD: Literal["weighted", "rrf"] = "weighted"  # Método de fusão dos scores
    WEIGHTS: Optional[Dict[str, float]] = None  # Pesos definidos em __post_init__
    NORMALIZATION: Literal["minmax", "zscore", "percentile", "robust"] = "robust"  # Método de normalização
    NORMALIZATION_EPS: float = 1e-6  # Epsilon para evitar divisão por zero

    # -----------------------------
    # Expansão de Consulta com Sinônimos
    # -----------------------------
    # Aplicar expansão apenas no BM25; manter desabilitado no sistema denso
    USE_QUERY_SYNONYMS_BM25: bool = True  # Habilita sinônimos para BM25
    USE_QUERY_SYNONYMS_DENSE: bool = False  # Mantém sinônimos desabilitados para busca densa
    QUERY_SYNONYMS: Optional[Dict[str, List[str]]] = None  # Dicionário de sinônimos definido em __post_init__

    # -----------------------------
    # Configurações Gerais
    # -----------------------------
    RANDOM_SEED: int = 42  # Semente para reprodutibilidade

    # -----------------------------
    # Integração com API da Leve
    # -----------------------------
    SOURCE: str = "api"  # Fonte de dados: "api" ou "files"
    TRAILS_API_BASE: str = "http://localhost:3030"  # URL base da API
    API_FILTER_PUBLISHED: bool = True  # Filtra apenas trilhas publicadas

    # Configurações de timeout para requisições HTTP (em segundos)
    HTTP_TIMEOUT_CONNECT: float = 3.0  # Timeout para conexão
    HTTP_TIMEOUT_READ: float = 10.0  # Timeout para leitura de resposta
    HTTP_TIMEOUT_WRITE: float = 3.0  # Timeout para escrita de requisição
    HTTP_TIMEOUT_POOL: float = 3.0  # Timeout para pool de conexões

    # Configurações de retry com backoff exponencial
    HTTP_RETRIES: int = 2  # Número de tentativas: 1 + HTTP_RETRIES
    HTTP_BACKOFF_BASE: float = 0.4  # Base do backoff exponencial

    # Limites de paginação para endpoints que suportam paginação
    API_MAX_PAGES: int = 10  # Número máximo de páginas a serem processadas
    API_PAGE_SIZE_HINT: Optional[int] = None  # Tamanho sugerido de página

    # -----------------------------
    # Versionamento e Índices Vetoriais
    # -----------------------------
    MODEL_VERSION: str = "mpnet_v1"  # Versão do modelo de embeddings
    INDEX_TRILHAS: str = "trilhas_mpnet_v1"  # Nome do índice para trilhas
    INDEX_VAGAS: str = "vagas_mpnet_v1"  # Nome do índice para vagas (futuro P2)

    # -----------------------------
    # Filtros de Segurança e Validação
    # -----------------------------
    ENFORCE_PUBLISHED: bool = True  # Força uso apenas de trilhas publicadas
    ENFORCE_LOCATION: bool = True  # Força validação de localização (vagas, P2)
    ENFORCE_SENIORITY: bool = True  # Força validação de senioridade (vagas, P2)
    ENFORCE_REGIME: bool = True  # Força validação de regime (vagas, P2)

    # -----------------------------
    # Sistema de Observabilidade e Logging
    # -----------------------------
    LOG_RETRIEVAL_DEBUG: bool = True  # Habilita logs detalhados de retrieval
    LOG_EXPLAIN_MATCH: bool = True  # Habilita logs de explicação de matches
    ANN_RANDOM_SEED: int = 42  # Semente para algoritmos de busca aproximada
    API_KEY: Optional[str] = None  # Chave de API opcional para data_loader_api

    # -----------------------------
    # Configurações do Banco de Dados (PostgreSQL + pgvector)
    # -----------------------------
    # Configurações individuais do banco de dados (mais seguras e flexíveis)
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "leve_core")
    
    # Schema e tabela do sistema de recomendação
    RECO_SCHEMA: str = "reco"  # Schema do sistema de recomendação
    MAIN_SCHEMA: str = "leve"  # Schema principal onde estão as trilhas
    EMBEDDING_TABLE: str = "trail_embeddings"  # Tabela de embeddings das trilhas
    
    # Configurações do pool de conexões (essenciais)
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))  # Tamanho do pool
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))  # Overflow máximo
    
    # Configurações de sincronização de embeddings (essenciais)
    SYNC_BATCH_SIZE: int = int(os.getenv("SYNC_BATCH_SIZE", "100"))  # Tamanho do lote de sincronização
    
    # Backend do índice vetorial
    USE_PERSISTENT_INDEX: bool = os.getenv("USE_PERSISTENT_INDEX", "true").lower() == "true"


    # -----------------------------
    # Configurações Derivadas e Pós-inicialização
    # -----------------------------
    def __post_init__(self):
        # Define pesos padrão para blending do sistema híbrido (prioriza busca semântica)
        if object.__getattribute__(self, "WEIGHTS") is None:
            object.__setattr__(self, "WEIGHTS", {"semantic": 0.75, "bm25": 0.25})

        # Define dicionário de sinônimos para expansão de consulta
        # Chaves em caixa-baixa e sem acento; o código normaliza antes de consultar
        if object.__getattribute__(self, "QUERY_SYNONYMS") is None:
            base = {
                "programacao": ["programar", "programador", "coding", "coder", "desenvolver", "dev"],
                "logica": ["algoritmo", "algoritmos", "raciocinio"],
                "javascript": ["js", "java script"],
                "iniciante": ["beginner", "basico", "do zero"],
                "automatizar": ["automacao", "script", "scripts", "macro", "automatizado"],
                "trabalho": ["empresa", "servico", "expediente"],
                "dados": ["data", "csv", "planilha", "planilhas", "analise"],
                "excel": ["planilha", "planilhas", "spreadsheet", "xls", "xlsx"],
                "python": ["py", "pythonico", "pythonista"],
                "ia": ["inteligencia artificial", "ai"],
                "ux": ["user experience", "ux design"],
                "ui": ["user interface", "ui design"],
                "direito": ["juridico", "juridica"],
                "saude": ["saude mental"],
                "nutricao": ["nutricional"],
            }
            object.__setattr__(self, "QUERY_SYNONYMS", base)
