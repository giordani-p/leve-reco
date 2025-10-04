#!/usr/bin/env python3
"""
Sistema de logging para consultas e resultados de recomendação.
Registra sessões de recomendação com modelo usado para monitoramento operacional.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class RecommendationQuery:
    """Representa uma consulta de recomendação."""
    query_id: str
    timestamp: str
    user_profile: Dict[str, Any]
    query_text: str
    num_recommendations: int
    filters: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

@dataclass
class RecommendationResult:
    """Representa um resultado de recomendação."""
    trail_id: str
    title: str
    score: float
    rank: int
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RecommendationSession:
    """Representa uma sessão completa de recomendação."""
    session_id: str
    timestamp: str
    query: RecommendationQuery
    results: List[RecommendationResult]
    execution_time_ms: float
    model_name: str
    total_trails_considered: int
    success: bool
    error_message: Optional[str] = None

class RecommendationLogger:
    """Logger para sistema de recomendação."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo principal de logs
        self.main_log_file = self.log_dir / "recommendation_logs.jsonl"
        
        # Inicializa arquivo se não existir
        self._initialize_files()
    
    def _initialize_files(self):
        """Inicializa arquivo de log se não existir."""
        if not self.main_log_file.exists():
            with open(self.main_log_file, 'w', encoding='utf-8') as f:
                f.write("")  # Arquivo vazio
    
    def log_recommendation_session(self, 
                                 user_profile: Dict[str, Any],
                                 query_text: str,
                                 results: List[Dict[str, Any]],
                                 execution_time_ms: float,
                                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                                 num_recommendations: int = 5,
                                 filters: Optional[Dict[str, Any]] = None,
                                 success: bool = True,
                                 error_message: Optional[str] = None) -> str:
        """
        Registra uma sessão completa de recomendação.
        
        Returns:
            str: ID da sessão registrada
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Cria query
        query = RecommendationQuery(
            query_id=str(uuid.uuid4()),
            timestamp=timestamp,
            user_profile=user_profile,
            query_text=query_text,
            num_recommendations=num_recommendations,
            filters=filters,
            session_id=session_id
        )
        
        # Converte resultados
        recommendation_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                rec_result = RecommendationResult(
                    trail_id=result.get('publicId', ''),
                    title=result.get('title', ''),
                    score=result.get('score', 0.0),
                    rank=i + 1,
                    explanation=result.get('explanation', ''),
                    metadata=result.get('metadata', {})
                )
                recommendation_results.append(rec_result)
        
        # Cria sessão
        session = RecommendationSession(
            session_id=session_id,
            timestamp=timestamp,
            query=query,
            results=recommendation_results,
            execution_time_ms=execution_time_ms,
            model_name=model_name,
            total_trails_considered=len(results),
            success=success,
            error_message=error_message
        )
        
        # Salva no arquivo principal (JSONL)
        self._append_to_main_log(session)
        
        return session_id
    
    def _append_to_main_log(self, session: RecommendationSession):
        """Adiciona sessão ao arquivo principal de logs."""
        session_dict = asdict(session)
        
        with open(self.main_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(session_dict, ensure_ascii=False) + '\n')
    
    
    def get_recommendation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recupera histórico de recomendações."""
        sessions = []
        
        with open(self.main_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    session = json.loads(line)
                    sessions.append(session)
        
        return sessions[-limit:] if limit else sessions
    
    
    def export_for_analysis(self, output_file: str = "recommendation_analysis.csv"):
        """Exporta dados para análise em CSV."""
        import csv
        
        sessions = self.get_recommendation_history()
        
        if not sessions:
            print("Nenhuma sessão para exportar")
            return
        
        output_path = self.log_dir / output_file
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Cabeçalho
            writer.writerow([
                'session_id', 'timestamp', 'model_name',
                'success', 'execution_time_ms', 'num_recommendations',
                'user_age', 'user_objetivo_principal', 'query_text',
                'top_recommendation_title', 'top_recommendation_score'
            ])
            
            # Dados
            for session in sessions:
                user_profile = session['query']['user_profile']
                results = session['results']
                
                top_recommendation = results[0] if results else {}
                
                writer.writerow([
                    session['session_id'],
                    session['timestamp'],
                    session['model_name'],
                    session['success'],
                    session['execution_time_ms'],
                    session['total_trails_considered'],
                    user_profile.get('dados_pessoais', {}).get('idade', ''),
                    user_profile.get('objetivos_carreira', {}).get('objetivo_principal', ''),
                    session['query']['query_text'],
                    top_recommendation.get('title', ''),
                    top_recommendation.get('score', 0)
                ])
        
        print(f"Dados exportados para {output_path}")
    

# Instância global do logger
recommendation_logger = RecommendationLogger()

def log_recommendation(user_profile: Dict[str, Any],
                      query_text: str,
                      results: List[Dict[str, Any]],
                      execution_time_ms: float,
                      **kwargs) -> str:
    """Função de conveniência para logging rápido."""
    return recommendation_logger.log_recommendation_session(
        user_profile=user_profile,
        query_text=query_text,
        results=results,
        execution_time_ms=execution_time_ms,
        **kwargs
    )
