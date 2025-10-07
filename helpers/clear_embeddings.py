#!/usr/bin/env python3
"""
Script rápido para limpar todos os embeddings do banco.

Uso:
    python helpers/clear_embeddings.py
    python helpers/clear_embeddings.py --confirm  # Pula confirmação
"""

import sys
from pathlib import Path

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reco.config import RecoConfig
from reco.database.connection import DatabaseConnection

def main():
    """Limpa todos os embeddings do banco."""
    print("Limpeza de Embeddings")
    print("=" * 40)
    
    # Verificar se deve pular confirmação
    skip_confirm = "--confirm" in sys.argv
    
    if not skip_confirm:
        print("ATENÇÃO: Isso irá deletar TODOS os embeddings do banco!")
        response = input("Tem certeza? Digite 'SIM' para confirmar: ").strip()
        if response != "SIM":
            print("Operação cancelada.")
            return
    
    # Conectar ao banco
    cfg = RecoConfig()
    db = DatabaseConnection(cfg)
    
    try:
        # Verificar quantos embeddings existem
        stats = db.get_embedding_stats()
        total = stats.get('total_embeddings', 0)
        
        print(f"Embeddings encontrados: {total}")
        
        if total == 0:
            print("Nenhum embedding encontrado - nada para limpar.")
            return
        
        # Limpar todos os embeddings
        print("Limpando embeddings...")
        query = f"DELETE FROM {cfg.RECO_SCHEMA}.trail_embeddings"
        db.execute_query(query)
        
        # Verificar resultado
        new_stats = db.get_embedding_stats()
        remaining = new_stats.get('total_embeddings', 0)
        
        print(f"Limpeza concluída!")
        print(f"Embeddings restantes: {remaining}")
        
        if remaining == 0:
            print("Todos os embeddings foram removidos com sucesso!")
        else:
            print(f"Ainda restam {remaining} embeddings")
    
    except Exception as e:
        print(f"Erro durante a limpeza: {e}")
        sys.exit(1)
    
    finally:
        db.close()

if __name__ == "__main__":
    main()
