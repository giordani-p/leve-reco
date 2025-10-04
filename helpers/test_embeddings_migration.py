#!/usr/bin/env python3
"""
Script para testar a migração completa de embeddings da API.

Este script:
1. Limpa todos os embeddings existentes
2. Migra embeddings da API
3. Valida o resultado
4. Mostra estatísticas detalhadas
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Executa um comando e mostra o resultado."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("OK Sucesso!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR Erro: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    """Executa o teste completo de migração."""
    print("Iniciando teste completo de migração de embeddings")
    print("Este script irá:")
    print("1. Limpar todos os embeddings existentes")
    print("2. Migrar embeddings da API")
    print("3. Validar o resultado")
    print("4. Mostrar estatísticas")
    
    # Confirmar execução
    response = input("\nDeseja continuar? (y/N): ").strip().lower()
    if response != 'y':
        print("Operação cancelada.")
        return
    
    # 1. Limpar todos os embeddings
    if not run_command(
        "python cli/migrate_embeddings.py clear-all --confirm --verbose",
        "Limpando todos os embeddings existentes"
    ):
        print("ERROR Falha na limpeza. Abortando teste.")
        return
    
    # 2. Verificar status após limpeza
    run_command(
        "python cli/migrate_embeddings.py validate --verbose",
        "Verificando status após limpeza"
    )
    
    # 3. Migrar da API
    if not run_command(
        "python cli/migrate_embeddings.py from-api --verbose --batch-size 10",
        "Migrando embeddings da API"
    ):
        print("ERROR Falha na migração. Abortando teste.")
        return
    
    # 4. Validar migração
    if not run_command(
        "python cli/migrate_embeddings.py validate --verbose",
        "Validando migração final"
    ):
        print("ERROR Validação falhou.")
        return
    
    print(f"\n{'='*60}")
    print("Teste de migração concluído com sucesso!")
    print("OK Embeddings limpos")
    print("OK Embeddings migrados da API")
    print("OK Validação aprovada")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
