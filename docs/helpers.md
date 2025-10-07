# Helpers - Sistema de Recomendação

Este diretório contém scripts auxiliares para gerenciar o sistema de recomendação baseado em embeddings vetoriais.

## 📁 Arquivos de Configuração

### `db_config.sql`
**Propósito:** Configuração inicial do banco PostgreSQL
- Habilita extensões `vector` e `unaccent`
- Cria schema `leve` para a aplicação principal
- Configura Full Text Search (FTS) em português
- Cria triggers para manter `search_vector` sincronizado
- **Uso:** Executar uma única vez na inicialização do banco

### `reco_schema.sql`
**Propósito:** Schema completo do sistema de recomendação
- Cria schema `reco` para embeddings
- Define tabela `trail_embeddings` com vetores 768D
- Configura índices ivfflat para busca vetorial
- Cria funções de busca por similaridade
- **Uso:** Executar após `db_config.sql`

## 🛠️ Scripts de Migração

### `migrate_embeddings.py`
**Propósito:** CLI principal para gerenciar embeddings
- **Comandos disponíveis:**
  - `setup-schema` - Configura schema do banco
  - `from-files` - Migra de arquivo JSON local
  - `from-api` - Migra da API
  - `validate` - Valida status da migração
  - `clear-all` - Remove todos os embeddings
  - `cleanup` - Limpa embeddings órfãos
- **Interface:** Rich (tabelas, progress bars, cores)
- **Uso:** `python helpers/migrate_embeddings.py <comando>`

### `test_embeddings_migration.py`
**Propósito:** Script de teste automatizado
- Executa fluxo completo: limpar → migrar → validar
- Orquestra comandos do `migrate_embeddings.py`
- Confirmação interativa antes de executar
- **Uso:** `python helpers/test_embeddings_migration.py`

### `clear_embeddings.py`
**Propósito:** Script simples para limpeza rápida
- Remove todos os embeddings do banco
- Confirmação de segurança obrigatória
- Mostra estatísticas antes/depois
- **Uso:** `python helpers/clear_embeddings.py [--confirm]`

## 🔧 Scripts de Teste

### `test_db_connection.py`
**Propósito:** Testa conectividade e configuração do banco
- Verifica conexão PostgreSQL + pgvector
- Testa extensões e schemas
- Valida operações vetoriais
- Mostra estatísticas do banco
- **Uso:** `python helpers/test_db_connection.py`

## 📊 Utilitários

### `json_extractor.py`
**Propósito:** Extrai JSON de respostas de LLM
- Função `try_extract_json()` para parsing robusto
- Suporta blocos ```json``` e JSON livre
- Limpa caracteres problemáticos
- **Uso:** Import como módulo

### `snapshot_selector.py`
**Propósito:** Seleção interativa de snapshots de usuários
- Carrega usuários de `files/snapshots/users_faker.json`
- Interface de seleção por nome/índice
- Opção de colar JSON manualmente
- **Uso:** Import como módulo

## 🚀 Fluxo de Uso Recomendado

1. **Configuração inicial:**
   ```bash
   # 1. Configurar banco
   psql -d your_database -f helpers/db_config.sql
   psql -d your_database -f helpers/reco_schema.sql
   
   # 2. Testar conexão
   python helpers/test_db_connection.py
   ```

2. **Migração de embeddings:**
   ```bash
   # Opção 1: Migração manual
   python helpers/migrate_embeddings.py setup-schema
   python helpers/migrate_embeddings.py from-api --verbose
   
   # Opção 2: Teste automatizado
   python helpers/test_embeddings_migration.py
   ```

3. **Validação:**
   ```bash
   python helpers/migrate_embeddings.py validate --verbose
   ```

## 📋 Dependências

- **Python:** 3.8+
- **PostgreSQL:** 13+ com pgvector
- **Python packages:** typer, rich, psycopg2, numpy, transformers

## ⚠️ Notas Importantes

- Execute `db_config.sql` e `reco_schema.sql` apenas uma vez
- Use `--confirm` com cuidado em operações destrutivas
- O script `test_embeddings_migration.py` limpa TODOS os embeddings
- Sempre valide após migrações importantes
