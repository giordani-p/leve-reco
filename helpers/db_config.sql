-- Habilita extensões (executado uma única vez quando o volume é inicializado)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Schema principal da aplicação
CREATE SCHEMA IF NOT EXISTS leve;

-- Define a ordem de busca: procura primeiro no schema 'leve', depois em 'public'
ALTER DATABASE leve_core SET search_path TO leve, public;

-- ============================================
-- FTS para leve.trails (português + unaccent)
-- Requisitos: coluna combined_text (text) e search_vector (tsvector) já criadas via Prisma
-- ============================================

-- Garante stemming/stopwords pt-br por padrão (para esta conexão/execução)
SET search_path TO leve, public;
SET default_text_search_config = 'pg_catalog.portuguese';

-- Function: mantém search_vector em sincronia com combined_text
-- Idempotente via OR REPLACE
CREATE OR REPLACE FUNCTION leve.update_trails_search_vector()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  -- Evita recomputar à toa
  IF TG_OP = 'INSERT'
     OR NEW.combined_text IS DISTINCT FROM OLD.combined_text
     OR NEW.search_vector IS NULL
  THEN
    NEW.search_vector :=
      to_tsvector('portuguese', unaccent(COALESCE(NEW.combined_text, '')));
  END IF;

  RETURN NEW;
END;
$$;

-- Trigger: BEFORE INSERT/UPDATE em leve.trails
-- Usa IF NOT EXISTS para evitar erro se a tabela não existir ainda
DO $$
BEGIN
  -- Verifica se a tabela existe antes de criar o trigger
  IF EXISTS (
    SELECT 1
    FROM   information_schema.tables
    WHERE  table_schema = 'leve'
      AND  table_name = 'trails'
  ) THEN
    -- Remove trigger existente se houver
    DROP TRIGGER IF EXISTS trails_search_vector_trigger ON leve.trails;

    -- Cria o trigger
    CREATE TRIGGER trails_search_vector_trigger
      BEFORE INSERT OR UPDATE OF combined_text
      ON leve.trails
      FOR EACH ROW
      EXECUTE FUNCTION leve.update_trails_search_vector();
  END IF;
END;
$$;

-- Índice GIN em search_vector (acelera FTS)
-- Usa IF NOT EXISTS para evitar erro se a tabela não existir ainda
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM   information_schema.tables
    WHERE  table_schema = 'leve'
      AND  table_name = 'trails'
  ) THEN
    CREATE INDEX IF NOT EXISTS trails_search_vector_idx
      ON leve.trails
      USING GIN (search_vector);
  END IF;
END;
$$;

