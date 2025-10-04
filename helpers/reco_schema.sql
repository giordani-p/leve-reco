-- ============================================
-- Sistema de Recomendação - Schema PostgreSQL + pgvector
-- Complementa o FTS existente em db_config.sql
-- ============================================

-- Schema para sistema de recomendação
CREATE SCHEMA IF NOT EXISTS reco;

-- Tabela de embeddings (referencia leve.trails por public_id)
CREATE TABLE reco.trail_embeddings (
    public_id UUID PRIMARY KEY,
    embedding VECTOR(768) NOT NULL,
    model_version VARCHAR(100) NOT NULL DEFAULT 'paraphrase-multilingual-mpnet-base-v2',
    content_hash VARCHAR(64) NOT NULL,  -- SHA256 do combined_text para detectar mudanças
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Índices para busca vetorial
CREATE INDEX trail_embeddings_vector_idx 
ON reco.trail_embeddings 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Comentário do índice
COMMENT ON INDEX trail_embeddings_vector_idx IS 'Índice ivfflat para busca por similaridade de cosseno';

-- Índices para sincronização e performance
CREATE INDEX trail_embeddings_public_id_idx 
ON reco.trail_embeddings (public_id);

CREATE INDEX trail_embeddings_content_hash_idx 
ON reco.trail_embeddings (content_hash);

CREATE INDEX trail_embeddings_model_version_idx 
ON reco.trail_embeddings (model_version);

-- Índice para metadados (filtros por status, difficulty, area)
CREATE INDEX trail_embeddings_metadata_idx 
ON reco.trail_embeddings USING GIN (metadata);

-- View para consultas unificadas (opcional)
CREATE VIEW reco.trails_with_embeddings AS
SELECT 
    t.*,
    e.embedding,
    e.model_version,
    e.content_hash,
    e.metadata as embedding_metadata,
    e.updated_at as embedding_updated_at
FROM leve.trails t
LEFT JOIN reco.trail_embeddings e ON t.public_id = e.public_id;

-- Function para detectar mudanças no conteúdo
CREATE OR REPLACE FUNCTION reco.detect_content_change()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    -- Se combined_text mudou, marca embedding como desatualizado
    IF TG_OP = 'UPDATE' AND NEW.combined_text IS DISTINCT FROM OLD.combined_text THEN
        -- Remove embedding desatualizado
        DELETE FROM reco.trail_embeddings 
        WHERE public_id = NEW.public_id;
    END IF;
    
    RETURN NEW;
END;
$$;

-- Trigger para invalidar embeddings quando conteúdo muda
-- Só cria se a tabela leve.trails existir
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'leve'
        AND table_name = 'trails'
    ) THEN
        -- Remove trigger existente se houver
        DROP TRIGGER IF EXISTS reco_trails_embedding_invalidation_trigger ON leve.trails;
        
        -- Cria o trigger
        CREATE TRIGGER reco_trails_embedding_invalidation_trigger
            AFTER UPDATE OF combined_text
            ON leve.trails
            FOR EACH ROW
            EXECUTE FUNCTION reco.detect_content_change();
    END IF;
END;
$$;

-- Function para buscar trilhas por similaridade vetorial
CREATE OR REPLACE FUNCTION reco.search_trails_by_embedding(
    query_embedding VECTOR(768),
    similarity_threshold FLOAT DEFAULT 0.0,
    max_results INTEGER DEFAULT 50,
    status_filter VARCHAR DEFAULT NULL,
    difficulty_filter VARCHAR DEFAULT NULL,
    area_filter VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    public_id UUID,
    similarity_score FLOAT,
    metadata JSONB,
    title VARCHAR,
    subtitle VARCHAR,
    description TEXT
) 
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        te.public_id,
        1 - (te.embedding <=> query_embedding) as similarity_score,
        te.metadata,
        t.title,
        t.subtitle,
        t.description
    FROM reco.trail_embeddings te
    JOIN leve.trails t ON t.public_id = te.public_id
    WHERE 
        (1 - (te.embedding <=> query_embedding)) >= similarity_threshold
        AND (status_filter IS NULL OR te.metadata->>'status' = status_filter)
        AND (difficulty_filter IS NULL OR te.metadata->>'difficulty' = difficulty_filter)
        AND (area_filter IS NULL OR te.metadata->>'area' = area_filter)
    ORDER BY te.embedding <=> query_embedding
    LIMIT max_results;
END;
$$;

-- Function para estatísticas do sistema
CREATE OR REPLACE FUNCTION reco.get_embedding_stats()
RETURNS TABLE (
    total_embeddings BIGINT,
    model_versions TEXT[],
    last_updated TIMESTAMP,
    avg_similarity_threshold FLOAT
) 
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_embeddings,
        ARRAY_AGG(DISTINCT model_version) as model_versions,
        MAX(updated_at) as last_updated,
        0.0 as avg_similarity_threshold  -- Placeholder
    FROM reco.trail_embeddings;
END;
$$;

-- Comentários para documentação
COMMENT ON SCHEMA reco IS 'Schema para sistema de recomendação com embeddings vetoriais';
COMMENT ON TABLE reco.trail_embeddings IS 'Armazena embeddings MPNet para busca semântica de trilhas';
COMMENT ON COLUMN reco.trail_embeddings.public_id IS 'Referência para leve.trails.public_id';
COMMENT ON COLUMN reco.trail_embeddings.embedding IS 'Vetor MPNet 768-dimensional normalizado L2';
COMMENT ON COLUMN reco.trail_embeddings.content_hash IS 'SHA256 do combined_text para detectar mudanças';
COMMENT ON COLUMN reco.trail_embeddings.metadata IS 'Metadados: status, difficulty, area, etc.';
COMMENT ON FUNCTION reco.search_trails_by_embedding IS 'Busca trilhas por similaridade vetorial com filtros';
