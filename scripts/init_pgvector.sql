-- PatentRank Phase 2: PGVector Database Schema
-- Enables pgvector extension and creates patent_embeddings table with HNSW cosine index.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS patent_embeddings (
    doc_id VARCHAR(64) PRIMARY KEY,
    title TEXT,
    category VARCHAR(32),
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW Index for ultra-fast Approximate Nearest Neighbor (ANN) search using cosine distance
CREATE INDEX IF NOT EXISTS patent_embeddings_hnsw_idx 
ON patent_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
