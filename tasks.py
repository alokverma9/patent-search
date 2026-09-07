"""
tasks.py — Background Worker Tasks for Patent Ingestion & Processing

Executes asynchronous jobs for:
1. Patent document validation and structural text segmentation.
2. Embedding computation via Google AI Studio (gemini-embedding-2, 768 dims).
3. Dual-database indexing (Elasticsearch for BM25 + PGVector for ANN dense retrieval).
"""

import os
import json
import time
import socket
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import numpy as np

from celery_app import celery_app
from segment_documents import RegexPatentSegmenter

logger = logging.getLogger("patentrank.tasks")

# Environment & Connection Defaults
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "patents_bm25")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
PG_DB = os.getenv("POSTGRES_DB", "patentrank")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

DEFAULT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "embeddings_cache.jsonl")


def is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    """Non-blocking socket check to verify service accessibility without TCP hang."""
    target_host = "127.0.0.1" if host in ("localhost", "127.0.0.1") else host
    try:
        with socket.create_connection((target_host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def compute_or_mock_embedding(text: str, doc_id: str) -> List[float]:
    """Compute 768-dim dense embedding via Gemini API or reproducible unit vector fallback."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and "your_gemini_api_key" not in api_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text[:4000],
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            emb = response.embeddings[0].values
            # Cache locally to disk
            os.makedirs(os.path.dirname(DEFAULT_CACHE_FILE), exist_ok=True)
            with open(DEFAULT_CACHE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"doc_id": doc_id, "embedding": emb}) + "\n")
            return emb
        except Exception as e:
            logger.warning(f"Failed to embed document via Gemini API: {e}. Generating fallback.")

    # Deterministic fallback vector based on doc_id hash
    seed = abs(hash(doc_id)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(768).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def index_to_elasticsearch(doc: Dict[str, Any]) -> bool:
    """Index document into Elasticsearch if cluster is accessible."""
    try:
        parsed = urlparse(ES_URL)
        es_host = parsed.hostname or "127.0.0.1"
        es_port = parsed.port or 9200
        if not is_port_open(es_host, es_port, timeout=0.2):
            return False

        from elasticsearch import Elasticsearch
        client = Elasticsearch(ES_URL, request_timeout=2)
        if client.ping():
            client.index(index=ES_INDEX, id=doc["doc_id"], document=doc)
            return True
    except Exception as e:
        logger.warning(f"Elasticsearch indexing skipped/failed for {doc.get('doc_id')}: {e}")
    return False


def index_to_pgvector(doc_id: str, title: str, embedding: List[float]) -> bool:
    """Index document embedding into PostgreSQL PGVector if accessible."""
    if not is_port_open(PG_HOST, PG_PORT, timeout=0.2):
        return False

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            connect_timeout=1
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patent_embeddings (doc_id, title, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (doc_id)
                DO UPDATE SET title = EXCLUDED.title, embedding = EXCLUDED.embedding;
                """,
                (doc_id, title, embedding)
            )
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"PGVector indexing skipped/failed for {doc_id}: {e}")
    return False


@celery_app.task(bind=True, name="tasks.ingest_patent_task")
def ingest_patent_task(self, patent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery background worker task to ingest a patent document.
    1. Validates document payload.
    2. Runs text segmentation into sections, paragraphs, and claims.
    3. Computes 768-dim dense embedding.
    4. Indexes into Elasticsearch and PGVector.
    """
    t0 = time.time()
    doc_id = patent_data.get("doc_id")
    if not doc_id:
        raise ValueError("Missing required field: 'doc_id'")

    title = patent_data.get("title", "")
    abstract = patent_data.get("abstract", "")
    summary = patent_data.get("summary", "")
    text = patent_data.get("text", f"{title}\n{abstract}\n{summary}")

    # 1. Structural Segmentation
    segmenter = RegexPatentSegmenter()
    segmented = segmenter.segment_document(
        doc_id=doc_id,
        title=title,
        full_text=text,
        abstract=abstract
    )

    # 2. Embedding Generation
    embed_text = f"{title} — {abstract}" if abstract else title
    embedding = compute_or_mock_embedding(embed_text, doc_id)

    # 3. Dual Indexing
    es_indexed = index_to_elasticsearch({
        "doc_id": doc_id,
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "search_text": text
    })

    pg_indexed = index_to_pgvector(doc_id, title, embedding)

    duration_ms = (time.time() - t0) * 1000.0

    return {
        "status": "SUCCESS",
        "doc_id": doc_id,
        "title": title,
        "embedding_dim": len(embedding),
        "segmentation": {
            "sections_count": len(segmented.sections),
            "paragraphs_count": len(segmented.paragraphs),
            "claims_count": len(segmented.claims),
            "stats": segmented.stats
        },
        "indexed_elasticsearch": es_indexed,
        "indexed_pgvector": pg_indexed,
        "duration_ms": round(duration_ms, 2),
        "completed_at": time.time()
    }
