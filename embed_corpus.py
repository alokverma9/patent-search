"""
embed_corpus.py — PatentRank Phase 2: Dense Corpus Embedding & PGVector Ingestion

Features:
1. Connects to Google AI Studio Gemini API using the official google-genai SDK.
2. Uses gemini-embedding-2 with Matryoshka dimension reduction (768 dims) for optimal PGVector indexing.
3. Automatically caches embeddings to data/embeddings_cache.jsonl to protect API quota across runs.
4. Prioritizes evaluation dataset documents (ground truth targets + distractors) first.
5. Rate-limit aware batching with exponential backoff and jitter for 429 quota exhaustion.
6. Bulk ingests embeddings into PostgreSQL pgvector table with HNSW index.
"""

import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Any, Set
from dotenv import load_dotenv
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

# Configuration Defaults
DEFAULT_CORPUS_FILE = os.path.join("data", "patents_raw.jsonl")
DEFAULT_QUERIES_FILE = os.path.join("data", "queries_labeled.jsonl")
DEFAULT_CACHE_FILE = os.path.join("data", "embeddings_cache.jsonl")
DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIM = 768
DEFAULT_BATCH_SIZE = 20
DEFAULT_DELAY = 12.0  # 20 items every 12s = ~100 items/minute (within free tier quota)

# PGVector connection defaults
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5433"))
PG_DB = os.getenv("POSTGRES_DB", "patentrank")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


def get_pg_connection():
    """Establish connection to PostgreSQL pgvector instance."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )


def init_pgvector_schema(conn, dim: int = DEFAULT_DIM):
    """Ensure pgvector extension and patent_embeddings table exist with HNSW index."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS patent_embeddings (
                doc_id VARCHAR(64) PRIMARY KEY,
                title TEXT,
                category VARCHAR(32),
                embedding vector({dim}),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS patent_embeddings_hnsw_idx 
            ON patent_embeddings 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
    conn.commit()
    print(f"[OK] PGVector schema initialized (table 'patent_embeddings', dim={dim}, HNSW cosine index).")


def load_cache(cache_file: str) -> Dict[str, List[float]]:
    """Load cached document embeddings from disk."""
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    cache[item["doc_id"]] = item["embedding"]
        print(f"[OK] Loaded {len(cache):,} cached document embeddings from {cache_file}.")
    return cache


def append_to_cache(cache_file: str, records: List[Dict[str, Any]]):
    """Append newly computed embeddings to disk cache immediately."""
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    with open(cache_file, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"doc_id": r["doc_id"], "embedding": r["embedding"]}) + "\n")


def get_gemini_client():
    """Instantiate Google GenAI Client with validation."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or "your_gemini_api_key" in api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
    from google import genai
    return genai.Client(api_key=api_key)


def embed_batch_with_retry(
    client,
    docs: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    dim: int = DEFAULT_DIM,
    max_retries: int = 5
) -> List[List[float]]:
    """Call Gemini embed_content API with adaptive retry and backoff on HTTP 429."""
    from google.genai import types

    # Prepare document texts: Title + Abstract provides maximum semantic precision
    contents = []
    for doc in docs:
        text = f"Title: {doc.get('title', '')}\nAbstract: {doc.get('abstract', '')}"
        # Truncate if extreme length (rare for title + abstract)
        if len(text) > 8000:
            text = text[:8000]
        contents.append(types.Content(parts=[types.Part.from_text(text=text)]))

    config = types.EmbedContentConfig(output_dimensionality=dim)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=contents,
                config=config
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_time = 35 + (attempt * 10) + random.uniform(1, 5)
                print(f"\n[Rate Limit] 429 Quota Exceeded. Sleeping {wait_time:.1f}s before retry (attempt {attempt}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"\n[API Error] {e} on attempt {attempt}/{max_retries}")
                if attempt == max_retries:
                    raise e
                time.sleep(5 * attempt)

    raise RuntimeError(f"Failed to embed batch of {len(docs)} docs after {max_retries} retries.")


def sync_cache_to_pgvector(conn, corpus_map: Dict[str, Dict[str, Any]], cache: Dict[str, List[float]]):
    """Bulk insert/upsert all cached embeddings into PGVector table."""
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM patent_embeddings;")
        existing_in_db = set(row[0] for row in cur.fetchall())

    to_insert = []
    for doc_id, emb in cache.items():
        if doc_id not in existing_in_db:
            doc_info = corpus_map.get(doc_id, {})
            to_insert.append((
                doc_id,
                doc_info.get("title", ""),
                doc_info.get("category", "G"),
                emb
            ))

    if to_insert:
        print(f"Syncing {len(to_insert):,} new embeddings from cache to PGVector...")
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO patent_embeddings (doc_id, title, category, embedding)
                VALUES %s
                ON CONFLICT (doc_id) DO NOTHING;
                """,
                to_insert,
                page_size=500
            )
        conn.commit()
        print(f"[OK] Successfully synced {len(to_insert):,} embeddings into PGVector.")
    else:
        print(f"[OK] PGVector is already fully in sync with disk cache ({len(existing_in_db):,} records).")


def main():
    parser = argparse.ArgumentParser(description="PatentRank Phase 2: Dense Corpus Embedding & PGVector Ingestion")
    parser.add_argument("--corpus-file", default=DEFAULT_CORPUS_FILE, help="Path to patents_raw.jsonl")
    parser.add_argument("--queries-file", default=DEFAULT_QUERIES_FILE, help="Path to queries_labeled.jsonl")
    parser.add_argument("--cache-file", default=DEFAULT_CACHE_FILE, help="Path to embeddings_cache.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini embedding model name")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Embedding dimension (default: 768)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Documents per API call")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Delay between API batches in seconds")
    parser.add_argument("--limit", type=int, default=500, help="Total number of documents to embed in this run (use -1 for all)")
    parser.add_argument("--sync-db-only", action="store_true", help="Only sync existing cache to PGVector without calling API")
    args = parser.parse_args()

    # 1. Connect to PGVector
    print(f"Connecting to PostgreSQL (PGVector) on port {PG_PORT}...")
    conn = get_pg_connection()
    init_pgvector_schema(conn, dim=args.dim)

    # 2. Load Corpus
    print(f"Loading corpus from {args.corpus_file}...")
    corpus_map: Dict[str, Dict[str, Any]] = {}
    with open(args.corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                corpus_map[d["doc_id"]] = d
    print(f"[OK] Loaded {len(corpus_map):,} documents from corpus.")

    # 3. Load Cache
    cache = load_cache(args.cache_file)

    # If sync-only requested, push cache to DB and exit
    if args.sync_db_only:
        sync_cache_to_pgvector(conn, corpus_map, cache)
        conn.close()
        return

    # 4. Identify and Prioritize Test Evaluation Documents First
    eval_doc_ids: Set[str] = set()
    if os.path.exists(args.queries_file):
        with open(args.queries_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    eval_doc_ids.add(q["relevant_doc_id"])
                    if "irrelevant_doc_id" in q:
                        eval_doc_ids.add(q["irrelevant_doc_id"])
        print(f"[OK] Found {len(eval_doc_ids):,} evaluation target documents from {args.queries_file}.")

    # Order docs: test docs first, then remaining corpus docs
    ordered_doc_ids = list(eval_doc_ids) + [d_id for d_id in corpus_map.keys() if d_id not in eval_doc_ids]

    # Filter for docs not yet in cache
    pending_doc_ids = [d_id for d_id in ordered_doc_ids if d_id not in cache]
    print(f"Total documents pending embedding: {len(pending_doc_ids):,}")

    if args.limit > 0 and len(pending_doc_ids) > args.limit:
        print(f"Applying limit: processing next {args.limit:,} documents in this run.")
        pending_doc_ids = pending_doc_ids[:args.limit]

    if not pending_doc_ids:
        print("[OK] All requested documents are already embedded and cached!")
        sync_cache_to_pgvector(conn, corpus_map, cache)
        conn.close()
        return

    # 5. Initialize Gemini Client
    client = get_gemini_client()
    print(f"[OK] Initialized Google GenAI Client (Model: {args.model}, Dim: {args.dim}).")

    # 6. Process in Batches
    batch_size = args.batch_size
    num_batches = (len(pending_doc_ids) + batch_size - 1) // batch_size
    print(f"\nStarting embedding pipeline: {len(pending_doc_ids):,} docs across {num_batches} batches...")

    for b_idx in tqdm(range(num_batches), desc="Embedding batches"):
        chunk_ids = pending_doc_ids[b_idx * batch_size: (b_idx + 1) * batch_size]
        chunk_docs = [corpus_map[d_id] for d_id in chunk_ids if d_id in corpus_map]

        try:
            embeddings = embed_batch_with_retry(
                client,
                chunk_docs,
                model=args.model,
                dim=args.dim
            )

            # Record and append to cache
            new_records = []
            for doc, emb in zip(chunk_docs, embeddings):
                cache[doc["doc_id"]] = emb
                new_records.append({"doc_id": doc["doc_id"], "embedding": emb})

            append_to_cache(args.cache_file, new_records)

            # Rate limiting delay
            if b_idx < num_batches - 1:
                time.sleep(args.delay)

        except Exception as e:
            print(f"\n[Error] Stopping pipeline due to error on batch {b_idx + 1}: {e}")
            break

    # 7. Sync updated cache to PGVector
    sync_cache_to_pgvector(conn, corpus_map, cache)
    conn.close()
    print("\n[OK] Phase 2 corpus embedding & vector ingestion complete!")


if __name__ == "__main__":
    main()
