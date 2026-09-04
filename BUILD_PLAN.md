# BUILD_PLAN.md — PatentRank: Hybrid Semantic Search & Ranking System

**Goal:** A portfolio project that proves — with real metrics — that you can fine-tune a ranking model, build hybrid search, evaluate it rigorously, and ship it as a service. This directly mirrors what the Rapid Alpha (EVOS) JD asks candidates to submit.

**Hardware constraint:** No NVIDIA GPU on your current laptop — local GPU training/inference is off the table entirely. This plan uses **Google AI Studio (Gemini API)** for embeddings and generation-assisted steps, and **Google Colab (free T4)** for the one step that genuinely requires fine-tuning a model (Phase 3's cross-encoder). Everything else runs CPU-side locally or via API calls — no local GPU dependency anywhere.

**Workflow:** Follow your established pattern — work through phases sequentially with Claude Code, run `/clear` between phases, keep `TRACK.md` updated after each phase so context doesn't bloat.

**Important tension to know about upfront:** The JD explicitly states *"using pretrained models or hosted APIs alone is not enough."* Google AI Studio's embedding/generation endpoints are hosted APIs — great for Phase 2 (semantic search baseline) and Phase 4 (segmentation assist), but they can't substitute for Phase 3 (fine-tuning your own cross-encoder), because you don't get to touch model weights through an API. Phase 3 still needs actual local training — but "local" now means **Google Colab's free T4 GPU**, not your laptop. This keeps your laptop GPU-free while still producing the one artifact recruiters actually care about: a model *you* trained, with weights you can point to.

---

## Phase 0 — Setup & Data Acquisition

**Objective:** Get a real, credible patent corpus + query set to work with.

- [ ] Set up Python env: `venv`, `google-generativeai` (or the newer `google-genai` SDK) for Google AI Studio access, `transformers` + `datasets` (CPU-only, no CUDA build needed), `psycopg2`/`pgvector` client
- [ ] Get a Google AI Studio API key (free tier) at aistudio.google.com, store it as an env var — never commit it
- [ ] Confirm API access works with a trivial embedding call before building anything on top of it
- [ ] Pull a patent dataset — options, pick one:
  - Google Patents Public Data (BigQuery, free tier) — patent abstracts + claims
  - USPTO PatentsView API (free, no auth needed for basic queries)
  - HuggingFace `datasets` — search for existing patent claim datasets (e.g. patent classification sets) to save scraping time
- [ ] Target corpus size: 5,000–20,000 patent documents (abstract + claims text). Enough to be credible, small enough to embed on your hardware in reasonable time.
- [ ] Build a small labeled query set: 100–300 (query, relevant patent, irrelevant patent) triples. This can be semi-synthetic — e.g., use patent titles/claims as pseudo-queries against their own abstracts, then manually verify 100–150 of them for quality. This *is* your "few hundred human-labeled examples" story for interviews.

**Deliverable:** `data/patents_raw.jsonl`, `data/queries_labeled.jsonl`, a `data/README.md` documenting exactly how the corpus and labels were built (this documentation matters as much as the data itself — it's what you'll describe in the application write-up).

---

## Phase 1 — Baseline: BM25 Keyword Search

**Objective:** Establish a baseline before touching any ML — this is what "hybrid" search improves on, and reviewers want to see you understand why hybrid beats pure keyword.

- [ ] Stand up Elasticsearch or OpenSearch locally via Docker (`docker-compose.yml`)
- [ ] Index the patent corpus with standard BM25 analyzer
- [ ] Run your labeled queries against BM25-only search
- [ ] Compute baseline metrics: precision@5, precision@10, NDCG@10, MRR
- [ ] Document these numbers — they're your "before" comparison point

**Deliverable:** `baseline_bm25_eval.py` + a results table in `RESULTS.md`.

---

## Phase 2 — Semantic Search: Embeddings + Vector DB

**Objective:** Add dense retrieval, combine with BM25 for hybrid search.

- [ ] Use **Google AI Studio's embedding model** (`text-embedding-004` or current equivalent) via API instead of running anything locally — no GPU needed, just API calls with rate-limit-aware batching
- [ ] Embed the full corpus, store vectors in **PGVector** (you already know Postgres — this is the fastest path to "hands-on production" credibility)
- [ ] Cache embeddings to disk as you go (`embeddings_cache.jsonl`) so re-runs don't burn API quota re-embedding the same docs
- [ ] Implement hybrid retrieval: BM25 top-N + vector top-N, merged (start with simple score fusion — e.g. reciprocal rank fusion — before anything fancier)
- [ ] Re-run eval against labeled queries, compare to BM25-only baseline

**Deliverable:** `embed_corpus.py`, `hybrid_search.py`, updated `RESULTS.md` showing pretrained-embedding hybrid vs. BM25-only.

---

## Phase 3 — The Core Deliverable: Fine-Tune a Cross-Encoder Ranker

**Objective:** This is the single most important phase — it's what separates "used an API" from "personally trained and shipped," which the JD explicitly screens for.

- [ ] This entire phase runs in **Google Colab** (free T4 GPU) — your laptop only writes/reviews the notebook, doesn't run training. Push training data to Colab via Google Drive or a repo pull.
- [ ] Pick a small cross-encoder base (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` as starting checkpoint, or fine-tune from `distilbert-base`) — Colab's T4 (16GB) comfortably fits this, no batch-size gymnastics needed
- [ ] Build training triples from your labeled query set: (query, relevant doc, hard negative)
- [ ] **Mine hard negatives properly** — don't use random negatives. Use BM25/embedding top-K misses (docs that score high but aren't actually relevant) as hard negatives. Explain this choice in your documentation — it's a specific thing the JD calls out.
- [ ] Fine-tune the cross-encoder to re-rank hybrid search's top-N results (e.g. re-rank top 50 → final top 10)
- [ ] Download the trained checkpoint from Colab, bring it back for local CPU inference (cross-encoder inference on a few thousand candidates is fine on CPU — it's only training that needed the GPU)
- [ ] Re-run full eval: BM25-only → hybrid (AI Studio embeddings) → hybrid + fine-tuned re-ranker. Three-way comparison table.

**Deliverable:** `train_cross_encoder.py`, saved model checkpoint, final `RESULTS.md` with the three-way metrics comparison (this table is the centerpiece of your portfolio).

---

## Phase 4 — Text Segmentation (Bonus, if time allows)

**Objective:** Covers the JD's "teach the system to segment text" requirement.

- [ ] Take long patent documents, split into claim-level or paragraph-level segments using a simple approach first (rule-based/regex on claim numbering)
- [ ] Optionally: fine-tune a small token-classification model (spaCy or a lightweight transformer) on a few hundred manually labeled segment boundaries
- [ ] Document the approach even if you don't fully train — a clear write-up of *how you'd approach it* is nearly as valuable as a finished model for this piece

**Deliverable:** `segment_documents.py`, a short `SEGMENTATION.md` explaining the approach and trade-offs.

---

## Phase 5 — Service Wrapper + Deployment

**Objective:** Prove you can ship this as infrastructure, not just a notebook.

- [ ] Wrap retrieval + re-ranking in a **FastAPI** service (`/search?q=...` endpoint)
- [ ] Containerize with **Docker**
- [ ] Add a minimal **Celery + Redis** async task for corpus ingestion/embedding (doesn't need to be elaborate — just prove you understand the pattern: submit job → worker processes → result stored)
- [ ] Deploy one piece to **GCP**: easiest realistic option is a Cloud Run endpoint serving the FastAPI app, with the model checkpoint pulled from GCS. Skip Vertex AI Pipelines/full MLOps — that's overkill for a portfolio piece; a working Cloud Run + GCS setup is enough signal that you've touched real GCP tools.
- [ ] Add basic Cloud Logging so you can say "monitoring" honestly

**Deliverable:** `Dockerfile`, `docker-compose.yml`, `main.py` (FastAPI), deployed Cloud Run URL, `DEPLOYMENT.md`.

---

## Phase 6 — Documentation & Packaging (Do Not Skip)

**Objective:** This is what recruiters actually read. A brilliant model with a bad README gets ignored.

- [ ] Write the main `README.md`: problem statement, architecture diagram (even a simple one), the three-way results table front and center, how to reproduce
- [ ] Write a standalone `EVALUATION.md` explaining precision@K, NDCG, MRR in your own words and why you chose the hard-negative mining approach — this doubles as prep for their 500-word application write-up
- [ ] Push to GitHub with clean commit history (not one giant commit)
- [ ] Optional: short blog post or LinkedIn write-up walking through the project

**Deliverable:** Polished public GitHub repo, ready to link in applications.

---

## Suggested Sequencing Against Your Current Load

Given your MERN internship push is still active, treat Phases 0–3 as the non-negotiable core (this is what makes the project *count*) and Phases 4–6 as valuable but flexible. Rough pacing at a few focused hours/week: Phase 0–1 in week 1–2, Phase 2 in week 3, Phase 3 (the big one) across weeks 4–6, Phases 4–6 as time allows after.

---

## Claude Code Handoff Prompt

Use this to kick off Phase 0 in a fresh Claude Code session:

```
I'm building "PatentRank" — a hybrid semantic search and ranking system
over patent documents, as a portfolio project for ML/NLP engineering
roles. Full spec is in BUILD_PLAN.md in this repo.

Hardware: no NVIDIA GPU on my laptop. Using Google AI Studio API for
embeddings, and Google Colab's free T4 GPU for the one fine-tuning
step (Phase 3) — everything else runs CPU-side or via API.

Start with Phase 0 only: set up the Python environment, verify CUDA,
and help me pull a patent dataset (~5,000-20,000 docs) plus build a
labeled query set of 100-300 (query, relevant doc, irrelevant doc)
triples. Ask me before making a final call on data source. Stop after
Phase 0 is complete and data files exist — do not start Phase 1 yet.
```

Repeat the pattern for each subsequent phase, referencing `TRACK.md` for what's already done.
