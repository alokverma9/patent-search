# PatentRank — Execution Tracker (`TRACK.md`)

This file tracks the implementation progress across all project phases as defined in `BUILD_PLAN.md`.

---

## 1. System & Architecture Constraints

- **Local Machine:** Laptop (Windows, CPU-only — NVIDIA GPU not present).
- **Embeddings & LLM API:** Google AI Studio (Gemini API) using `gemini-embedding-2` and `gemini-3.5-flash-lite`.
- **Model Fine-Tuning:** Google Colab (Free T4 GPU) reserved strictly for Phase 3 (Cross-Encoder Ranker).
- **Environment:** Python 3.14.7 virtual environment (`.venv`) with CPU-optimized PyTorch 2.14.0.

---

## 2. Phase Status Matrix

| Phase | Description | Deliverables | Status |
|---|---|---|:---:|
| **Phase 0** | Setup & Data Acquisition | `.venv`, `verify_setup.py`, `data/patents_raw.jsonl`, `data/queries_labeled.jsonl`, `data/README.md` | **COMPLETED** |
| **Phase 1** | Baseline: BM25 Keyword Search | `docker-compose.yml`, `baseline_bm25_eval.py`, `RESULTS.md` | PENDING |
| **Phase 2** | Semantic Search: Embeddings + Vector DB | `embed_corpus.py`, `hybrid_search.py`, PGVector integration, `RESULTS.md` | PENDING |
| **Phase 3** | Core: Fine-Tune Cross-Encoder Ranker | Colab notebook/script, hard-negative mining, model checkpoint, `train_cross_encoder.py`, `RESULTS.md` | PENDING |
| **Phase 4** | Text Segmentation | `segment_documents.py`, `SEGMENTATION.md` | PENDING |
| **Phase 5** | Service Wrapper + GCP Deployment | FastAPI service, Docker, Celery/Redis, GCP Cloud Run | PENDING |
| **Phase 6** | Documentation & Packaging | `README.md`, `EVALUATION.md`, GitHub repository polish | PENDING |

---

## 3. Phase 0 Detailed Completion Log

- [x] **Virtual Environment Setup:**
  - Initialized isolated virtual environment `.venv`.
  - Installed all required libraries (`torch` CPU, `transformers`, `datasets`, `google-genai`, `psycopg2-binary`, `pgvector`, `scikit-learn`, `pandas`, `numpy`, `python-dotenv`, `tqdm`).
  - Generated `requirements.txt`, `.gitignore`, and `.env.example`.
- [x] **Hardware & CUDA Verification:**
  - Ran `scripts/verify_setup.py`:
    - PyTorch version: `2.14.0+cpu`
    - `torch.cuda.is_available()`: `False` (0 GPUs — CPU-only local execution verified).
- [x] **Google AI Studio (Gemini API) Verification:**
  - Verified API key connectivity.
  - Tested `gemini-embedding-2` embedding endpoint (successfully produced 3072-dimensional dense embeddings).
  - Tested `gemini-3.5-flash-lite` text generation endpoint for prompt/query formulation.
- [x] **Data Source Selection:**
  - Evaluated candidate sources (BigPatent, PatentsView API, Google Patents BigQuery, USPTO Claims).
  - Consulted user and selected **BigPatent (USPTO, CPC Category G: Physics, Computing & Information Technology)**.
- [x] **Corpus Acquisition & Structuring (`data/patents_raw.jsonl`):**
  - Downloaded and processed partition via direct Parquet stream to avoid HuggingFace Hub directory rate-limits.
  - Extracted structured fields: `doc_id`, `category`, `title`, `abstract`, `summary`, and unified `search_text`.
  - Size: **10,000 patent documents** (~53.01 MB), averaging **423.1 words/doc**.
- [x] **Labeled Evaluation Query Set (`data/queries_labeled.jsonl`):**
  - Generated **200 realistic examiner/engineer search queries** spanning diverse technological domains.
  - Paired each query with ground-truth positive (`relevant_doc_id`) and orthogonal negative (`irrelevant_doc_id`).
  - Annotated with query types (`technical_mechanism`, `system_architecture`, `method_and_process`, `device_and_apparatus`).
- [x] **Documentation:**
  - Published comprehensive `data/README.md` detailing dataset provenance, data schemas, generation methodology, and reproduction steps.

---

## 4. Next Step: Phase 1 (BM25 Keyword Search Baseline)

Phase 0 is complete. Awaiting user sign-off before commencing Phase 1.
