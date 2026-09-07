"""
celery_app.py — Celery Application Configuration for PatentRank

Manages distributed background task processing for corpus ingestion,
embedding generation, and text segmentation indexing via Redis.
"""

import os
from celery import Celery

# Redis connection defaults
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery(
    "patentrank",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]
)

# Production task settings
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,         # 10 minutes max per task
    task_soft_time_limit=540,    # 9 minutes soft limit
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=86400,        # 24 hours result persistence
)

if __name__ == "__main__":
    celery_app.start()
