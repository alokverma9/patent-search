"""
sync_gcs_model.py — Google Cloud Storage Model Checkpoint Sync Utility

Automates synchronizing the fine-tuned Cross-Encoder ranker checkpoint
(models/patentrank-cross-encoder/) between local storage and Google Cloud Storage (GCS).
Used in GCP Cloud Run startup hooks or CI/CD pipelines to decouple heavy model weights
from the container base image.
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("patentrank.gcs_sync")

DEFAULT_BUCKET = os.getenv("GCS_MODEL_BUCKET", "patentrank-models")
DEFAULT_REMOTE_PREFIX = "models/patentrank-cross-encoder"
DEFAULT_LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "patentrank-cross-encoder")

REQUIRED_MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def upload_to_gcs(bucket_name: str, local_dir: str, remote_prefix: str, dry_run: bool = False):
    """Upload local model directory to GCS bucket."""
    logger.info(f"Uploading model artifacts from {local_dir} to gs://{bucket_name}/{remote_prefix}...")

    if not os.path.exists(local_dir):
        logger.error(f"Local model directory does not exist: {local_dir}")
        sys.exit(1)

    files_to_upload = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    logger.info(f"Found {len(files_to_upload)} files to upload: {files_to_upload}")

    if dry_run:
        logger.info("[DRY-RUN] Simulating upload. Skipping actual network calls.")
        for f in files_to_upload:
            logger.info(f"  [DRY-RUN] -> gs://{bucket_name}/{remote_prefix}/{f}")
        return

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for filename in files_to_upload:
            local_path = os.path.join(local_dir, filename)
            remote_path = f"{remote_prefix}/{filename}".replace("//", "/")
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            logger.info(f"  [UPLOADED] {filename} -> gs://{bucket_name}/{remote_path} ({os.path.getsize(local_path)/1024/1024:.2f} MB)")

        logger.info(f"[SUCCESS] All model checkpoint files synced to gs://{bucket_name}/{remote_prefix}.")
    except ImportError:
        logger.warning("google-cloud-storage package not installed. Run 'pip install google-cloud-storage'.")
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")
        sys.exit(1)


def download_from_gcs(bucket_name: str, remote_prefix: str, local_dir: str, dry_run: bool = False):
    """Download model checkpoint files from GCS bucket to local directory."""
    logger.info(f"Downloading model artifacts from gs://{bucket_name}/{remote_prefix} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)

    if dry_run:
        logger.info("[DRY-RUN] Simulating download.")
        for f in REQUIRED_MODEL_FILES:
            logger.info(f"  [DRY-RUN] <- gs://{bucket_name}/{remote_prefix}/{f}")
        return

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(client.list_blobs(bucket, prefix=remote_prefix))

        if not blobs:
            logger.error(f"No blobs found in gs://{bucket_name}/{remote_prefix}")
            sys.exit(1)

        for blob in blobs:
            rel_name = os.path.basename(blob.name)
            if not rel_name:
                continue
            dest_file = os.path.join(local_dir, rel_name)
            blob.download_to_filename(dest_file)
            logger.info(f"  [DOWNLOADED] {rel_name} ({os.path.getsize(dest_file)/1024/1024:.2f} MB)")

        logger.info(f"[SUCCESS] Checkpoint downloaded successfully to {local_dir}.")
    except ImportError:
        logger.warning("google-cloud-storage not installed. Using local checkpoint if present.")
    except Exception as e:
        logger.error(f"GCS download failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Sync PatentRank Cross-Encoder weights with Google Cloud Storage")
    parser.add_argument("--action", choices=["upload", "download"], default="download", help="Sync direction")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket name")
    parser.add_argument("--remote-prefix", default=DEFAULT_REMOTE_PREFIX, help="GCS directory prefix")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help="Local directory path")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without network calls")
    args = parser.parse_args()

    if args.action == "upload":
        upload_to_gcs(args.bucket, args.local_dir, args.remote_prefix, dry_run=args.dry_run)
    else:
        download_from_gcs(args.bucket, args.remote_prefix, args.local_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
