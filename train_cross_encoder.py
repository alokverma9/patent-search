"""
train_cross_encoder.py — PatentRank Phase 3: Fine-Tune Cross-Encoder Ranker

This script:
1. Loads the hard-negative mined training & validation datasets (data/train_pairs.jsonl & data/val_pairs.jsonl).
2. Initializes a pretrained cross-encoder (default: cross-encoder/ms-marco-MiniLM-L-6-v2).
3. Fine-tunes the transformer cross-attention layers using AdamW optimizer with cosine learning rate schedule.
4. Optimizes binary cross-entropy loss over positive vs. hard-negative / orthogonal pairs.
5. Evaluates validation loss, MRR, and Hit@1 at each epoch.
6. Saves the best model checkpoint (weights, config, tokenizer) to models/patentrank-cross-encoder.
7. Logs training trajectory to models/patentrank-cross-encoder/training_history.json.

Compatible with:
- Local CPU execution (functional verification & evaluation)
- Google Colab free T4 GPU (accelerated high-throughput fine-tuning)
"""

import os
import sys
import time
import json
import math
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm


class PatentRankingPairDataset(Dataset):
    """PyTorch Dataset for Query-Document ranking pairs with binary relevance labels."""

    def __init__(self, jsonl_file: str):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        return {
            "query_id": item["query_id"],
            "query": item["query"],
            "doc_id": item["doc_id"],
            "doc_text": item["doc_text"],
            "label": float(item["label"]),
            "type": item.get("type", "unknown"),
            "query_type": item.get("query_type", "general")
        }


def make_collate_fn(tokenizer, max_length: int = 256):
    """Custom collator that batches and dynamically pads query-document text pairs."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [item["query"] for item in batch]
        docs = [item["doc_text"] for item in batch]
        labels = [item["label"] for item in batch]
        query_ids = [item["query_id"] for item in batch]
        doc_ids = [item["doc_id"] for item in batch]

        encoded = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoded["labels"] = torch.tensor(labels, dtype=torch.float)
        encoded["query_ids"] = query_ids
        encoded["doc_ids"] = doc_ids
        return encoded

    return collate_fn


def evaluate_ranking(model, dataloader, device, loss_fn) -> Tuple[float, float, float]:
    """
    Evaluate validation loss and ranking metrics (MRR and Hit@1) across validation queries.
    Each validation query has 1 positive and multiple hard/orthogonal negatives.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Collect scores grouped by query_id: {q_id: [(score, label), ...]}
    query_scores: Dict[str, List[Tuple[float, float]]] = {}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            query_ids = batch["query_ids"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            # Transfer scores to CPU
            scores = logits.cpu().tolist()
            lbls = labels.cpu().tolist()

            for q_id, sc, lb in zip(query_ids, scores, lbls):
                query_scores.setdefault(q_id, []).append((sc, lb))

    avg_loss = total_loss / max(1, num_batches)

    # Compute Ranking MRR and Hit@1
    mrrs = []
    hit1s = []

    for q_id, candidates in query_scores.items():
        # Sort candidates by predicted score descending
        sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        # Find 1-based rank of the ground truth positive (label == 1.0)
        pos_rank = None
        for r, (sc, lb) in enumerate(sorted_candidates, start=1):
            if lb == 1.0:
                pos_rank = r
                break

        if pos_rank is not None:
            mrrs.append(1.0 / pos_rank)
            hit1s.append(1.0 if pos_rank == 1 else 0.0)
        else:
            mrrs.append(0.0)
            hit1s.append(0.0)

    val_mrr = sum(mrrs) / max(1, len(mrrs))
    val_hit1 = sum(hit1s) / max(1, len(hit1s))

    return avg_loss, val_mrr, val_hit1


def main():
    parser = argparse.ArgumentParser(description="Train PatentRank Cross-Encoder Ranker")
    parser.add_argument("--train-file", default="data/train_pairs.jsonl", help="Path to training pairs JSONL")
    parser.add_argument("--val-file", default="data/val_pairs.jsonl", help="Path to validation pairs JSONL")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Pretrained base model")
    parser.add_argument("--output-dir", default="models/patentrank-cross-encoder", help="Output directory for checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup fraction for LR scheduler")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cuda', or 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional limit on training samples for test run")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("PATENTRANK PHASE 3: CROSS-ENCODER RANKER FINE-TUNING")
    print("=" * 70)
    print(f"Target Device:      {device}")
    if device.type == "cuda":
        print(f"GPU Device Name:    {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory (VRAM):  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Base Model:         {args.base_model}")
    print(f"Training Data:      {args.train_file}")
    print(f"Validation Data:    {args.val_file}")
    print(f"Output Directory:   {args.output_dir}")
    print(f"Hyperparameters:    epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, max_len={args.max_length}")
    print("=" * 70)

    # 1. Load Tokenizer & Model
    print(f"\nInitializing tokenizer and base cross-encoder ({args.base_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=1)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model loaded. Total parameters: {total_params:,} (Trainable: {trainable_params:,})")

    # 2. Datasets & Loaders
    print("\nPreparing Datasets and DataLoaders...")
    train_dataset = PatentRankingPairDataset(args.train_file)
    val_dataset = PatentRankingPairDataset(args.val_file)

    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        train_dataset.samples = train_dataset.samples[:args.max_train_samples]
        print(f"[NOTE] Truncated training dataset to {len(train_dataset)} samples for fast run.")

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    print(f"[OK] Train set: {len(train_dataset)} pairs ({len(train_loader)} batches)")
    print(f"[OK] Val set:   {len(val_dataset)} pairs ({len(val_loader)} batches)")

    # 3. Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Positive weight handling: in our data, negative:positive is ~6:1
    # Standard BCEWithLogitsLoss works cleanly with rank-based evaluation
    loss_fn = nn.BCEWithLogitsLoss()

    # 4. Zero-Shot Baseline Evaluation (Pretrained Weights)
    print("\nEvaluating Pretrained (Zero-Shot) Baseline on Validation Set...")
    init_val_loss, init_val_mrr, init_val_hit1 = evaluate_ranking(model, val_loader, device, loss_fn)
    print(f"Pretrained Baseline — Val Loss: {init_val_loss:.4f} | Val MRR: {init_val_mrr:.4f} | Val Hit@1: {init_val_hit1 * 100:.2f}%")

    # 5. Training Loop
    best_val_mrr = init_val_mrr
    best_epoch = 0
    training_history = {
        "base_model": args.base_model,
        "device": str(device),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps
        },
        "zero_shot_baseline": {
            "val_loss": init_val_loss,
            "val_mrr": init_val_mrr,
            "val_hit@1": init_val_hit1
        },
        "epochs": []
    }

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    print("\nBeginning Fine-Tuning...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0_epoch = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)

            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        train_loss = epoch_loss / max(1, step_count)
        epoch_time = time.time() - t0_epoch

        # Validation Step
        val_loss, val_mrr, val_hit1 = evaluate_ranking(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch:2d}/{args.epochs:2d} ({epoch_time:.1f}s) — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val MRR: {val_mrr:.4f} | Val Hit@1: {val_hit1 * 100:.2f}%"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mrr": val_mrr,
            "val_hit@1": val_hit1,
            "lr": current_lr,
            "duration_sec": epoch_time
        }
        training_history["epochs"].append(epoch_record)

        # Save Best Checkpoint
        if val_mrr >= best_val_mrr:
            best_val_mrr = val_mrr
            best_epoch = epoch
            print(f"  --> New best Val MRR: {val_mrr:.4f}! Saving checkpoint to {args.output_dir}...")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    total_training_time = time.time() - t_start
    training_history["total_duration_sec"] = total_training_time
    training_history["best_epoch"] = best_epoch
    training_history["best_val_mrr"] = best_val_mrr

    # Save training history JSON
    history_file = os.path.join(args.output_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    print("\n" + "=" * 70)
    print("CROSS-ENCODER TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Duration:     {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    print(f"Best Epoch:         {best_epoch} (Val MRR: {best_val_mrr:.4f})")
    print(f"Checkpoint Saved:   {args.output_dir}")
    print(f"History File:       {history_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
