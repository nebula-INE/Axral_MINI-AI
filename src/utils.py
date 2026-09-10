"""
utils.py
config読込、チェックポイントの保存・復帰・検索、Kaggleセッション時間管理のヘルパー。
"""
from __future__ import annotations

import glob
import os
from datetime import datetime

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    step: int,
    model,
    optimizer,
    scheduler,
    ckpt_dir: str,
    is_best: bool = False,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "timestamp": datetime.now().isoformat(),
    }

    path = os.path.join(ckpt_dir, f"checkpoint_{step:06d}.pt")
    torch.save(payload, path)

    if is_best:
        best_path = os.path.join(ckpt_dir, "checkpoint_best.pt")
        torch.save(payload, best_path)

    return path


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_[0-9]*.pt")))
    return candidates[-1] if candidates else None


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu") -> int:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["step"]
