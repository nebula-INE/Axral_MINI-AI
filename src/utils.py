"""
utils.py
config読込、チェックポイントの保存・復帰・検索、Kaggleセッション時間管理のヘルパー。
"""
from __future__ import annotations

import glob
import hashlib
import os
from datetime import datetime

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_tokenizer_fingerprint(tokenizer_path: str | None) -> str | None:
    """トークナイザーファイル（.model）の中身のハッシュを取る。

    checkpointがどのトークナイザーで学習されたかを記録するために使う。
    vocab_sizeが同じでも、character_coverageの変更等でトークンIDの意味が
    変わることがあり、その場合 load_state_dict はエラーにならず
    「見た目は成功するが中身は意味不明」という壊れ方をする
    （実データで、resume_from_checkpoint=true のまま character_coverage を
    変えてトークナイザーを再学習し、古いcheckpointから再開してしまい、
    テンプレート同士が溶け合ったような出力になるトラブルが発生した）。
    ハッシュ不一致を検出できれば、このクラスの事故を機械的に防げる。
    """
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        return None
    with open(tokenizer_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def save_checkpoint(
    step: int,
    model,
    optimizer,
    scheduler,
    ckpt_dir: str,
    is_best: bool = False,
    tokenizer_fingerprint: str | None = None,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "timestamp": datetime.now().isoformat(),
        "tokenizer_fingerprint": tokenizer_fingerprint,
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


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    expected_tokenizer_fingerprint: str | None = None,
    strict_tokenizer_check: bool = True,
) -> int:
    """checkpointを読み込む。expected_tokenizer_fingerprintを渡した場合、
    checkpoint内に記録された指紋と比較し、不一致なら例外を投げて
    （壊れた状態で）学習が進んでしまうのを未然に防ぐ。

    古いcheckpoint（このフィールドが無い、None）は
    strict_tokenizer_check=True でも許容する（過去の互換性のため）。
    明確に「別物と分かっている」指紋同士が食い違う場合のみ拒否する。
    """
    checkpoint = torch.load(path, map_location=map_location)

    stored_fp = checkpoint.get("tokenizer_fingerprint")
    if (
        strict_tokenizer_check
        and expected_tokenizer_fingerprint is not None
        and stored_fp is not None
        and stored_fp != expected_tokenizer_fingerprint
    ):
        raise RuntimeError(
            f"checkpoint '{path}' は現在のトークナイザーと異なるトークナイザーで学習されています "
            f"(checkpoint側: {stored_fp}, 現在: {expected_tokenizer_fingerprint})。\n"
            f"このまま再開すると、トークンIDの意味がズレたまま学習が汚染されます。\n"
            f"対処: config の training.resume_from_checkpoint を false にするか、"
            f"古いcheckpointが入ったディレクトリを削除/移動してから再実行してください。"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["step"]
