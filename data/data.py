"""
data.py
前処理済み jsonl（token_ids 付与済み）を読み込み、学習用/評価用のバッチを作る。

想定される1行の構造（plan §2.2, §3.1 段階9 に準拠）:
{
  "id": "...",
  "input": "...",
  "cot": "...",
  "answer": "...",
  "token_ids": [ ... ],          # SentencePieceでエンコード済みの input のトークン列
  "cot_token_ids": [ ... ],      # 同上、cot のトークン列（空リスト可）
  "meta": {"category": "arithmetic", ...}
}

学習系列は [BOS] input_ids cot_token_ids answer_ids [EOS] を連結して構成する。
評価（生成ベース）では prompt_ids（= input_ids のみ）と answer_text を別途保持する。
"""
from __future__ import annotations

import json
from typing import Optional

import torch
from torch.utils.data import Dataset


class JsonlLMDataset(Dataset):
    def __init__(
        self,
        path: str,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        max_seq_length: int = 512,
        answer_token_ids_key: str = "answer_token_ids",
    ):
        self.examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_seq_length = max_seq_length
        self.answer_token_ids_key = answer_token_ids_key

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        item = self.examples[idx]

        prompt_ids = item.get("token_ids", [])
        cot_ids = item.get("cot_token_ids", []) or []
        answer_ids = item.get(self.answer_token_ids_key, []) or []

        full_ids = [self.bos_id] + prompt_ids + cot_ids + answer_ids + [self.eos_id]
        full_ids = full_ids[: self.max_seq_length]

        return {
            "full_ids": full_ids,
            "prompt_ids": [self.bos_id] + prompt_ids[: self.max_seq_length - 1],
            "answer_text": item.get("answer", ""),
            "category": item.get("meta", {}).get("category", "unknown"),
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        max_len = max(len(b["full_ids"]) for b in batch)
        max_prompt_len = max(len(b["prompt_ids"]) for b in batch)

        input_ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  # -100 は損失計算で無視
        prompt_ids = torch.full((len(batch), max_prompt_len), self.pad_id, dtype=torch.long)

        for i, b in enumerate(batch):
            ids = b["full_ids"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            # 次トークン予測なので labels は1つシフトしたもの。ここでは同じ配列を使い、
            # 損失計算側（train.py の criterion 呼び出し）で shift する設計にしている。
            labels[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

            p_ids = b["prompt_ids"]
            prompt_ids[i, : len(p_ids)] = torch.tensor(p_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_ids": prompt_ids,
            "answer_text": [b["answer_text"] for b in batch],
            "category": [b["category"] for b in batch],
        }


def build_dataloaders(config: dict, tokenizer_meta: dict):
    """config['data'] と tokenizer_meta（pad_id/bos_id/eos_id）からtrain/valのDataLoaderを作る。"""
    from torch.utils.data import DataLoader

    train_ds = JsonlLMDataset(
        path=config["data"]["train_path"],
        pad_id=tokenizer_meta["pad_id"],
        bos_id=tokenizer_meta["bos_id"],
        eos_id=tokenizer_meta["eos_id"],
        max_seq_length=config["model"]["max_seq_length"],
    )
    val_ds = JsonlLMDataset(
        path=config["data"]["val_path"],
        pad_id=tokenizer_meta["pad_id"],
        bos_id=tokenizer_meta["bos_id"],
        eos_id=tokenizer_meta["eos_id"],
        max_seq_length=config["model"]["max_seq_length"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"].get("batch_size", 16),
        shuffle=True,
        collate_fn=train_ds.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"].get("eval_batch_size", 16),
        shuffle=False,
        collate_fn=val_ds.collate_fn,
    )
    return train_loader, val_loader
