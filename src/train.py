"""
train.py
学習ループ本体。Kaggle Notebook からも `python -m src.train --config configs/exp01.yaml`
としてローカル/スクリプト実行からも呼べるようにCLI化してある。

plan §4.3 の擬似コードを実装に落としたもの。差分:
  - evaluate() は eval.py の生成ベース版を使用（tokenizer引数を渡す）
  - loss計算を shift させたcross entropyに統一（model.py 側は生の logits を返すだけ）
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.model import TransformerLM
from src.data import build_dataloaders
from src.eval import evaluate
from src.utils import load_config, save_checkpoint, find_latest_checkpoint, load_checkpoint


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def load_tokenizer(tokenizer_path: str | None):
    """SentencePieceモデルをロードする薄いラッパー。tokenizer_path が None ならNoneを返す
    （evaluate()側でgeneration評価をスキップする分岐がある）。
    """
    if not tokenizer_path:
        return None, {"pad_id": 0, "bos_id": 1, "eos_id": 2}
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    class _TokenizerWrapper:
        def decode(self, ids: list[int]) -> str:
            return sp.DecodeIds([i for i in ids if i not in (pad_id, bos_id, eos_id)])

    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 1
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2

    return _TokenizerWrapper(), {"pad_id": pad_id, "bos_id": bos_id, "eos_id": eos_id}


def train(config_path: str):
    config = load_config(config_path)
    exp_name = config.get("experiment_name", os.path.splitext(os.path.basename(config_path))[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, tok_meta = load_tokenizer(config["data"].get("tokenizer_path"))

    model_cfg = dict(config["model"])
    model_cfg["vocab_size"] = config["data"].get("vocab_size", 16000)
    model = TransformerLM(model_cfg).to(device)
    print(f"[{exp_name}] パラメータ数: {model.num_parameters():,}")

    train_loader, val_loader = build_dataloaders(config, tok_meta)

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.95)),
        eps=opt_cfg.get("eps", 1e-8),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )

    train_cfg = config["training"]
    total_steps = train_cfg["epochs"] * max(1, len(train_loader) // train_cfg.get("grad_accum_steps", 1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=opt_cfg.get("lr_warmup_steps", 0),
        num_training_steps=total_steps,
    )

    ckpt_dir = os.path.join(config.get("results_dir", "results/checkpoints"), exp_name)
    log_path = os.path.join(config.get("logs_dir", "logs"), f"{exp_name}_training.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    use_wandb = config.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb

        wandb.init(project=config["wandb"].get("project", "vose-initial-llm"), config=config)

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # --- チェックポイント復帰 ---
    global_step = 0
    if train_cfg.get("resume_from_checkpoint", False):
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            global_step = load_checkpoint(latest, model, optimizer, scheduler, map_location=device)
            log(f"Resumed from {latest} at step {global_step}")

    session_start = datetime.now()
    max_session_time = timedelta(minutes=config.get("kaggle", {}).get("max_session_time_minutes", 540))

    best_val_loss = float("inf")
    no_improve_count = 0
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    stop_training = False

    for epoch in range(train_cfg["epochs"]):
        if stop_training:
            break
        model.train()
        train_loss_accum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            elapsed = datetime.now() - session_start
            if elapsed > max_session_time * 0.9:
                log(f"⚠️ セッション時間制限に接近。step {global_step} でcheckpoint保存して終了します。")
                save_checkpoint(global_step, model, optimizer, scheduler, ckpt_dir)
                stop_training = True
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            loss = compute_loss(outputs.logits, labels, label_smoothing=train_cfg.get("label_smoothing", 0.0))
            (loss / grad_accum_steps).backward()
            train_loss_accum += loss.item()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    avg = train_loss_accum / (batch_idx + 1)
                    log(f"Epoch {epoch}, Step {global_step}, Loss: {avg:.4f}")
                    if use_wandb:
                        wandb.log({"train_loss": avg, "lr": scheduler.get_last_lr()[0], "step": global_step})

                if global_step % train_cfg.get("eval_frequency", 500) == 0:
                    val_loss, metrics = evaluate(model, val_loader, device, tokenizer=tokenizer)
                    log(
                        f"Val Loss(PPL): {val_loss:.4f} ({metrics['ppl']:.2f}), "
                        f"EM: {metrics.get('em', 0):.3f}, F1: {metrics.get('f1', 0):.3f}"
                    )
                    if use_wandb:
                        wandb.log({"val_loss": val_loss, "step": global_step, **metrics})

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        save_checkpoint(global_step, model, optimizer, scheduler, ckpt_dir, is_best=True)
                        log(f"✓ Best checkpoint 保存 (val_loss={val_loss:.4f})")
                    else:
                        no_improve_count += 1
                        if no_improve_count >= train_cfg.get("early_stopping_patience", 5):
                            log(f"Early stopping (step {global_step})")
                            stop_training = True
                            break

                if global_step % train_cfg.get("checkpoint_frequency", 1000) == 0:
                    save_checkpoint(global_step, model, optimizer, scheduler, ckpt_dir)

    log(f"✓ Training complete at step {global_step}")
    if use_wandb:
        wandb.finish()

    return {"final_step": global_step, "best_val_loss": best_val_loss}


def _main():
    parser = argparse.ArgumentParser(description="LLM学習ループ")
    parser.add_argument("--config", required=True, help="configs/exp01.yaml など")
    args = parser.parse_args()
    result = train(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
