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
from src.utils import load_config, save_checkpoint, find_latest_checkpoint, load_checkpoint, compute_tokenizer_fingerprint


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

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"SentencePieceモデルが見つかりません: {tokenizer_path}\n"
            f"先に以下を実行してトークナイザーを学習してください:\n"
            f"  python src/train_tokenizer.py --corpus data/corpus.txt --output_dir data/ --vocab_size 16000\n"
            f"（Kaggle上で corpus.txt が /kaggle/input/... 側にある場合はそのパスを --corpus に指定）\n"
            f"学習後、preprocess.py も --tokenizer_path 付きで再実行してtoken_idsを実データに合わせること:\n"
            f"  python -m src.preprocess --train_path data/v001.train.jsonl --val_path data/v001.val.jsonl "
            f"--output_dir data/ --tokenizer_path data/spm_16k.model\n"
            f"トークナイザーがまだ無い場合は、configs/*.yaml の data.tokenizer_path を null にすれば"
            f"生成ベース評価（EM/F1）をスキップしてloss/PPLのみで学習を進めることもできます。"
        )

    import sentencepiece as spm

    from src.code_text import CodecTokenizer

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    codec = CodecTokenizer(sp)

    class _TokenizerWrapper:
        def decode(self, ids: list[int]) -> str:
            return codec.decode([i for i in ids if i not in (pad_id, bos_id, eos_id)])

    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
    bos_id = sp.bos_id() if sp.bos_id() >= 0 else 1
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 2

    return _TokenizerWrapper(), {"pad_id": pad_id, "bos_id": bos_id, "eos_id": eos_id}


def train(config_path: str):
    config = load_config(config_path)
    exp_name = config.get("experiment_name", os.path.splitext(os.path.basename(config_path))[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, tok_meta = load_tokenizer(config["data"].get("tokenizer_path"))
    tokenizer_fp = compute_tokenizer_fingerprint(config["data"].get("tokenizer_path"))

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
            try:
                global_step = load_checkpoint(
                    latest, model, optimizer, scheduler, map_location=device,
                    expected_tokenizer_fingerprint=tokenizer_fp,
                )
                log(f"Resumed from {latest} at step {global_step}")
            except RuntimeError as e:
                # トークナイザー不一致を検出した場合は、汚染された状態で学習を
                # 続けるより安全な「新規学習」にフォールバックする。
                # ckpt_dir を退避してから0から学習し直す（古いcheckpointは残す）。
                log(f"⚠️ {e}")
                backup_dir = f"{ckpt_dir}_incompatible_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                os.rename(ckpt_dir, backup_dir)
                os.makedirs(ckpt_dir, exist_ok=True)
                log(f"⚠️ 互換性の無いcheckpointを {backup_dir} に退避し、0から学習を開始します。")
                global_step = 0

    # --- 事前学習済みの重みで初期化（再開用checkpointが無いときだけ） ---
    init_from = train_cfg.get("init_from")
    if init_from and global_step == 0:
        ck = torch.load(init_from, map_location=device)
        stored = ck.get("tokenizer_fingerprint")
        if stored and tokenizer_fp and stored != tokenizer_fp:
            raise RuntimeError(f"init_from '{init_from}' は別のトークナイザーで学習されています "
                               f"({stored} ≠ {tokenizer_fp})。事前学習と同じトークナイザーを使ってください。")
        model.load_state_dict(ck["model_state_dict"])
        log(f"事前学習済みの重みで初期化: {init_from}（事前学習 {ck.get('tokens_seen', '?')} トークン）")

    # --- 混合精度（training.amp: auto/fp16/bf16/off。既定は off = 従来どおり） ---
    amp_mode = train_cfg.get("amp", "off")
    amp_dtype, use_scaler = None, False
    if device.type == "cuda" and amp_mode != "off":
        if amp_mode == "bf16" or (amp_mode == "auto" and torch.cuda.is_bf16_supported()):
            amp_dtype = torch.bfloat16
        else:
            amp_dtype, use_scaler = torch.float16, True
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    if amp_dtype:
        log(f"混合精度: {amp_dtype}")

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
                save_checkpoint(global_step, model, optimizer, scheduler, ckpt_dir, tokenizer_fingerprint=tokenizer_fp)
                stop_training = True
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(input_ids)
            loss = compute_loss(outputs.logits.float(), labels, label_smoothing=train_cfg.get("label_smoothing", 0.0))
            ((loss / grad_accum_steps) if scaler is None else scaler.scale(loss / grad_accum_steps)).backward()
            train_loss_accum += loss.item()

            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                        save_checkpoint(global_step, model, optimizer, scheduler, ckpt_dir, is_best=True, tokenizer_fingerprint=tokenizer_fp)
                        log(f"✓ Best checkpoint 保存 (val_loss={val_loss:.4f})")
                    else:
                        no_improve_count += 1
                        if no_improve_count >= train_cfg.get("early_stopping_patience", 5):
                            log(f"Early stopping (step {global_step})")
                            stop_training = True
                            break

                if global_step % train_cfg.get("checkpoint_frequency", 1000) == 0:
                    save_checkpoint(global_step, model, optimizer, scheduler, ckpt_dir, tokenizer_fingerprint=tokenizer_fp)

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
