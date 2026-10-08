"""
pretrain.py
事前学習（日本語の文章とコードを大量に読ませる段階）。pack_tokens.py が作ったトークン列（.bin）から
ランダムな位置の窓を切り出して、次のトークンを予測させる。

  python -m src.pretrain --config configs/exp_p7_pretrain.yaml

設計:
  - データの種類（wiki / code）を、configの data.mix の比率で混ぜて読む（ファイルは種類ごとに別）。
  - 混合精度（fp16/bf16）で速くする。fp16では勾配のスケーリングを使う。
  - 学習の長さは「総トークン数」(training.total_tokens) か「時間予算」(training.time_budget_hours) で決める。
    時間予算のときは、学習率の下げ方（コサイン）を経過時間で決めるので、途中でセッションが切れて
    再開しても、予算を使い切った時点で学習率がちょうど最小になる。
  - Kaggleのセッション上限（kaggle.max_session_time_minutes）に近づくと、checkpointを保存して正常終了する。
    同じコマンドをもう一度実行すれば、続きから再開する（resume: true）。
  - 検証損失は wiki / code を別々に出す（どちらが伸びているかが分かる）。評価のたびに短い生成例も表示する。
  - checkpoint は train.py の init_from がそのまま読める形式（model_state_dict を含む）。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from src.code_text import CodecTokenizer
from src.model import TransformerLM
from src.utils import compute_tokenizer_fingerprint, load_config, native_bf16_supported

SOURCES = ("wiki", "code")


# --------------------------------------------------------------------------- データ
class PackedData:
    """種類ごとのトークン列（uint16のmemmap）から、ランダムな窓を取り出す。"""

    def __init__(self, token_dir: str, split: str, block_size: int):
        self.block = block_size
        self.arrays: dict[str, np.memmap] = {}
        for src in SOURCES:
            path = os.path.join(token_dir, f"{split}_{src}.bin")
            if os.path.exists(path) and os.path.getsize(path) > 2 * (block_size + 2):
                self.arrays[src] = np.memmap(path, dtype=np.uint16, mode="r")

    def tokens(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.arrays.items()}

    def batch(self, batch_size: int, weights: dict[str, float], rng: np.random.Generator, device,
              only: str | None = None):
        srcs = [only] if only else [s for s in SOURCES if s in self.arrays and weights.get(s, 0) > 0]
        w = np.array([1.0 if only else weights[s] for s in srcs], dtype=np.float64)
        w /= w.sum()
        picks = rng.choice(len(srcs), size=batch_size, p=w)
        xs, ys = [], []
        for p in picks:
            arr = self.arrays[srcs[p]]
            i = int(rng.integers(0, len(arr) - self.block - 1))
            chunk = np.asarray(arr[i: i + self.block + 1], dtype=np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)
        y = torch.from_numpy(np.stack(ys)).to(device, non_blocking=True)
        return x, y


# --------------------------------------------------------------------------- 学習率
def lr_at(progress: float, lr: float, min_lr: float, warmup_progress: float) -> float:
    """progress は 0〜1（学習全体のどこまで進んだか）。ウォームアップ→コサインで min_lr まで下げる。"""
    if progress < warmup_progress:
        return lr * (progress / max(warmup_progress, 1e-9))
    t = min(1.0, (progress - warmup_progress) / max(1.0 - warmup_progress, 1e-9))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))


def pick_amp(mode: str, device):
    """(autocast用dtype or None, GradScalerを使うか)"""
    if device.type != "cuda" or mode == "off":
        return None, False
    if mode == "bf16" or (mode == "auto" and native_bf16_supported()):
        return torch.bfloat16, False
    return torch.float16, True


# --------------------------------------------------------------------------- checkpoint
def save_ckpt(path: str, model, optimizer, scaler, step: int, tokens_seen: int, elapsed: float,
              tok_fp: str | None, keep: int = 2) -> None:
    payload = {
        "step": step, "tokens_seen": tokens_seen, "elapsed_seconds": elapsed,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None, "scaler_state_dict": scaler.state_dict() if scaler else None,
        "tokenizer_fingerprint": tok_fp, "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # 保存の途中で止まっても、前のcheckpointが壊れないようにする
    olds = sorted(glob.glob(os.path.join(os.path.dirname(path), "checkpoint_[0-9]*.pt")))
    for old in olds[:-keep]:
        os.remove(old)


# --------------------------------------------------------------------------- 評価
@torch.no_grad()
def evaluate(model, data: PackedData, batch_size: int, iters: int, device, amp_dtype, seed: int = 0) -> dict:
    model.eval()
    out = {}
    for src in data.arrays:
        rng = np.random.default_rng(seed)  # 毎回同じ窓で測る（評価ごとの比較が公平になる）
        total = 0.0
        for _ in range(iters):
            x, y = data.batch(batch_size, {}, rng, device, only=src)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x).logits
            total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item()
        out[src] = total / iters
    model.train()
    return out


@torch.no_grad()
def sample_texts(model, codec: CodecTokenizer | None, device, prompts: list[str], max_new: int = 48) -> list[str]:
    if codec is None:
        return []
    model.eval()
    outs = []
    for p in prompts:
        ids = torch.tensor([codec.encode(p)], dtype=torch.long, device=device)
        gen = model.generate(ids, max_new_tokens=max_new, strategy="greedy")
        outs.append(codec.decode(gen[0].tolist()))
    model.train()
    return outs


# --------------------------------------------------------------------------- メイン
def pretrain(config_path: str) -> dict:
    cfg = load_config(config_path)
    exp = cfg.get("experiment_name", "p7_pretrain")
    pc, tc, oc = cfg["pretrain"], cfg["training"], cfg["optimizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = os.path.join(pc["token_dir"], "pack_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        pack_meta = json.load(f)
    vocab = pack_meta["vocab_size"]

    model_cfg = dict(cfg["model"])
    model_cfg["vocab_size"] = vocab
    block = model_cfg["max_seq_length"]
    model = TransformerLM(model_cfg).to(device)
    n_params = model.num_parameters()

    tok_path = cfg["data"].get("tokenizer_path")
    tok_fp = compute_tokenizer_fingerprint(tok_path)
    codec = None
    if tok_path and os.path.exists(tok_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(tok_path)
        codec = CodecTokenizer(sp)

    train_data = PackedData(pc["token_dir"], "train", block)
    val_data = PackedData(pc["token_dir"], "val", block)
    if not train_data.arrays:
        raise FileNotFoundError(f"{pc['token_dir']} に train_*.bin が見つかりません。先に pack_tokens pack を実行してください")
    mix = {s: float(w) for s, w in pc.get("mix", {"wiki": 0.7, "code": 0.3}).items()}
    mix = {s: w for s, w in mix.items() if s in train_data.arrays}
    print(f"[{exp}] パラメータ数 {n_params:,} / 語彙 {vocab} / 窓 {block}")
    print(f"  学習トークン数: {train_data.tokens()} / 混合比率: {mix}")

    micro_bs = tc["micro_batch_size"]
    accum = tc["grad_accum_steps"]
    tokens_per_step = micro_bs * accum * block
    total_tokens = int(tc.get("total_tokens") or 0)
    budget_s = float(tc.get("time_budget_hours") or 0) * 3600
    if not total_tokens and not budget_s:
        raise ValueError("training.total_tokens か training.time_budget_hours のどちらかを指定してください")
    total_steps = math.ceil(total_tokens / tokens_per_step) if total_tokens else 0
    if total_tokens:
        epochs = total_tokens / max(sum(train_data.tokens().values()), 1)
        print(f"  総 {total_tokens:,} トークン = {total_steps:,}ステップ（学習データ全体の約{epochs:.2f}周）")
    else:
        print(f"  時間予算 {budget_s / 3600:.1f}時間（1ステップ={tokens_per_step:,}トークン）")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 and "emb" not in n else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": oc.get("weight_decay", 0.1)}, {"params": no_decay, "weight_decay": 0.0}],
        lr=oc["lr"], betas=(oc.get("beta1", 0.9), oc.get("beta2", 0.95)), eps=oc.get("eps", 1e-8))
    amp_dtype, use_scaler = pick_amp(tc.get("amp", "auto"), device)
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    print(f"  混合精度: {amp_dtype if amp_dtype else 'なし'}")

    ckpt_dir = os.path.join(cfg.get("results_dir", "results/checkpoints"), exp)
    log_path = os.path.join(cfg.get("logs_dir", "logs"), f"{exp}_pretrain.jsonl")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    step, tokens_seen, elapsed_prev = 0, 0, 0.0
    if tc.get("resume", True):
        cks = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_[0-9]*.pt")))
        if cks:
            ck = torch.load(cks[-1], map_location=device)
            stored = ck.get("tokenizer_fingerprint")
            if stored and tok_fp and stored != tok_fp:
                raise RuntimeError(f"{cks[-1]} は別のトークナイザーで学習されています。ディレクトリを移動してやり直してください")
            model.load_state_dict(ck["model_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            if scaler and ck.get("scaler_state_dict"):
                scaler.load_state_dict(ck["scaler_state_dict"])
            step, tokens_seen, elapsed_prev = ck["step"], ck.get("tokens_seen", 0), ck.get("elapsed_seconds", 0.0)
            print(f"  再開: {cks[-1]}（step {step}, {tokens_seen:,}トークン, 経過{elapsed_prev / 3600:.2f}時間）")

    session_limit = tc.get("session_max_minutes", cfg.get("kaggle", {}).get("max_session_time_minutes", 540)) * 60 * 0.92
    rng = np.random.default_rng(1234 + step)
    t_session = time.time()
    warm = float(oc.get("warmup_fraction", 0.02))
    eval_every, ckpt_every = tc.get("eval_every", 500), tc.get("checkpoint_every", 500)
    prompts = tc.get("sample_prompts", ["日本の首都は", "def add(a, b):\n"])
    last_loss, t_log, tok_log = None, time.time(), tokens_seen
    model.train()
    done = False

    def progress() -> float:
        if total_steps:
            return step / total_steps
        return (elapsed_prev + time.time() - t_session) / budget_s

    while not done:
        prog = progress()
        if prog >= 1.0:
            break
        lr_now = lr_at(prog, oc["lr"], oc.get("min_lr", oc["lr"] * 0.1), warm)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        loss_acc = 0.0
        for _ in range(accum):
            x, y = train_data.batch(micro_bs, mix, rng, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x).logits
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)) / accum
            (scaler.scale(loss) if scaler else loss).backward()
            loss_acc += loss.item()
        if scaler:
            scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("max_grad_norm", 1.0))
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tokens_per_step
        last_loss = loss_acc
        if not math.isfinite(loss_acc):
            raise FloatingPointError(f"損失が発散しました（step {step}）。学習率を下げてやり直してください")

        elapsed = elapsed_prev + time.time() - t_session
        if step % tc.get("log_every", 50) == 0:
            now = time.time()
            tps = (tokens_seen - tok_log) / max(now - t_log, 1e-6)
            t_log, tok_log = now, tokens_seen
            rec = {"step": step, "tokens": tokens_seen, "loss": round(loss_acc, 4), "lr": lr_now,
                   "gnorm": round(float(gnorm), 3), "tok_per_s": round(tps), "elapsed_h": round(elapsed / 3600, 3)}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if step % eval_every == 0:
            vl = evaluate(model, val_data, micro_bs, tc.get("eval_iters", 40), device, amp_dtype)
            rec = {"step": step, "val_loss": {k: round(v, 4) for k, v in vl.items()},
                   "val_ppl": {k: round(math.exp(v), 2) for k, v in vl.items()}}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            for pr, out in zip(prompts, sample_texts(model, codec, device, prompts)):
                print(f"  [生成例] {pr!r} → {out!r}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        stop_for_session = (time.time() - t_session) > session_limit
        if step % ckpt_every == 0 or stop_for_session:
            save_ckpt(os.path.join(ckpt_dir, f"checkpoint_{step:07d}.pt"), model, optimizer, scaler,
                      step, tokens_seen, elapsed, tok_fp)
        if stop_for_session:
            print(f"⚠ セッション時間の上限に近づいたため、step {step} で保存して終了します。"
                  f"同じコマンドをもう一度実行すると続きから再開します。")
            return {"finished": False, "step": step, "tokens": tokens_seen}

    elapsed = elapsed_prev + time.time() - t_session
    final = os.path.join(ckpt_dir, f"checkpoint_{step:07d}.pt")
    save_ckpt(final, model, optimizer, scaler, step, tokens_seen, elapsed, tok_fp)
    torch.save({"step": step, "model_state_dict": model.state_dict(), "tokenizer_fingerprint": tok_fp,
                "tokens_seen": tokens_seen}, os.path.join(ckpt_dir, "pretrained_final.pt"))
    vl = evaluate(model, val_data, micro_bs, max(tc.get("eval_iters", 40), 100), device, amp_dtype)
    print(f"✓ 事前学習完了: {tokens_seen:,}トークン / {elapsed / 3600:.2f}時間 / 最終の検証損失 "
          f"{ {k: round(v, 4) for k, v in vl.items()} }")
    return {"finished": True, "step": step, "tokens": tokens_seen, "val_loss": vl}


def _main() -> None:
    ap = argparse.ArgumentParser(description="事前学習")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    print(json.dumps(pretrain(args.config), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _main()
