"""
bench_throughput.py
モデルの大きさごとに「1秒に何トークン学習できるか」を測り、GPU時間の予算で何トークン読ませられるかを見積もる。
事前学習のモデルサイズを決める前に、まずこれを実行する（所要: 数分）。

  !cd /kaggle/working/Axral_MINI-AI && python -m src.bench_throughput --budget_hours 20 --vocab_size 32000

見方:
  - 「予算内トークン数」÷「パラメータ数」が約20以上なら、その大きさは予算に見合う（Chinchilla則の目安）。
    これが20を大きく下回る大きさは、予算に対してモデルが大きすぎる（学習不足になる）。
  - 「推奨」は、その目安を満たす中で最大のモデル。ただし、小さいモデルを多めのトークンで学習するほうが
    使うときの計算量は軽く、品質も良いことが多いので、1段小さいサイズを選ぶのも妥当。
  - 実測は合成データ（ランダムなトークン）で、実際の学習より少し速く出る。データ読み込みなどの分は
    余裕を見て、--efficiency（既定0.85）を掛けている。
  - このコードは1枚のGPUしか使わない（複数GPUの並列学習は未対応）。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from src.model import TransformerLM

SIZES = [  # (名前, d_model, n_layers, n_heads)。d_ff は 4 × d_model
    ("S", 384, 8, 6),
    ("M", 512, 8, 8),
    ("L", 640, 10, 10),
    ("XL", 768, 12, 12),
    ("XXL", 896, 14, 14),
]


def build(vocab: int, d: int, layers: int, heads: int, block: int, dropout: float = 0.0) -> TransformerLM:
    cfg = dict(vocab_size=vocab, d_model=d, n_layers=layers, n_heads=heads, d_ff=4 * d, max_seq_length=block,
               dropout=dropout, attention_dropout=dropout, residual_dropout=dropout)
    return TransformerLM(cfg)


def measure(model, vocab: int, block: int, micro_bs: int, steps: int, warmup: int, device, amp_dtype, use_scaler):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    x = torch.randint(0, vocab, (micro_bs, block), device=device)
    y = torch.randint(0, vocab, (micro_bs, block), device=device)
    model.train()
    for i in range(warmup + steps):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x).logits
        loss = F.cross_entropy(logits.float().view(-1, vocab), y.view(-1))
        (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return steps * micro_bs * block / dt


def main() -> None:
    ap = argparse.ArgumentParser(description="学習速度の測定とモデルサイズの見積もり")
    ap.add_argument("--budget_hours", type=float, default=20.0, help="事前学習に使えるGPU時間")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--micro_bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--efficiency", type=float, default=0.85)
    ap.add_argument("--amp", choices=["auto", "fp16", "bf16", "off"], default="auto")
    ap.add_argument("--available_tokens", type=float, default=0,
                    help="用意できる学習データの総トークン数（分かれば指定。足りないときは注意を表示する）")
    ap.add_argument("--out", default="logs/bench_throughput.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠ GPUが見つかりません。測定値は実際の学習の参考になりません（Notebookの設定でGPUを有効にしてください）")
    amp_dtype, use_scaler = None, False
    if device.type == "cuda" and args.amp != "off":
        if args.amp == "bf16" or (args.amp == "auto" and torch.cuda.is_bf16_supported()):
            amp_dtype = torch.bfloat16
        else:
            amp_dtype, use_scaler = torch.float16, True
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"GPU: {gpu} / 混合精度: {amp_dtype} / 窓 {args.block} / 語彙 {args.vocab_size} / 予算 {args.budget_hours}時間\n")

    rows = []
    for name, d, layers, heads in SIZES:
        bs = args.micro_bs
        tps, err = None, None
        while bs >= 1:
            model = None
            try:
                model = build(args.vocab_size, d, layers, heads, args.block).to(device)
                tps = measure(model, args.vocab_size, args.block, bs, args.steps, args.warmup, device,
                              amp_dtype, use_scaler)
                break
            except torch.cuda.OutOfMemoryError:
                bs //= 2
                err = "OOM"
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        n_params = build(args.vocab_size, d, layers, heads, args.block).num_parameters()
        if tps is None:
            rows.append({"name": name, "params": n_params, "error": err})
            print(f"  {name}: {n_params / 1e6:.1f}M → 測定できませんでした（{err}）")
            continue
        tokens = tps * args.efficiency * args.budget_hours * 3600
        rows.append({"name": name, "d_model": d, "n_layers": layers, "n_heads": heads, "d_ff": 4 * d,
                     "params": n_params, "micro_bs": bs, "tok_per_s": round(tps),
                     "budget_tokens": round(tokens), "tokens_per_param": round(tokens / n_params, 1)})
        print(f"  {name}: {n_params / 1e6:6.1f}M パラメータ  {tps:>9,.0f} トークン/秒  "
              f"予算内 {tokens / 1e9:5.2f}B トークン  (パラメータあたり {tokens / n_params:6.1f}) micro_bs={bs}")

    ok = [r for r in rows if "tok_per_s" in r and r["tokens_per_param"] >= 20]
    print()
    if ok:
        best = max(ok, key=lambda r: r["params"])
        print(f"推奨（パラメータあたり20トークン以上を満たす最大）: {best['name']} = d_model {best['d_model']}, "
              f"layers {best['n_layers']}, heads {best['n_heads']}, d_ff {best['d_ff']} "
              f"（約{best['params'] / 1e6:.0f}M、予算内 {best['budget_tokens'] / 1e9:.2f}B トークン）")
        if args.available_tokens and best["budget_tokens"] > args.available_tokens:
            print(f"  ⚠ 用意できるデータ({args.available_tokens / 1e9:.2f}B)が予算内トークン数より少ないため、"
                  f"同じデータを繰り返して読むことになります。小さめのサイズか、データの追加を検討してください。")
    else:
        print("パラメータあたり20トークンを満たすサイズがありません。S より小さい構成を試すか、予算を増やしてください。")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"gpu": gpu, "amp": str(amp_dtype), "budget_hours": args.budget_hours, "efficiency": args.efficiency,
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
