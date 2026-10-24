"""
decode_compare.py
同じ質問・同じ重みで、デコード設定（繰り返しペナルティなど）だけを変えて出力を比べる。
GPUは不要（CPUで数分）。ガードと計算機は切り、モデル本来の出力を見る。

実行例:
  python -m src.decode_compare --config configs/exp_p8_sft.yaml \
      --checkpoint results/checkpoints/p8_sft/checkpoint_best.pt --n 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
from statistics import mean

# A: そのまま / B: 軽い繰り返しペナルティ / C: ペナルティ+同じ3語の連続を禁止 / D: C+サンプリング
CONFIGS = [
    ("A_greedy", dict(strategy="greedy", repetition_penalty=1.0, no_repeat_ngram_size=0)),
    ("B_rp1.1", dict(strategy="greedy", repetition_penalty=1.1, no_repeat_ngram_size=0)),
    ("C_rp1.15_ng3", dict(strategy="greedy", repetition_penalty=1.15, no_repeat_ngram_size=3)),
    ("D_sample_rp1.1_ng3", dict(strategy="sampling", temperature=0.7, repetition_penalty=1.1, no_repeat_ngram_size=3)),
]

BUILTIN_PROMPTS = [
    "日本の首都は？", "水が凍るのは何度？", "空気の主成分は？", "人は呼吸で何の気体を使う？",
    "太陽系で一番大きい惑星は？", "心臓の部屋はいくつ？", "光合成とは何ですか？",
    "3人で1200円を割り勘すると1人いくら？", "富士山の高さは？", "Pythonでリストの長さを求めるには？",
]


def rep_ratio(text: str, n: int = 4) -> float:
    """文字n-gramの重複率（0=繰り返しなし、1に近いほど同じ並びの繰り返し）。短すぎる出力は0。"""
    s = re.sub(r"\s+", "", text or "")
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    if len(grams) < 8:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def keyword_hit(text: str, keywords) -> bool | None:
    """正解語が1つも指定されていなければ None。どれか1つでも含まれていれば True。"""
    if not keywords:
        return None
    return any(str(k) in (text or "") for k in keywords)


def load_prompts(yardstick: str | None, n: int) -> list[dict]:
    """物差しの先頭n問（正解語の列があれば拾う）。無ければ組み込みの短い質問を使う。"""
    items: list[dict] = []
    if yardstick and os.path.exists(yardstick):
        with open(yardstick, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                kw = None
                for k, v in row.items():
                    if isinstance(v, list) and v and all(isinstance(x, str) for x in v) and \
                            re.search(r"word|keyword|expect|must|answer", k, re.I):
                        kw = v
                        break
                items.append({"input": row["input"], "keywords": kw})
                if len(items) >= n:
                    break
    if not items:
        items = [{"input": p, "keywords": None} for p in BUILTIN_PROMPTS[:n]]
    return items


def summarize(results: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for name, outs in results.items():
        hits = [o["hit"] for o in outs if o["hit"] is not None]
        rows.append({
            "設定": name,
            "反復率(低いほど良い)": round(mean(o["rep"] for o in outs), 3),
            "反復が強い出力の数": sum(1 for o in outs if o["rep"] >= 0.5),
            "正解語を含む率": (round(sum(hits) / len(hits), 3) if hits else None),
            "平均文字数": round(mean(len(o["text"]) for o in outs), 1),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="デコード設定の比較（反復の度合い・正解語）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--yardstick", default="eval_sets/yardstick_v1.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--show", type=int, default=5, help="設定ごとに表示する出力例の数")
    ap.add_argument("--out", default="logs/decode_compare.json")
    args = ap.parse_args()

    import torch  # 遅延import（指標の関数だけテストできるように）
    from src.infer import generate_answer, load_model_and_tokenizer
    from src.utils import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sp, tok_meta = load_model_and_tokenizer(load_config(args.config), args.checkpoint, device)
    prompts = load_prompts(args.yardstick, args.n)
    print(f"{len(prompts)}問 × {len(CONFIGS)}設定（device={device}）")

    results: dict[str, list[dict]] = {}
    for name, kw in CONFIGS:
        outs = []
        for p in prompts:
            text = generate_answer(model, sp, tok_meta, p["input"], device, max_new_tokens=args.max_new_tokens,
                                   use_calculator=False, use_topic_guard=False, **kw)
            outs.append({"q": p["input"], "text": text, "rep": rep_ratio(text), "hit": keyword_hit(text, p["keywords"])})
        results[name] = outs

    print("\n=== 比較 ===")
    rows = summarize(results)
    for r in rows:
        print(r)
    print("\n=== 出力例 ===")
    for name, outs in results.items():
        print(f"\n[{name}]")
        for o in outs[:args.show]:
            print(f"  Q: {o['q']}\n  A: {o['text'][:120]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": rows, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\n✓ 保存: {args.out}")
    best = min(rows, key=lambda r: (r["反復率(低いほど良い)"], -(r["正解語を含む率"] or 0)))
    print(f"反復が最も少ない設定: {best['設定']}（正解語を含む率も合わせて見て選んでください）")


if __name__ == "__main__":
    main()
