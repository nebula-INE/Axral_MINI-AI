"""
sanity_check.py
val setの質問をそのまま入力し、想定通りの答えが返るかを確認する。

目的: 「未知の言い回しに弱い」のが汎化不足なのか、それとも
学習データ自体に対しても答えられていない（学習そのものに問題がある）のかを切り分ける。

  - 学習データの質問には高精度で答えられる → 汎化不足（データの多様性を増やすべき）
  - 学習データの質問にすら答えられない     → 学習/評価パイプライン自体に問題がある

実行例:
  python -m src.sanity_check --config configs/exp01_kaggle.yaml \
      --checkpoint /kaggle/input/datasets/sekoia29/mini-ai/checkpoint_best.pt \
      --val_path data/v001.val.jsonl \
      --num_samples 50
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch

from src.eval import _normalize_text, extract_final_answer, extract_final_segment
from src.infer import load_model_and_tokenizer, generate_answer
from src.utils import load_config


def extract_answer_for_category(generated: str, category: str) -> str:
    """カテゴリに応じて適切な抽出関数を使う（eval.pyのevaluate()と同じロジック）。"""
    if category == "arithmetic":
        return extract_final_answer(generated)
    return extract_final_segment(generated)


def main():
    parser = argparse.ArgumentParser(description="学習データに対する丸暗記度を確認するサニティチェック")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_path", default="data/v001.val.jsonl",
                        help="トークン化前の生データ（input/answer/meta.categoryを読む）")
    parser.add_argument("--num_samples", type=int, default=50, help="サンプル数（多いと時間がかかる）")
    parser.add_argument("--show_examples", type=int, default=3, help="カテゴリごとに表示する失敗例の数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    model, sp, tok_meta = load_model_and_tokenizer(config, args.checkpoint, device)

    items = []
    with open(args.val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    sample = random.sample(items, min(args.num_samples, len(items)))

    results = defaultdict(lambda: {"correct": 0, "total": 0, "failures": []})

    print(f"サニティチェック開始（{len(sample)}件、val_path={args.val_path}）...\n")

    for i, item in enumerate(sample):
        category = item.get("meta", {}).get("category", "unknown")
        question = item["input"]
        true_answer = item.get("answer", "")

        generated = generate_answer(model, sp, tok_meta, question, device, max_new_tokens=128)
        extracted = extract_answer_for_category(generated, category)

        is_correct = _normalize_text(extracted) == _normalize_text(true_answer)
        results[category]["total"] += 1
        if is_correct:
            results[category]["correct"] += 1
        else:
            results[category]["failures"].append({
                "question": question,
                "true_answer": true_answer,
                "generated": generated,
                "extracted": extracted,
            })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sample)} 件処理済み...")

    print("\n" + "=" * 60)
    print("【サニティチェック結果】学習データ（val set）への正答率")
    print("=" * 60)

    overall_correct = sum(r["correct"] for r in results.values())
    overall_total = sum(r["total"] for r in results.values())
    print(f"\n全体: {overall_correct}/{overall_total} ({overall_correct / overall_total:.1%})\n")

    for category, r in sorted(results.items()):
        acc = r["correct"] / r["total"] if r["total"] else 0
        print(f"  {category:15s}: {r['correct']:3d}/{r['total']:3d} ({acc:.1%})")

    print("\n" + "=" * 60)
    print("【判定】")
    if overall_correct / overall_total >= 0.8:
        print("✓ 学習データへの正答率は高い（80%以上）。")
        print("  → 学習・評価パイプライン自体は正常に機能している。")
        print("  → 未知の言い回しへの弱さは「データの多様性不足による汎化性能の問題」と確定。")
        print("    対策: テンプレートのバリエーションを増やす（数値ではなく言い回し自体を増やす）。")
    elif overall_correct / overall_total >= 0.3:
        print("△ 学習データへの正答率は中程度。汎化不足と学習不足の両方が疑われる。")
        print("  → まずエポック数を増やすか、学習率などのハイパーパラメータを見直す価値がある。")
    else:
        print("✗ 学習データにすら答えられていない（30%未満）。")
        print("  → データの多様性以前に、学習・評価パイプライン自体を疑うべき。")
        print("    確認ポイント: token_idsが正しくSentencePieceで作られているか、")
        print("    generate()のprompt_ids構成、answer抽出ロジックがカテゴリに対応しているか等。")
    print("=" * 60)

    print(f"\n【カテゴリ別 失敗例（各最大{args.show_examples}件）】")
    for category, r in sorted(results.items()):
        if not r["failures"]:
            continue
        print(f"\n--- {category} ---")
        for f in r["failures"][: args.show_examples]:
            print(f"  Q: {f['question']}")
            print(f"  正解: {f['true_answer']}")
            print(f"  生成(抽出後): {f['extracted']}")
            print(f"  生成(全文): {f['generated'][:100]}...")
            print()


if __name__ == "__main__":
    main()
