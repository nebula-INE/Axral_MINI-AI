"""
eval_yardstick.py
問題集 yardstick_v1（eval_sets/yardstick_v1.jsonl）でモデルを採点し、成長の度合いを測る。

使い方（Kaggleのセルで）:
  !cd /kaggle/working/Axral_MINI-AI && python -m src.eval_yardstick \\
      --config configs/exp_mix.yaml \\
      --checkpoint /kaggle/working/kaggle_dataset_mix/checkpoint_best_p2_mix_hf_11m.pt

  # 前回の結果と比べる
  !cd /kaggle/working/Axral_MINI-AI && python -m src.eval_yardstick \\
      --config configs/exp_p3_mix_lr3e4.yaml --checkpoint results/checkpoints/p3_mix_lr3e4/checkpoint_best.pt \\
      --baseline logs/yardstick_p2_mix_hf_11m.json

測る指標:
  モデル正答率   知識ガードを外した、モデル自身の力（学習済みの内容を別の言い方で聞いて答えられるか）
  ガード通過率   問題が知識ガード（is_known_topic）で「既知」と判定され、モデルに届く割合
                 （言い換えに弱いと、正しく答えられる問題まで「分かりません」になる）
  総合正答率     実際の利用で見える結果（ガードを通り、かつ正解）。範囲外の問題は「分かりません」が正解
  拒否率         範囲外の問題を、正しく「分かりません」にできた割合（ガードの評価）

算数は、計算機による自動訂正を使わず、モデルの素の計算力を測る（--with_calculator で有効化）。
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Callable

from src.knowledge_base import is_known_topic, is_refusal

DEFAULT_YARDSTICK = "eval_sets/yardstick_v1.jsonl"
_STRIP_CHARS = "、。，,．・（）()「」『』“”\"'！!？?：:；;~〜 \u3000\t\n"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if ch not in _STRIP_CHARS)


def matches(final_answer: str, expected_any: list[str]) -> bool:
    """期待語のいずれかが最終回答に含まれれば正解。数字だけの期待語は、独立した数として一致を見る
    （例: 期待「0」が「100」に誤って一致しないようにする）。"""
    nf = normalize(final_answer)
    for e in expected_any:
        ne = normalize(e)
        if not ne:
            continue
        if ne.isdigit():
            if re.search(rf"(?<!\d){re.escape(ne)}(?!\d)", nf):
                return True
        elif ne in nf:
            return True
    return False


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_items(items: list[dict], generate: Callable[[str], str], progress: bool = True) -> list[dict]:
    """各問題を採点する。generate(入力文) は「ガード無しのモデルの生出力」を返す関数。"""
    from src.utils import split_reasoning_answer

    results = []
    for i, it in enumerate(items, 1):
        raw = generate(it["input"])
        _reasoning, final = split_reasoning_answer(raw)
        guard_pass = is_known_topic(it["input"])
        model_refused = is_refusal(raw)  # モデル自身が「学習していないので答えられない」と答えたか（拒否学習の効果）
        if it["behavior"] == "refuse":
            model_correct = None
            e2e = (not guard_pass) or model_refused  # ガード、またはモデル自身が断れれば、正しい挙動
        else:
            model_correct = matches(final, it["expected_any"])
            e2e = bool(model_correct and guard_pass)
        results.append({**it, "raw": raw, "final": final, "guard_pass": guard_pass,
                        "model_refused": model_refused, "model_correct": model_correct, "e2e_correct": e2e})
        if progress and i % 20 == 0:
            print(f"  {i}/{len(items)} 問 採点済み...")
    return results


def _group_stats(rows: list[dict]) -> dict:
    inscope = [r for r in rows if r["behavior"] == "answer"]
    refuse = [r for r in rows if r["behavior"] == "refuse"]
    out = {"n": len(rows)}
    if inscope:
        out["n_inscope"] = len(inscope)
        out["model_acc"] = sum(bool(r["model_correct"]) for r in inscope) / len(inscope)
        out["guard_pass"] = sum(r["guard_pass"] for r in inscope) / len(inscope)
        out["e2e_acc"] = sum(r["e2e_correct"] for r in inscope) / len(inscope)
        out["false_refusal"] = sum(bool(r.get("model_refused")) for r in inscope) / len(inscope)
    if refuse:
        out["n_refuse"] = len(refuse)
        out["refusal_rate"] = sum(r["e2e_correct"] for r in refuse) / len(refuse)
        out["guard_refusal"] = sum(not r["guard_pass"] for r in refuse) / len(refuse)
        out["model_refusal"] = sum(bool(r.get("model_refused")) for r in refuse) / len(refuse)
    out["overall_e2e"] = sum(r["e2e_correct"] for r in rows) / len(rows)
    return out


def summarize(results: list[dict]) -> dict:
    summary = {"overall": _group_stats(results), "by_category": {}, "by_type": {}}
    for key, field in (("by_category", "category"), ("by_type", "type")):
        for name in sorted({r[field] for r in results}):
            summary[key][name] = _group_stats([r for r in results if r[field] == name])
    return summary


def _pct(x) -> str:
    return "  -  " if x is None else f"{x * 100:5.1f}%"


def print_report(summary: dict) -> None:
    o = summary["overall"]
    print("\n" + "=" * 64)
    print("【物差し yardstick_v1】学習で見ていない言い回しへの対応力")
    print("=" * 64)
    if "e2e_acc" in o:
        print(f"【見出し】答えられるはずの問題の総合正答率（ガードを通り、かつ正解）: {_pct(o['e2e_acc'])}"
              f"  （{o['n_inscope']}問）")
        print(f"  モデル正答率（ガード無し）        : {_pct(o['model_acc'])}  ← モデル自身の汎化力")
        print(f"  ガード通過率                      : {_pct(o['guard_pass'])}  ← 低いと、答えられる問題まで拒否している")
        if "false_refusal" in o:
            print(f"  過剰な拒否（答えられる問題をモデルが断った）: {_pct(o['false_refusal'])}  ← 低いほどよい")
    if "refusal_rate" in o:
        print(f"範囲外の拒否率（別指標、ガードかモデルのどちらかが断れた割合）: {_pct(o['refusal_rate'])}（{o['n_refuse']}問）")
        if "model_refusal" in o:
            print(f"  うち ガードだけ: {_pct(o['guard_refusal'])} ／ モデル自身（拒否学習の効果）: {_pct(o['model_refusal'])}")
    print("※ 範囲外の拒否を正解に含めると、全部「分かりません」でも点が取れてしまうため、見出しからは除いています。")
    for title, key in (("カテゴリ別", "by_category"), ("種類別", "by_type")):
        print(f"\n--- {title} ---")
        print(f"{'':14}{'問数':>5}{'モデル':>9}{'ガード':>9}{'総合':>9}{'拒否率':>9}")
        for name, g in summary[key].items():
            print(f"{name:14}{g['n']:>5}{_pct(g.get('model_acc')):>9}{_pct(g.get('guard_pass')):>9}"
                  f"{_pct(g.get('e2e_acc')):>9}{_pct(g.get('refusal_rate')):>9}")


def print_failures(results: list[dict], limit: int) -> None:
    fails = [r for r in results if not r["e2e_correct"]]
    print(f"\n--- 不正解の例（全{len(fails)}件中、先頭{min(limit, len(fails))}件）---")
    for r in fails[:limit]:
        if r["behavior"] == "refuse":
            print(f"[{r['category']}] Q: {r['input']}\n    正解: 分かりません ／ ガードが通してしまい、モデルは: {r['final'][:60]}")
        else:
            why = "ガードが拒否" if not r["guard_pass"] else "モデルの誤答"
            print(f"[{r['category']}/{r['type']}] Q: {r['input']}\n    正解に含まれるべき語: {r['expected_any']} ／ {why}"
                  f" ／ モデルの回答: {r['final'][:60]}")


def print_delta(base: dict, cur: dict) -> None:
    print("\n--- 前回との比較（現在 − 前回、ポイント）---")
    def d(a, b):
        return "   -  " if a is None or b is None else f"{(a - b) * 100:+6.1f}"
    print(f"{'':14}{'モデル':>9}{'ガード':>9}{'総合':>9}")
    rows = [("全体", base["overall"], cur["overall"])]
    for name, g in cur["by_category"].items():
        rows.append((name, base["by_category"].get(name, {}), g))
    for name, b, c in rows:
        print(f"{name:14}{d(c.get('model_acc'), b.get('model_acc')):>9}{d(c.get('guard_pass'), b.get('guard_pass')):>9}"
              f"{d(c.get('e2e_acc'), b.get('e2e_acc')):>9}")


def main():
    parser = argparse.ArgumentParser(description="物差し問題集でモデルを採点する")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--yardstick", default=DEFAULT_YARDSTICK)
    parser.add_argument("--out", default=None, help="結果JSONの保存先（既定: logs/yardstick_{実験名}.json）")
    parser.add_argument("--baseline", default=None, help="比較する前回の結果JSON")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--with_calculator", action="store_true", help="算数の計算機訂正を有効にする（既定は無効）")
    parser.add_argument("--show_fail", type=int, default=15, help="表示する不正解の件数")
    args = parser.parse_args()

    import torch
    from src.infer import generate_answer, load_model_and_tokenizer
    from src.utils import load_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    model, sp, tok_meta = load_model_and_tokenizer(config, args.checkpoint, device)

    def generate(text: str) -> str:
        return generate_answer(model, sp, tok_meta, text, device, max_new_tokens=args.max_new_tokens,
                               use_calculator=args.with_calculator, use_topic_guard=False)

    items = load_items(args.yardstick)
    print(f"物差し {args.yardstick}（{len(items)}問）で採点します...")
    results = evaluate_items(items, generate)
    summary = summarize(results)
    print_report(summary)
    print_failures(results, args.show_fail)

    exp = config.get("experiment_name", "model")
    out = Path(args.out or f"logs/yardstick_{exp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"experiment": exp, "checkpoint": args.checkpoint, "yardstick": args.yardstick,
                   "with_calculator": args.with_calculator, "summary": summary, "items": results},
                  f, ensure_ascii=False, indent=1)
    print(f"\n✓ 結果を保存しました: {out}")

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            print_delta(json.load(f)["summary"], summary)


if __name__ == "__main__":
    main()
