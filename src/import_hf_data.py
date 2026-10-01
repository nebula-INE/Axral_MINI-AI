"""
import_hf_data.py
Hugging Faceの公開日本語指示データ（dolly-15k-ja / oasst1-21k-ja / oasst2-33k-ja）を取得し、
このプロジェクトの学習用jsonl形式（id / input / cot / answer / meta）に変換して保存する。

実行例（Kaggleノートブックのセルで）:
  !cd /kaggle/working/Axral_MINI-AI && python -m src.import_hf_data

出力（デフォルトは /kaggle/working/hf_import/）:
  hf_import.train.jsonl   学習用
  hf_import.val.jsonl     検証用（全体の約2%）
  hf_corpus.txt           トークナイザー再学習用（input と answer を1行ずつ）

設計メモ:
  - cot は空文字にする（これらのデータには思考過程が無いため。cot_ratioの仕組みと
    同じく、cot空のサンプルは「input answer」だけで学習される）。
  - 7.6Mの小型モデルと max_seq_length=512 に合わせ、長い入力・回答は除外する。
  - dollyのcontext（参照文）付きの行は、入力が長くなるため除外する。
  - oasstは複数ターンの会話だが、前の発言に依存する後半のターンは文脈が失われるため、
    最初の「ユーザー発言→アシスタント返答」の1組だけを使う。
  - oasst系の列名は未確認のため、複数の形式に対応し、取得件数を必ず表示する。
  - 各データセットのライセンスは、利用前にHugging Faceのデータセットページで確認すること。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

DATASETS = [
    ("llm-jp/databricks-dolly-15k-ja", "dolly"),
    ("llm-jp/oasst1-21k-ja", "oasst1"),
    ("llm-jp/oasst2-33k-ja", "oasst2"),
]

# dollyのカテゴリのうち、一問一答に近いものはqa、それ以外は会話として扱う。
_QA_CATEGORIES = {"closed_qa", "open_qa", "general_qa"}
_USER_ROLES = {"user", "human", "prompter"}
_ASSISTANT_ROLES = {"assistant", "gpt", "bot"}
_URL_RE = re.compile(r"https?://")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _pairs_from_dolly(row: dict) -> list[tuple[str, str, str]]:
    if _norm(row.get("context")):
        return []  # 参照文付きは入力が長くなるため除外
    category = "qa" if row.get("category") in _QA_CATEGORIES else "conversation"
    return [(_norm(row.get("instruction")), _norm(row.get("response")), category)]


def _turns(row: dict) -> list[tuple[str, str]] | None:
    for key in ("conversations", "messages", "conversation"):
        value = row.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            turns = []
            for t in value:
                role = str(t.get("role") or t.get("from") or "").lower()
                text = t.get("text") or t.get("content") or t.get("value") or ""
                turns.append((role, _norm(text)))
            return turns
    return None


def _pairs_from_chat(row: dict) -> list[tuple[str, str, str]]:
    turns = _turns(row)
    if turns is None:
        instr = row.get("instruction") or row.get("input") or row.get("question") or row.get("text")
        resp = row.get("output") or row.get("response") or row.get("answer")
        if instr and resp:
            return [(_norm(instr), _norm(resp), "conversation")]
        return []
    for (r1, t1), (r2, t2) in zip(turns, turns[1:]):
        if r1 in _USER_ROLES and r2 in _ASSISTANT_ROLES:
            return [(t1, t2, "conversation")]  # 最初の1組だけ使う
    return []


def _extract_pairs(row: dict, source: str) -> list[tuple[str, str, str]]:
    return _pairs_from_dolly(row) if source == "dolly" else _pairs_from_chat(row)


def _keep(q: str, a: str, max_input: int, max_answer: int) -> bool:
    if not q or not a:
        return False
    if len(q) > max_input or len(a) > max_answer:
        return False
    if _URL_RE.search(q) or _URL_RE.search(a):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="HF公開日本語指示データを学習用jsonlに変換")
    parser.add_argument("--output_dir", default="/kaggle/working/hf_import")
    parser.add_argument("--max_input_chars", type=int, default=120)
    parser.add_argument("--max_answer_chars", type=int, default=200)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_dataset", type=int, default=0,
                        help="各データセットから使う最大件数（0なら制限なし。動作確認用）")
    args = parser.parse_args()

    from datasets import load_dataset  # 遅延import（変換関数だけテストできるように）

    items: list[dict] = []
    seen_inputs: set[str] = set()

    for repo, source in DATASETS:
        try:
            ds = load_dataset(repo, split="train")
        except Exception as e:  # ネットワーク不通・名前変更など
            print(f"[skip] {repo}: 読み込めませんでした ({type(e).__name__}: {e})")
            continue

        extracted = kept = 0
        for row in ds:
            pairs = _extract_pairs(row, source)
            extracted += len(pairs)
            for q, a, category in pairs:
                if not _keep(q, a, args.max_input_chars, args.max_answer_chars):
                    continue
                if q in seen_inputs:
                    continue
                seen_inputs.add(q)
                items.append({
                    "id": f"{source}_{kept:06d}",
                    "input": q,
                    "cot": "",
                    "answer": a,
                    "meta": {"category": category, "source": source, "has_cot": False},
                })
                kept += 1
                if args.max_per_dataset and kept >= args.max_per_dataset:
                    break
            if args.max_per_dataset and kept >= args.max_per_dataset:
                break

        if extracted == 0 and len(ds) > 0:
            # 1件も取り出せなかった＝列名が想定外の可能性が高い。デバッグ用に列名と最初の1行を表示する
            first = ds[0]
            print(f"  [注意] {repo}: 会話ペアを1件も取り出せませんでした。列名: {list(first.keys())}")
            print(f"  最初の行: {str(first)[:300]}")

        print(f"{repo}: 全{len(ds)}行 → ペア抽出{extracted}件 → 長さ等の条件を満たした{kept}件")

    if not items:
        print("変換できたデータがありません。上の表示（列名など）を確認してください。")
        return

    random.Random(args.seed).shuffle(items)
    n_val = max(1, int(len(items) * args.val_ratio))
    val, train = items[:n_val], items[n_val:]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("hf_import.train.jsonl", train), ("hf_import.val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out / "hf_corpus.txt").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item["input"] + "\n" + item["answer"] + "\n")

    print(f"\n✓ 保存しました: {out}")
    print(f"  train {len(train)}件 / val {len(val)}件")
    by_cat: dict[str, int] = {}
    for item in items:
        by_cat[item["meta"]["category"]] = by_cat.get(item["meta"]["category"], 0) + 1
    print(f"  カテゴリ内訳: {by_cat}")
    print(f"  サンプル: {items[0]['input']} → {items[0]['answer']}")


if __name__ == "__main__":
    main()
