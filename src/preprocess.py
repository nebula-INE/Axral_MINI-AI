"""
preprocess.py
plan §3 の前処理パイプライン実装。

手順:
  1. generate_data.py で生成した v001.train.jsonl / v001.val.jsonl を入力
  2. CoT評価スコア（§2.3）を計算し meta に追加
  3. 簡易トークン化（文字→トークンID）を実行し token_ids / answer_token_ids を追加
  4. v001.train.jsonl, v001.val.jsonl を上書き保存
  5. preprocessing_log.json を出力

実行: python src/preprocess.py --train_path data/v001.train.jsonl --val_path data/v001.val.jsonl --output_dir data/
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.eval_cot import evaluate_cot_quality


def simple_tokenize(text: str) -> list[int]:
    """
    超簡易トークン化（デモ用）。
    本来はSentencePieceなどで真正なトークン化を行うべきだが、このスクリプト段階では
    文字をUTF-8のバイト値に変換するだけで済ます。実際には §3.1段階8で
    SentencePiece学習（corpus.txtから）で置き換える。
    
    args_model_from_path() で sp.model をロードした場合は
    token_ids = sp.EncodeAsIds(text) に置き換える。
    """
    tokens = []
    for char in text:
        # UTF-8のバイト値をトークンとする（0-255）
        for byte in char.encode("utf-8"):
            tokens.append(byte + 256)  # オフセット（vocab_size=512想定）
    return tokens


def process_jsonl(input_path: str, output_path: str, is_train: bool = False) -> dict:
    """
    JSONL ファイルを読み込み、CoT評価 + トークン化を行い、上書き保存する。
    """
    stats = {
        "total": 0,
        "adopt": 0,
        "revise": 0,
        "reject": 0,
        "manual_review": 0,
        "avg_cot_score": 0.0,
    }
    cot_scores = []

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            stats["total"] += 1

            # === CoT評価（段階4） ===
            if item.get("cot"):
                result = evaluate_cot_quality(
                    item.get("input", ""),
                    item["cot"],
                    item.get("answer", ""),
                    item.get("meta", {}).get("category", "unknown"),
                )
                item.setdefault("meta", {})
                item["meta"]["cot_quality_score"] = result.cot_quality_score
                item["meta"]["cot_status"] = result.status
                item["meta"]["needs_manual_review"] = result.needs_manual_review

                cot_scores.append(result.cot_quality_score)
                stats[result.status] += 1
                if result.needs_manual_review:
                    stats["manual_review"] += 1

            # === トークン化（段階9） ===
            item["token_ids"] = simple_tokenize(item.get("input", ""))
            if item.get("cot"):
                item["cot_token_ids"] = simple_tokenize(item["cot"])
            else:
                item["cot_token_ids"] = []
            item["answer_token_ids"] = simple_tokenize(item.get("answer", ""))

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    if cot_scores:
        stats["avg_cot_score"] = sum(cot_scores) / len(cot_scores)

    return stats


def main():
    parser = argparse.ArgumentParser(description="前処理パイプライン")
    parser.add_argument("--train_path", required=True, help="train.jsonl")
    parser.add_argument("--val_path", required=True, help="val.jsonl")
    parser.add_argument("--output_dir", default="data/", help="出力ディレクトリ")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "version": "v001",
        "timestamp": datetime.now().isoformat(),
        "stages": {}
    }

    print("前処理パイプライン開始...")

    # 出力ファイルを明示的に指定（上書きしない）
    train_output = str(output_dir / "v001_processed.train.jsonl")
    val_output = str(output_dir / "v001_processed.val.jsonl")

    # === Train データ処理 ===
    print(f"段階4-9: Train データ処理中...", end="", flush=True)
    train_stats = process_jsonl(args.train_path, train_output)
    log["stages"]["train_processing"] = train_stats
    print(f" ✓ ({train_stats['total']}件)")

    # === Val データ処理 ===
    print(f"段階4-9: Val データ処理中...", end="", flush=True)
    val_stats = process_jsonl(args.val_path, val_output)
    log["stages"]["val_processing"] = val_stats
    print(f" ✓ ({val_stats['total']}件)")

    # === ゲート1確認 ===
    total_cot = train_stats["adopt"] + train_stats["revise"] + train_stats["reject"]
    adoption_rate = train_stats["adopt"] / total_cot if total_cot > 0 else 0
    gate1_pass = adoption_rate >= 0.80
    log["stages"]["gate1"] = {
        "adoption_rate": adoption_rate,
        "pass": gate1_pass,
        "message": "OK" if gate1_pass else "NG（要改善。CoTをより多段ステップに分解すること）"
    }

    print(f"\n【Gate 1: データ品質】")
    print(f"  Train の CoT 採用率: {adoption_rate:.1%} {'✓' if gate1_pass else '✗'}")
    print(f"  採用: {train_stats['adopt']}, 修正: {train_stats['revise']}, "
          f"削除: {train_stats['reject']}, 要手動確認: {train_stats['manual_review']}")
    print(f"  平均 CoT スコア: {train_stats['avg_cot_score']:.3f}")

    # ログ保存
    log_path = output_dir / f"v001_preprocessing_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {log_path} 保存完了")

    print(f"\n✅ 前処理完了")
    print(f"   {train_output}")
    print(f"   {val_output}")


if __name__ == "__main__":
    main()
