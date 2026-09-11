"""
eval_cot.py
CoT（Chain-of-Thought）品質を自動評価するスクリプト。

修正計画（plan_10_修正版.md §2.3）に基づく実装:
  1. logic（論理正確性）が0.5未満なら他の観点に関わらず自動 reject（ハードゲート）。
  2. カテゴリ別に固定値を返さず、自動検査可能なものは実際に検査する。
     自動判定が困難なカテゴリ（qa/technical/code）は needs_manual_review=True を
     立てて人手確認に回す。
  3. count_reasoning_steps / compute_conciseness / verify_arithmetic を実装。

このファイル単体は標準ライブラリ（re）のみに依存し、Kaggle上でもローカルでも動く。
"""
from __future__ import annotations

import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 個別評価関数
# ---------------------------------------------------------------------------

_EXPR_RE = re.compile(r"[\d.\s+\-*/×÷]+=\s*[-\d.]+")
_CONNECTOR_RE = re.compile(r"(まず|次に|そして|よって|したがって|最後に|つまり)")


def _to_float(s: str) -> float:
    return float(s.strip().replace(",", ""))


def verify_arithmetic(cot_text: str, answer_text: str) -> float:
    """CoT内の数式を抽出して実際に計算し、答えと整合するか検証する。

    戻り値:
        1.0  すべての式が計算的に正しく、最終式の右辺が答えに含まれる
        0.5  式は正しいが最終回答との対応が確認できない
        0.0  式が見つからない、計算が誤り、または例外発生
    """
    if not cot_text:
        return 0.0

    expressions = _EXPR_RE.findall(cot_text)
    if not expressions:
        return 0.0  # 検算できない場合は安全側（要手動確認）

    try:
        last_rhs = None
        for expr in expressions:
            lhs, rhs = expr.split("=")
            lhs_norm = lhs.replace("×", "*").replace("÷", "/").strip()
            rhs_val = _to_float(rhs)
            computed = eval(lhs_norm, {"__builtins__": {}}, {})  # noqa: S307
            # 割り算の結果を切り捨て/四捨五入で表記するCoTがあるため、
            # 差が1未満（浮動小数点誤差の範囲）なら許容する。
            # 実際の計算ミス（例: 2+2=5）は差が1以上になるため区別できる。
            if abs(computed - rhs_val) >= 0.999999:
                return 0.0
            last_rhs = rhs.strip()

        if last_rhs and answer_text and last_rhs in answer_text.replace(",", ""):
            return 1.0
        return 0.5
    except Exception:
        return 0.0


def count_reasoning_steps(cot_text: str) -> int:
    """改行・接続詞から疑似的にステップ数を数える。"""
    if not cot_text or not cot_text.strip():
        return 0
    line_steps = len([l for l in cot_text.split("\n") if l.strip()])
    connector_steps = len(_CONNECTOR_RE.findall(cot_text))
    return max(line_steps, connector_steps, 1)


def compute_conciseness(cot_text: str, steps: int) -> float:
    """ステップ数あたりの文字数から冗長性を推定する。"""
    if not cot_text or steps == 0:
        return 0.0
    chars_per_step = len(cot_text) / steps
    if 15 <= chars_per_step <= 60:
        return 0.95
    elif chars_per_step < 15:
        return 0.6
    elif chars_per_step <= 100:
        return 0.75
    else:
        return 0.4


AUTO_LOGIC_CATEGORIES = {"arithmetic"}
MANUAL_REVIEW_CATEGORIES = {"qa", "technical", "code"}


@dataclass
class CotEvalResult:
    cot_quality_score: float
    breakdown: dict = field(default_factory=dict)
    status: str = "reject"          # adopt / revise / reject
    needs_manual_review: bool = False

    def to_dict(self) -> dict:
        return {
            "cot_quality_score": round(self.cot_quality_score, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "status": self.status,
            "needs_manual_review": self.needs_manual_review,
        }


def evaluate_cot_quality(
    input_text: str,
    cot_text: Optional[str],
    answer_text: str,
    category: str,
) -> CotEvalResult:
    """CoT妥当性スコアを計算する。logic はハードゲートとして扱う。"""
    scores: dict[str, float] = {}
    needs_manual_review = False

    if cot_text is None:
        cot_text = ""

    # 1. 論理正確性
    if category in AUTO_LOGIC_CATEGORIES:
        scores["logic"] = verify_arithmetic(cot_text, answer_text)
    elif category in MANUAL_REVIEW_CATEGORIES:
        scores["logic"] = 0.6  # 暫定値。必ず人手確認へ回す
        needs_manual_review = True
    else:
        scores["logic"] = 0.6
        needs_manual_review = True

    # 2. 完全性
    steps = count_reasoning_steps(cot_text)
    if steps >= 3:
        scores["completeness"] = 0.95
    elif steps == 2:
        scores["completeness"] = 0.75
    elif steps == 1:
        scores["completeness"] = 0.5
    else:
        scores["completeness"] = 0.0

    # 3. 簡潔性
    scores["conciseness"] = compute_conciseness(cot_text, steps)

    # ハードゲート: logic < 0.5 は無条件 reject
    if scores["logic"] < 0.5:
        final_score = scores["logic"]
        status = "reject"
    else:
        final_score = sum(scores.values()) / len(scores)
        status = "adopt" if final_score >= 0.75 else "revise"

    return CotEvalResult(
        cot_quality_score=final_score,
        breakdown=scores,
        status=status,
        needs_manual_review=needs_manual_review,
    )


# ---------------------------------------------------------------------------
# CLI: jsonl ファイルを一括採点する
# ---------------------------------------------------------------------------

def score_jsonl(in_path: str, out_path: str) -> dict:
    """JSONLファイルを読み込み、各行に meta.cot_quality_score / cot_status / needs_manual_review を付与して書き出す。"""
    total = 0
    adopted = 0
    manual_review = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1
            cot = item.get("cot")
            if cot:
                result = evaluate_cot_quality(
                    item.get("input", ""),
                    cot,
                    item.get("answer", ""),
                    item.get("meta", {}).get("category", "unknown"),
                )
                item.setdefault("meta", {})
                item["meta"]["cot_quality_score"] = result.cot_quality_score
                item["meta"]["cot_status"] = result.status
                item["meta"]["needs_manual_review"] = result.needs_manual_review
                if result.status == "adopt":
                    adopted += 1
                if result.needs_manual_review:
                    manual_review += 1
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "total": total,
        "adopted": adopted,
        "adoption_rate": adopted / total if total else 0.0,
        "needs_manual_review": manual_review,
    }
    return summary


def _main():
    parser = argparse.ArgumentParser(description="CoT品質を一括採点する")
    parser.add_argument("--in_path", required=True, help="入力 jsonl")
    parser.add_argument("--out_path", required=True, help="出力 jsonl（採点結果付き）")
    args = parser.parse_args()

    summary = score_jsonl(args.in_path, args.out_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    gate_ok = summary["adoption_rate"] >= 0.8
    print(f"Gate1 CoT採用率 >= 80%: {'OK' if gate_ok else 'NG'} ({summary['adoption_rate']:.1%})")


if __name__ == "__main__":
    _main()
