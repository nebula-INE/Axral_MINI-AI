# CoT評価ルーブリック（実装対応版）

`src/eval_cot.py` の実装と1対1で対応するルーブリック。plan_10_修正版.md §2.3 参照。

## スコア構成

| 観点 | 重み | 自動判定方法 |
|------|------|------|
| 論理正確性 (logic) | ハードゲート | `arithmetic`: 式を実際に計算し検算。`qa`/`technical`/`code`: 自動判定不可のため暫定値0.6＋`needs_manual_review=True` |
| 完全性 (completeness) | 平均の1/3 | 改行数・接続詞（まず/次に/よって等）からステップ数を推定 |
| 簡潔性 (conciseness) | 平均の1/3 | 1ステップあたりの文字数（目安15〜60字が適正） |

## 判定ロジック

```
if logic < 0.5:
    status = "reject"              # 他の観点に関わらず不採用
else:
    final_score = mean(logic, completeness, conciseness)
    status = "adopt"  if final_score >= 0.75
             "revise" if 0.5 <= final_score < 0.75
```

## 運用上の注意（実データで判明した点）

サンプル8件（`data/v001_sample_cot.jsonl`）で試したところ、1文で完結する短いCoT
（例:「1200 ÷ 3 = 400」のみ）は `completeness` が0.5にとどまり、`logic=1.0` でも
全体スコアが0.75未満で `revise` 判定になる。Gate 1 の採用率80%を満たすには、
**CoTを最低2〜3ステップに分解して書く**運用ルールをデータ作成時点で徹底する必要がある
（例:「合計を確認する。1200円。→ 人数で割る。1200 ÷ 3 = 400。→ 答えは400円。」）。

`qa` / `technical` / `code` カテゴリは自動では論理正確性を保証できないため、
`needs_manual_review=True` が付いたサンプルは前処理パイプライン段階4で
人手レビューのキューに回すこと（自動でadopt/rejectを確定させない）。
