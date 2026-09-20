# Gate 3: CoT比率A/B実験（cot20 vs cot40 vs cot60）

plan §10 Gate 3 に対応。CoT（思考過程）をどの割合のサンプルに付与すると
数学タスク等の正答率が最も伸びるかを比較する。

## 前提

- トークナイザーは3実験で共通（`data/spm_16k_v4.model`）。全CoTテキストを
  含むコーパスで学習済みのため、CoT比率を変えたデータでもそのまま使える
  （再学習不要）。
- 既存の `docs/BASELINE_COMPLETE.md` の結果は実際には cot_ratio=1.0（全件CoT付き）
  で学習されたものであり、"p1_cot20" という実験名は当時ラベルのみでCoT比率制御が
  未実装だった。本実験で初めて実際のCoT比率制御を導入する。

## 手順

```bash
# 1. 3つの比率でデータを生成（同じseedで比較可能にする）
python src/generate_data.py --output_dir data/ --num_samples 30000 --seed 42 --cot_ratio 0.2 --version v_cot20
python src/generate_data.py --output_dir data/ --num_samples 30000 --seed 42 --cot_ratio 0.4 --version v_cot40
python src/generate_data.py --output_dir data/ --num_samples 30000 --seed 42 --cot_ratio 0.6 --version v_cot60

# 2. それぞれ前処理（共通トークナイザーを使用）
python -m src.preprocess --train_path data/v_cot20.train.jsonl --val_path data/v_cot20.val.jsonl --output_dir data/ --tokenizer_path data/spm_16k_v4.model
python -m src.preprocess --train_path data/v_cot40.train.jsonl --val_path data/v_cot40.val.jsonl --output_dir data/ --tokenizer_path data/spm_16k_v4.model
python -m src.preprocess --train_path data/v_cot60.train.jsonl --val_path data/v_cot60.val.jsonl --output_dir data/ --tokenizer_path data/spm_16k_v4.model

# 3. 3つとも学習（checkpointは experiment_name ごとに自動的に別ディレクトリになる）
python -m src.train --config configs/exp_cot20.yaml
python -m src.train --config configs/exp_cot40.yaml
python -m src.train --config configs/exp_cot60.yaml
```

## 比較方法

各実験終了後、以下を記録・比較する（plan §10 Gate 3 の判定基準）。

| 実験 | Val Loss（最終） | math_accuracy | qa_em | qa_f1 |
|------|------------------|----------------|-------|-------|
| cot20 | | | | |
| cot40 | | | | |
| cot60 | | | | |

- `sanity_check.py` を各checkpointに対して実行し、`arithmetic`カテゴリの
  正答率（math_accuracy相当）を比較するのが手早い:
  ```bash
  python -m src.sanity_check --config configs/exp_cot20.yaml --checkpoint results/checkpoints/p1_cot20_gate3/checkpoint_best.pt --val_path data/v_cot20.val.jsonl --num_samples 50
  # cot40, cot60 も同様（configとcheckpointとval_pathをそれぞれ対応させる）
  ```
- 最もmath_accuracyが高い（かつVal Lossも大きく崩れていない）比率を
  「最適なCoT比率」として採用する。plan §1の成功基準は
  「数学タスク正答率がベースライン比+5%以上」。
