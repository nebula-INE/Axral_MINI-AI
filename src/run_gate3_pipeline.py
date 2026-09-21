"""
run_gate3_pipeline.py
Gate3実験（CoT比率20/40/60のA/B実験）を、Kaggle Notebookの1セルで
最初から最後まで通しで実行するためのオールインワンスクリプト。

これまで繰り返し発生した事故を踏まえた設計:
  - 冒頭で必ずプロジェクトルートに強制移動する（ディレクトリのズレ対策）
  - 各ステップの直後に必ず生成物の存在確認を入れ、無ければ即座に停止する
    （前のコマンドが失敗したまま気づかず次に進む事故の防止）
  - コマンドの実行順序を1つのスクリプトに固定し、コピペミスや
    コマンド順の取り違えが起きないようにする

Kaggle Notebookでの使い方:
  1. このファイルの中身を丸ごと1つのセルに貼り付けて実行する
     （%%writefile 等は不要。そのままコード全体を貼り付けるだけ）
  2. 何かのステップで失敗したら、そのエラーメッセージを見れば
     どの段階で止まったか一目で分かる
"""
import os
import subprocess
import sys

# ============================================================
# 設定（必要に応じてここだけ書き換える）
# ============================================================
PROJECT_ROOT = "/kaggle/working/Axral_MINI-AI"
NUM_SAMPLES = 30000
SEED = 42
VOCAB_SIZE = 4000
TOKENIZER_PREFIX = "spm_16k_v4"
COT_VERSIONS = [("v_cot20", 0.2), ("v_cot40", 0.4), ("v_cot60", 0.6)]
CONFIGS = {
    "v_cot20": "configs/exp_cot20.yaml",
    "v_cot40": "configs/exp_cot40.yaml",
    "v_cot60": "configs/exp_cot60.yaml",
}


def run(cmd: list[str], step_name: str) -> None:
    """コマンドを実行し、失敗したら即座に例外を出して停止する。"""
    print(f"\n{'=' * 60}\n▶ {step_name}\n{'=' * 60}")
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"❌ ステップ「{step_name}」が失敗しました（終了コード {result.returncode}）。"
            f"ここで停止します。上のログを確認してください。"
        )
    print(f"✓ {step_name} 完了")


def require_file(relpath: str, step_name: str) -> None:
    """ファイルが存在しなければ即座に停止する。"""
    full_path = os.path.join(PROJECT_ROOT, relpath)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"❌ 「{step_name}」の後に期待されるファイルが見つかりません: {relpath}\n"
            f"このステップが実際には失敗していた可能性があります。上のログを確認してください。"
        )
    size = os.path.getsize(full_path)
    print(f"  ✓ 確認: {relpath} ({size:,} bytes)")


# ============================================================
# 0. プロジェクトルートへ移動（最重要: ここがズレると全部失敗する）
# ============================================================
os.chdir(PROJECT_ROOT)
print(f"現在地: {os.getcwd()}")
assert os.path.exists("src") and os.path.exists("configs"), (
    f"❌ プロジェクトルートが見つかりません: {PROJECT_ROOT}\n"
    f"PROJECT_ROOT変数を実際のパスに書き換えてから再実行してください。"
)
print("✓ プロジェクトルート確認OK")

# ============================================================
# 1. データ生成（cot_ratio 20% / 40% / 60% の3種類）
# ============================================================
for version, cot_ratio in COT_VERSIONS:
    run(
        [
            sys.executable, "src/generate_data.py",
            "--output_dir", "data/",
            "--num_samples", str(NUM_SAMPLES),
            "--seed", str(SEED),
            "--cot_ratio", str(cot_ratio),
            "--version", version,
        ],
        f"データ生成（{version}, cot_ratio={cot_ratio}）",
    )
    require_file(f"data/{version}.train.jsonl", f"データ生成（{version}）")
    require_file(f"data/{version}.val.jsonl", f"データ生成（{version}）")
    require_file(f"data/{version}_corpus.txt", f"データ生成（{version}）")

# ============================================================
# 2. トークナイザー学習（3実験で共通利用、v_cot20のコーパスを使う）
# ============================================================
tokenizer_model = f"data/{TOKENIZER_PREFIX}.model"
run(
    [
        sys.executable, "src/train_tokenizer.py",
        "--corpus", "data/v_cot20_corpus.txt",
        "--output_dir", "data/",
        "--vocab_size", str(VOCAB_SIZE),
        "--model_prefix", TOKENIZER_PREFIX,
        "--character_coverage", "0.9999",
    ],
    "トークナイザー学習",
)
require_file(tokenizer_model, "トークナイザー学習")

# ============================================================
# 3. 前処理（3種類とも共通トークナイザーでtoken_ids化）
# ============================================================
for version, _ in COT_VERSIONS:
    run(
        [
            sys.executable, "-m", "src.preprocess",
            "--train_path", f"data/{version}.train.jsonl",
            "--val_path", f"data/{version}.val.jsonl",
            "--output_dir", "data/",
            "--tokenizer_path", tokenizer_model,
        ],
        f"前処理（{version}）",
    )
    require_file(f"data/{version}_processed.train.jsonl", f"前処理（{version}）")
    require_file(f"data/{version}_processed.val.jsonl", f"前処理（{version}）")

# ============================================================
# 4. 学習（3種類）
# ============================================================
for version, _ in COT_VERSIONS:
    config_path = CONFIGS[version]
    run(
        [sys.executable, "-m", "src.train", "--config", config_path],
        f"学習（{version}, config={config_path}）",
    )

print("\n" + "=" * 60)
print("🎉 Gate3パイプライン、全ステップ完了")
print("=" * 60)
print("\n次に sanity_check.py で3つのcheckpointを比較してください。")
