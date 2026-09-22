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
EXPERIMENT_NAMES = {
    "v_cot20": "p1_cot20_gate3",
    "v_cot40": "p1_cot40_gate3",
    "v_cot60": "p1_cot60_gate3",
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
    ckpt_path = f"results/checkpoints/{EXPERIMENT_NAMES[version]}/checkpoint_best.pt"
    require_file(ckpt_path, f"学習（{version}）")

print("\n" + "=" * 60)
print("🎉 Gate3パイプライン、全ステップ完了")
print("=" * 60)

# ============================================================
# 5. checkpointとデータをKaggle Datasetに退避
# ============================================================
# /kaggle/working はセッションが切れると消える一時領域のため、
# セッション終了前に学習成果を必ずKaggle Datasetへアップロードしておく
# （過去に何度も「セッションが切れてcheckpointごと消えた」事故が起きたため）。
print("\n" + "=" * 60)
print("▶ checkpoint・データをKaggle Datasetへ退避")
print("=" * 60)

KAGGLE_DATASET_DIR = os.path.join(PROJECT_ROOT, "kaggle_dataset")
os.makedirs(KAGGLE_DATASET_DIR, exist_ok=True)

# 5-1. package_kaggle_dataset.py で data/ 配下の必須ファイル一式をコピーし、
#      dataset-metadata.json を自動生成する（既存の仕組みを流用）
try:
    run(
        [sys.executable, "src/package_kaggle_dataset.py"],
        "kaggle_dataset/ へのデータファイルコピー",
    )
except RuntimeError as e:
    print(f"⚠️ {e}")
    print("⚠️ データファイルのコピーに失敗しましたが、checkpointのコピーは続行します。")

# 5-2. 3実験分のcheckpointを、experiment_nameを含む名前でコピー
#      （同じ "checkpoint_best.pt" のまま3つコピーすると上書きされるため）
import shutil

copied_checkpoints = []
for version, _ in COT_VERSIONS:
    exp_name = EXPERIMENT_NAMES[version]
    src_ckpt = os.path.join(PROJECT_ROOT, f"results/checkpoints/{exp_name}/checkpoint_best.pt")
    if os.path.exists(src_ckpt):
        dst_name = f"checkpoint_best_{exp_name}.pt"
        dst_ckpt = os.path.join(KAGGLE_DATASET_DIR, dst_name)
        shutil.copy(src_ckpt, dst_ckpt)
        print(f"  ✓ {dst_name} ({os.path.getsize(dst_ckpt):,} bytes)")
        copied_checkpoints.append(dst_name)
    else:
        print(f"  ✗ {src_ckpt} が見つかりません（このcheckpointは退避されません）")

# 5-3. dataset-metadata.json の id がプレースホルダのままでないか確認してからアップロード
metadata_path = os.path.join(KAGGLE_DATASET_DIR, "dataset-metadata.json")
should_upload = False
if os.path.exists(metadata_path):
    import json as _json
    with open(metadata_path, encoding="utf-8") as f:
        meta = _json.load(f)
    dataset_id = meta.get("id", "")
    if "YOUR_KAGGLE_USERNAME" in dataset_id:
        print(f"\n🛑 dataset-metadata.json の id がプレースホルダのままです（{dataset_id}）。")
        print("   アップロードをスキップします。checkpointは kaggle_dataset/ に")
        print("   コピー済みなので、id を修正してから手動で以下を実行してください:")
        print("     kaggle datasets version -p kaggle_dataset/ -m '更新内容'")
    else:
        should_upload = True
        print(f"\n  dataset id: {dataset_id}")
else:
    print("\n⚠️ dataset-metadata.json が見つからないため、アップロードをスキップします。")

if should_upload and copied_checkpoints:
    result = subprocess.run(
        [
            "kaggle", "datasets", "version",
            "-p", KAGGLE_DATASET_DIR,
            "-m", "Gate3実験（cot20/40/60）のcheckpoint・データを更新",
        ],
        cwd=PROJECT_ROOT,
    )
    if result.returncode == 0:
        print("\n✅ Kaggle Datasetへのアップロード完了")
    else:
        print(f"\n⚠️ アップロードに失敗しました（終了コード {result.returncode}）。")
        print("   ただし学習・checkpointの保存自体は正常に完了しています。")
        print("   kaggle_dataset/ の中身を確認し、手動で再アップロードしてください。")
elif not copied_checkpoints:
    print("\n⚠️ コピーできたcheckpointが1つも無いため、アップロードをスキップしました。")

print("\n次に sanity_check.py で3つのcheckpointを比較してください。")
print("（今後セッションが切れても、Kaggle Dataset側からcheckpointを復元できます）")
