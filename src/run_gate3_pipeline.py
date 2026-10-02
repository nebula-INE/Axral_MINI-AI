"""
run_gate3_pipeline.py
テンプレートデータだけでCoT比率ごとの学習を行うパイプライン（データ生成→トークナイザー→前処理→学習→退避）。

使い方（Kaggleノートブックのセルで）:
  # 既定: cot60 の1本だけ（GPU時間を節約。Gate3で最良だった比率）
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_gate3_pipeline

  # 比率を指定（複数ならカンマ区切り。従来の3本比較なら cot20,cot40,cot60）
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_gate3_pipeline --versions cot40,cot60

  # 学習の手前（前処理まで）で止めて確認する
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_gate3_pipeline --skip_train

※ 公開データ（dolly/oasst）を混ぜた新しい学習は、このスクリプトではなく
   src/run_mix_pipeline.py を使う（語彙16,000・混合データ用）。

設計（これまでの事故を踏まえた方針）:
  - 冒頭でプロジェクトルートへ強制移動する（ディレクトリのズレ対策）。
  - 各ステップの直後に生成物の存在確認を入れ、無ければ即停止する。
  - checkpoint とトークナイザー・configは必ずセットで退避する
    （片方だけ消えると推論できなくなるため）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

COT_RATIOS = {"cot20": 0.2, "cot40": 0.4, "cot60": 0.6}
TOKENIZER_PREFIX = "spm_16k_v4"


def exp_name(version: str) -> str:
    return f"p1_{version}_gate3_7m"


def run(cmd: list[str], step: str, cwd: str, fatal: bool = True) -> bool:
    print(f"\n{'=' * 60}\n▶ {step}\n{'=' * 60}")
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        msg = f"❌ ステップ「{step}」が失敗しました（終了コード {result.returncode}）。"
        if fatal:
            raise RuntimeError(msg + "ここで停止します。上のログを確認してください。")
        print("⚠️ " + msg + "（このステップは失敗しても続行します）")
        return False
    print(f"✓ {step} 完了")
    return True


def require_file(root: str, relpath: str, step: str) -> None:
    full = os.path.join(root, relpath)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"❌ 「{step}」の後に期待されるファイルが見つかりません: {relpath}\n"
            f"このステップが実際には失敗していた可能性があります。上のログを確認してください。"
        )
    print(f"  ✓ 確認: {relpath} ({os.path.getsize(full):,} bytes)")


def parse_versions(text: str) -> list[str]:
    versions = [v.strip() for v in text.split(",") if v.strip()]
    unknown = [v for v in versions if v not in COT_RATIOS]
    if not versions or unknown:
        raise SystemExit(
            f"--versions は {', '.join(COT_RATIOS)} をカンマ区切りで指定してください（指定値: {text}）"
        )
    return versions


def backup(root: str, versions: list[str], tok_model: str) -> None:
    """checkpoint・トークナイザー・configをセットで kaggle_dataset/ に退避し、可能ならアップロードする。"""
    out_dir = os.path.join(root, "kaggle_dataset")
    os.makedirs(out_dir, exist_ok=True)

    # 既存の仕組みでデータ一式をコピー（失敗しても続行）
    run([sys.executable, "src/package_kaggle_dataset.py"], "kaggle_dataset/ へのデータコピー", root, fatal=False)

    copied = []
    for v in versions:
        name = exp_name(v)
        src = os.path.join(root, f"results/checkpoints/{name}/checkpoint_best.pt")
        if os.path.exists(src):
            dst = os.path.join(out_dir, f"checkpoint_best_{name}.pt")
            shutil.copy(src, dst)
            copied.append(os.path.basename(dst))
            print(f"  ✓ {os.path.basename(dst)} ({os.path.getsize(dst):,} bytes)")
        else:
            print(f"  ✗ {src} が見つかりません（退避されません）")

    # トークナイザーとconfigもセットで退避（checkpointだけでは推論できない）
    for rel in (tok_model, tok_model.replace(".model", ".vocab"),
                *[f"configs/exp_{v}.yaml" for v in versions]):
        src = os.path.join(root, rel)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, os.path.basename(rel)))
            print(f"  ✓ {os.path.basename(rel)}")

    meta_path = os.path.join(out_dir, "dataset-metadata.json")
    if not (os.path.exists(meta_path) and copied):
        print("\n⚠️ dataset-metadata.json が無い、またはcheckpointが無いため、自動アップロードはスキップします。")
        print(f"   {out_dir} の中身を、Notebookの Save Version → Output から Dataset として保存してください。")
        return
    with open(meta_path, encoding="utf-8") as f:
        dataset_id = json.load(f).get("id", "")
    if "YOUR_KAGGLE_USERNAME" in dataset_id:
        print(f"\n🛑 dataset-metadata.json の id がプレースホルダのままです（{dataset_id}）。自動アップロードはスキップします。")
        print(f"   id を修正して手動で: kaggle datasets version -p {out_dir} -m '更新内容'")
        return
    result = subprocess.run(
        ["kaggle", "datasets", "version", "-p", out_dir, "-m", f"{', '.join(versions)} のcheckpoint・トークナイザーを更新"],
        cwd=root,
    )
    print("\n✅ Kaggle Datasetへのアップロード完了" if result.returncode == 0
          else f"\n⚠️ アップロードに失敗しました（終了コード {result.returncode}）。{out_dir} から手動で再アップロードしてください。")


def main():
    parser = argparse.ArgumentParser(description="テンプレートデータでのCoT比率別学習パイプライン")
    parser.add_argument("--project_root", default="/kaggle/working/Axral_MINI-AI")
    parser.add_argument("--versions", default="cot60", help="例: cot60 / cot40,cot60 / cot20,cot40,cot60")
    parser.add_argument("--num_samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=4000,
                        help="configs/exp_cot*.yaml の data.vocab_size と一致させること")
    parser.add_argument("--skip_train", action="store_true", help="学習の手前（前処理まで）で止める")
    args = parser.parse_args()

    versions = parse_versions(args.versions)
    root = args.project_root
    os.chdir(root)
    assert os.path.exists("src") and os.path.exists("configs"), f"プロジェクトルートが見つかりません: {root}"
    print(f"現在地: {os.getcwd()}\n対象: {versions}")
    py = sys.executable
    tok_model = f"data/{TOKENIZER_PREFIX}.model"

    # 1. データ生成
    for v in versions:
        run([py, "src/generate_data.py", "--output_dir", "data/", "--num_samples", str(args.num_samples),
             "--seed", str(args.seed), "--cot_ratio", str(COT_RATIOS[v]), "--version", f"v_{v}"],
            f"データ生成（{v}, cot_ratio={COT_RATIOS[v]}）", root)
        for suffix in (".train.jsonl", ".val.jsonl", "_corpus.txt"):
            require_file(root, f"data/v_{v}{suffix}", f"データ生成（{v}）")

    # 2. トークナイザー（選んだ先頭バージョンのコーパスで学習し、全バージョンで共有）
    run([py, "src/train_tokenizer.py", "--corpus", f"data/v_{versions[0]}_corpus.txt",
         "--output_dir", "data/", "--vocab_size", str(args.vocab_size),
         "--model_prefix", TOKENIZER_PREFIX, "--character_coverage", "0.9999"],
        "トークナイザー学習", root)
    require_file(root, tok_model, "トークナイザー学習")

    # 3. 前処理
    for v in versions:
        run([py, "-m", "src.preprocess", "--train_path", f"data/v_{v}.train.jsonl",
             "--val_path", f"data/v_{v}.val.jsonl", "--output_dir", "data/", "--tokenizer_path", tok_model],
            f"前処理（{v}）", root)
        require_file(root, f"data/v_{v}_processed.train.jsonl", f"前処理（{v}）")
        require_file(root, f"data/v_{v}_processed.val.jsonl", f"前処理（{v}）")

    if args.skip_train:
        print("\n--skip_train 指定のため、学習の手前で終了します。")
        return

    # 4. 学習
    for v in versions:
        run([py, "-m", "src.train", "--config", f"configs/exp_{v}.yaml"], f"学習（{v}）", root)
        require_file(root, f"results/checkpoints/{exp_name(v)}/checkpoint_best.pt", f"学習（{v}）")

    # 5. 退避（checkpoint・トークナイザー・configをセットで）
    print(f"\n{'=' * 60}\n▶ checkpoint・トークナイザー・configをKaggle Datasetへ退避\n{'=' * 60}")
    backup(root, versions, tok_model)

    # 6. sanity_check（失敗しても続行）
    for v in versions:
        run([py, "-m", "src.sanity_check", "--config", f"configs/exp_{v}.yaml",
             "--checkpoint", f"results/checkpoints/{exp_name(v)}/checkpoint_best.pt",
             "--val_path", f"data/v_{v}.val.jsonl", "--num_samples", "50"],
            f"sanity_check（{v}）", root, fatal=False)

    print(f"\n{'=' * 60}\n🎉 パイプライン完了: {versions}\n{'=' * 60}")


if __name__ == "__main__":
    main()
