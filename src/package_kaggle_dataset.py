"""
package_kaggle_dataset.py
data/ 配下の生成済みデータを kaggle_dataset/ にコピーし、
`kaggle datasets create` / `kaggle datasets version` にそのまま渡せる形にする。

dataset-metadata.json が kaggle_dataset/ に無い場合は自動生成する
（gitで管理し忘れていたり、zip展開時に欠落しても、このスクリプトが必ず用意する）。
id は --slug 指定 > Kaggle認証情報からの自動検出（環境変数 KAGGLE_USERNAME、
または ~/.kaggle/kaggle.json）> プレースホルダ、の優先順で決定する。
kaggle CLIが使える環境（= 認証済み）なら通常は自動検出できるはずなので、
毎回 --slug を手入力する必要はない。

実行: python src/package_kaggle_dataset.py [--slug ユーザー名/データセット名]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REQUIRED_FILES = [
    "v001_processed.train.jsonl",
    "v001_processed.val.jsonl",
    "corpus.txt",
    "v001_preprocessing_log.json",
]
OPTIONAL_FILES = [
    "spm_16k.model",
    "spm_16k.vocab",
]

DATASET_SLUG_NAME = "axral-mini-ai-dataset"  # 既存のKaggle Datasetに合わせる（新規作成しない）
DEFAULT_SLUG = f"YOUR_KAGGLE_USERNAME/{DATASET_SLUG_NAME}"


def detect_kaggle_username() -> str | None:
    """kaggle CLIが認識しているユーザー名を検出する。
    Kaggle Notebook環境では認証情報の保存場所や方式が ~/.kaggle/kaggle.json とは
    限らない（KAGGLE_KEY はあるがKAGGLE_USERNAMEが無い、設定ディレクトリが
    KAGGLE_CONFIG_DIR で変更されている等）ため、自前でファイルを探すのではなく
    `kaggle config view` を実際に呼び出して kaggle CLI 自身の解決結果を使う。
    これが失敗する場合のみ、環境変数と標準パスのkaggle.jsonにフォールバックする。
    """
    # 方法1: kaggle CLI自身に聞く（最も信頼できる。認証方式の違いを自前で再実装しない）
    try:
        result = subprocess.run(
            ["kaggle", "config", "view"],
            capture_output=True, text=True, timeout=10,
        )
        for raw_line in result.stdout.splitlines():
            # 出力例: "- username: sekoia29"（先頭に "- " が付く）や "username: sekoia29"
            line = raw_line.strip().lstrip("-").strip()
            if line.lower().startswith("username"):
                candidate = line.split(":", 1)[-1].strip()
                if candidate and candidate.upper() != "NONE":
                    return candidate
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # 方法2: 環境変数（Kaggle Secrets経由で設定されるケース）
    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_user:
        return env_user

    # 方法3: 標準パスのkaggle.json（KAGGLE_CONFIG_DIRが未設定なら~/.kaggle/）
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    kaggle_json_path = Path(config_dir) / "kaggle.json"
    if kaggle_json_path.exists():
        try:
            data = json.loads(kaggle_json_path.read_text(encoding="utf-8"))
            username = data.get("username")
            if username:
                return username
        except (json.JSONDecodeError, OSError):
            pass

    return None


def ensure_metadata(out_dir: Path, slug: str | None) -> tuple[bool, str, str]:
    """dataset-metadata.json が無ければ作成する。既存の場合は上書きしない。
    id の決定優先順: 明示的な --slug > 自動検出したユーザー名 > プレースホルダ。
    戻り値: (今回新規作成したか, 実際に書かれているid, idの決定方法を示すラベル)
    """
    meta_path = out_dir / "dataset-metadata.json"

    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            return False, existing.get("id", "(idフィールドが読めません)"), "既存ファイル"
        except json.JSONDecodeError:
            pass  # 壊れている場合は下で作り直す

    if slug:
        resolved_id, source = slug, "--slug指定"
    else:
        detected_user = detect_kaggle_username()
        if detected_user:
            resolved_id, source = f"{detected_user}/{DATASET_SLUG_NAME}", "Kaggle認証情報から自動検出"
        else:
            resolved_id, source = DEFAULT_SLUG, "プレースホルダ（自動検出失敗）"

    payload = {
        "title": DATASET_SLUG_NAME,
        "id": resolved_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, resolved_id, source


def main():
    parser = argparse.ArgumentParser(description="Kaggle Dataset用にdata/をパッケージング")
    parser.add_argument("--slug", default=None,
                        help="dataset-metadata.jsonが未生成の場合に使うid（例: myname/vose-initial-llm-data-v001）。"
                             "省略時はプレースホルダを書き込むので後で手動編集すること。")
    args = parser.parse_args()

    data_dir = Path("data")
    out_dir = Path("kaggle_dataset")
    out_dir.mkdir(exist_ok=True)

    copied = []
    missing_required = []
    missing_optional = []

    for fname in REQUIRED_FILES:
        src = data_dir / fname
        if src.exists():
            dst = out_dir / fname
            shutil.copy(src, dst)
            copied.append((fname, dst.stat().st_size / (1024 * 1024)))
        else:
            missing_required.append(fname)

    for fname in OPTIONAL_FILES:
        src = data_dir / fname
        if src.exists():
            dst = out_dir / fname
            shutil.copy(src, dst)
            copied.append((fname, dst.stat().st_size / (1024 * 1024)))
        else:
            missing_optional.append(fname)

    # dataset-metadata.json は package_kaggle_dataset.py 自身が必ず用意する
    # （gitでの同期漏れやzip展開時の欠落に依存しない。idはKaggle認証情報から自動検出を試みる）
    meta_created, meta_id, meta_source = ensure_metadata(out_dir, args.slug)

    print("=== kaggle_dataset/ へのコピー結果 ===")
    for fname, size in copied:
        print(f"  ✓ {fname} ({size:.1f} MB)")
    for fname in missing_required:
        print(f"  ✗ {fname} が見つかりません（先に generate_data.py / preprocess.py を実行してください）")
    for fname in missing_optional:
        print(f"  - {fname} は未生成（任意）。train_tokenizer.py を実行すると同梱できます。")

    if meta_created:
        print(f"  ✓ dataset-metadata.json を新規作成しました（id: {meta_id} / 決定方法: {meta_source}）")
    else:
        print(f"  ✓ dataset-metadata.json は既存のものを使用（id: {meta_id}）")

    if meta_id == DEFAULT_SLUG:
        print(f"\n🛑 id がプレースホルダ（{DEFAULT_SLUG}）のままです。")
        print(f"   Kaggle認証情報（環境変数 KAGGLE_USERNAME、または ~/.kaggle/kaggle.json）からの")
        print(f"   自動検出にも失敗しました。このままアップロードすると403 Forbiddenになります。")
        print(f"\n   次のいずれかで修正してから再実行してください:")
        print(f"     a) python src/package_kaggle_dataset.py --slug あなたのKaggleユーザー名/{DATASET_SLUG_NAME}")
        print(f"     b) kaggle_dataset/dataset-metadata.json を直接編集し、\"id\" を書き換える")
        print(f"     c) kaggle CLIの認証設定を確認する（kaggle.json が ~/.kaggle/ に無い可能性があります）")
        print(f"\n   修正するまで kaggle datasets create/version は実行しないでください。")
        return

    if not missing_required:
        print("\n次のコマンドでKaggleにアップロードできます:")
        print("  kaggle datasets create -p kaggle_dataset/          # 初回（データセットがまだ存在しない場合）")
        print("  kaggle datasets version -p kaggle_dataset/ -m '更新内容'  # 2回目以降（既に存在する場合）")
        print("\n  ※ create と version を間違えると404/403エラーになります。")
        print("     初回かどうか分からない場合は、Kaggleの自分のDatasetsページで")
        print(f"     「{meta_id}」が既に存在するか確認してください。")


if __name__ == "__main__":
    main()
