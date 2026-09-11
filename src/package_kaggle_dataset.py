"""
package_kaggle_dataset.py
data/ 配下の生成済みデータを kaggle_dataset/ にコピーし、
`kaggle datasets create` / `kaggle datasets version` にそのまま渡せる形にする。

dataset-metadata.json が kaggle_dataset/ に無い場合は自動生成する
（gitで管理し忘れていたり、zip展開時に欠落しても、このスクリプトが必ず用意する）。

実行: python src/package_kaggle_dataset.py [--slug ユーザー名/データセット名]
"""
from __future__ import annotations

import argparse
import json
import shutil
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

DEFAULT_SLUG = "YOUR_KAGGLE_USERNAME/vose-initial-llm-data-v001"


def ensure_metadata(out_dir: Path, slug: str | None) -> tuple[bool, str]:
    """dataset-metadata.json が無ければ作成する。既存の場合は上書きしない。
    戻り値: (今回新規作成したか, 実際に書かれているid)
    """
    meta_path = out_dir / "dataset-metadata.json"

    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            return False, existing.get("id", "(idフィールドが読めません)")
        except json.JSONDecodeError:
            pass  # 壊れている場合は下で作り直す

    payload = {
        "title": "vose-initial-llm-data-v001",
        "id": slug or DEFAULT_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, payload["id"]


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
    # （gitでの同期漏れやzip展開時の欠落に依存しない）
    meta_created, meta_id = ensure_metadata(out_dir, args.slug)

    print("=== kaggle_dataset/ へのコピー結果 ===")
    for fname, size in copied:
        print(f"  ✓ {fname} ({size:.1f} MB)")
    for fname in missing_required:
        print(f"  ✗ {fname} が見つかりません（先に generate_data.py / preprocess.py を実行してください）")
    for fname in missing_optional:
        print(f"  - {fname} は未生成（任意）。train_tokenizer.py を実行すると同梱できます。")

    if meta_created:
        print(f"  ✓ dataset-metadata.json を新規作成しました（id: {meta_id}）")
    else:
        print(f"  ✓ dataset-metadata.json は既存のものを使用（id: {meta_id}）")

    if meta_id == DEFAULT_SLUG:
        print(f"\n⚠️  id がプレースホルダのままです。アップロード前に必ず修正してください:")
        print(f"     kaggle_dataset/dataset-metadata.json の \"id\" を")
        print(f"     \"あなたのKaggleユーザー名/vose-initial-llm-data-v001\" に書き換える")
        print(f"     （または次回このスクリプトを --slug ユーザー名/データセット名 付きで実行する）")

    if not missing_required:
        print("\n次のコマンドでKaggleにアップロードできます:")
        print("  kaggle datasets create -p kaggle_dataset/          # 初回")
        print("  kaggle datasets version -p kaggle_dataset/ -m '更新内容'  # 2回目以降")


if __name__ == "__main__":
    main()
