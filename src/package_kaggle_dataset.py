"""
package_kaggle_dataset.py
data/ 配下の生成済みデータを kaggle_dataset/ にコピーし、
`kaggle datasets create` / `kaggle datasets version` にそのまま渡せる形にする。

実行: python src/package_kaggle_dataset.py
"""
from __future__ import annotations

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


def main():
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

    print("=== kaggle_dataset/ へのコピー結果 ===")
    for fname, size in copied:
        print(f"  ✓ {fname} ({size:.1f} MB)")
    for fname in missing_required:
        print(f"  ✗ {fname} が見つかりません（先に generate_data.py / preprocess.py を実行してください）")
    for fname in missing_optional:
        print(f"  - {fname} は未生成（任意）。train_tokenizer.py を実行すると同梱できます。")

    if not missing_required:
        print("\n次のコマンドでKaggleにアップロードできます:")
        print("  kaggle datasets create -p kaggle_dataset/          # 初回")
        print("  kaggle datasets version -p kaggle_dataset/ -m '更新内容'  # 2回目以降")
        print("\n(事前に kaggle_dataset/dataset-metadata.json の id を書き換えておくこと)")


if __name__ == "__main__":
    main()
