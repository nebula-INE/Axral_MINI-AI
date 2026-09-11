# Kaggle Datasets によるデータ管理

`data/v001_processed.train.jsonl`（50MB超）をGitHubにコミットすると
push制限（単一ファイル100MB、リポジトリ全体の推奨1GB）に抵触しやすく、
差分もテキストながら巨大jsonlのため履歴が肥大化する。そのため、
**コードはGitHub、データはKaggle Datasets** に分離する。

## 方針

| 種別 | 保存場所 | 理由 |
|------|---------|------|
| コード（`src/`, `configs/`, `tests/`） | GitHub | 差分管理・レビューが必要 |
| 生データ・処理済みデータ（`data/*.jsonl`, `corpus.txt`） | Kaggle Datasets | 再生成可能・サイズが大きい |
| データ生成スクリプト（`generate_data.py`, `preprocess.py`） | GitHub | データそのものではなく手順を管理 |
| `sources.csv`（出典一覧、数KB） | GitHub | サイズが小さく差分追跡の価値がある |

`.gitignore` で `data/*.jsonl` 等を除外済み。ローカル/Kaggle Notebookで
`generate_data.py` → `preprocess.py` を実行すればいつでも同じデータを再現できる
（`--seed 42` で決定的に再現可能）。

## 手順1: データセットの作成（初回のみ）

### 方法A: Kaggle Web UIから（推奨・簡単）

1. ローカルでデータを生成する
   ```bash
   python src/generate_data.py --output_dir data/ --num_samples 30000 --seed 42
   python -m src.preprocess --train_path data/v001.train.jsonl --val_path data/v001.val.jsonl --output_dir data/
   ```
2. [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
3. 以下のファイルをアップロード:
   - `data/v001_processed.train.jsonl`
   - `data/v001_processed.val.jsonl`
   - `data/corpus.txt`
   - `data/v001_preprocessing_log.json`
4. タイトルを `vose-initial-llm-data-v001` などに設定して **Create**

### 方法B: Kaggle API から（自動化したい場合）

```bash
pip install kaggle
# ~/.kaggle/kaggle.json に API トークンを配置しておくこと

# 1. アップロード用フォルダにデータをコピー
mkdir -p kaggle_dataset
cp data/v001_processed.train.jsonl kaggle_dataset/
cp data/v001_processed.val.jsonl kaggle_dataset/
cp data/corpus.txt kaggle_dataset/
cp data/v001_preprocessing_log.json kaggle_dataset/

# 2. kaggle_dataset/dataset-metadata.json の "id" を
#    "あなたのKaggleユーザー名/vose-initial-llm-data-v001" に書き換える

# 3. 新規作成（初回）
kaggle datasets create -p kaggle_dataset/

# 4. 更新（2回目以降、データを再生成した時）
kaggle datasets version -p kaggle_dataset/ -m "v2: CoTテンプレート改善、採用率97.6%"
```

## 手順2: Kaggle Notebookでの利用

Notebook編集画面 → 右側パネル **Add Input** → 作成したデータセットを検索して追加すると、
`/kaggle/input/vose-initial-llm-data-v001/` にファイルが展開される。

```python
import shutil, os

KAGGLE_DATA_DIR = "/kaggle/input/vose-initial-llm-data-v001"
LOCAL_DATA_DIR = "data"

os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
for fname in ["v001_processed.train.jsonl", "v001_processed.val.jsonl", "corpus.txt"]:
    src = f"{KAGGLE_DATA_DIR}/{fname}"
    dst = f"{LOCAL_DATA_DIR}/{fname}"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)  # または configs/exp01.yaml のパスを直接 /kaggle/input/... に向けてもよい
```

`configs/exp01.yaml` の `data.train_path` / `data.val_path` は、Kaggle Notebook上では
直接 `/kaggle/input/vose-initial-llm-data-v001/v001_processed.train.jsonl` に
書き換えるのが最も簡単（`/kaggle/input` は読み取り専用なのでコピー不要）。

```yaml
data:
  train_path: /kaggle/input/vose-initial-llm-data-v001/v001_processed.train.jsonl
  val_path: /kaggle/input/vose-initial-llm-data-v001/v001_processed.val.jsonl
```

`configs/exp01_kaggle.yaml` として別ファイルを用意し、ローカル実験用の
`configs/exp01.yaml`（`data/...` 相対パス）とは分けておくと、
GitHubにpushしても環境依存パスが混ざらない。

## バージョニング方針

データを再生成するたび（CoTテンプレート改善、サンプル数変更など）に
Kaggle Dataset側もバージョンを切る。

```bash
kaggle datasets version -p kaggle_dataset/ -m "変更内容を一言で（例: v3: サンプル数を5万件に増加）"
```

`data/v001_preprocessing_log.json` にタイムスタンプとGate1採用率が記録されるので、
このファイルも毎回同梱し、どのデータセットバージョンがどのGate1結果だったか
後から追跡できるようにする。

## なぜコード側に手順だけ残すのか

- `generate_data.py --seed 42` は決定的なので、コードさえあれば
  誰でも同じデータを再現できる（データそのものをコミットする必要がない）
- CoT評価ロジックが変わった場合（`eval_cot.py`のバグ修正など）は
  `preprocess.py` を再実行するだけで新しいスコアが付与されたデータを再生成できる
- リポジトリが軽量に保たれ、`git clone` や CI が高速になる
