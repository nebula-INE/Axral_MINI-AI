"""
import_pretrain_data.py
p7（2段階学習）用のデータを取得・整形する。

  事前学習用（普通の文章とコードを大量に読ませる）
    - 日本語Wikipedia
    - CodeSearchNet（Pythonの関数）
  指示学習用（既存の hf_import と同じ形式 = mix_data.py がそのまま読める）
    - llm-japanese-dataset-vanilla（日本語の指示データ）
    - コード指示データ（code_instructions系）

実行例（Kaggleのセルで。ネットワーク必須）:
  !cd /kaggle/working/Axral_MINI-AI && python -m src.import_pretrain_data

小さく動作確認だけしたいとき:
  !cd /kaggle/working/Axral_MINI-AI && python -m src.import_pretrain_data --wiki_max_docs 2000 --code_max_docs 2000 \\
      --sft_max_vanilla 2000 --sft_max_code 2000

出力:
  {out_dir}/pretrain.train.jsonl, pretrain.val.jsonl     {"src": "wiki"|"code", "text": ...}
  {sft_dir}/hf_import.train.jsonl, hf_import.val.jsonl, hf_corpus.txt   指示データ（物差し・調整用との重複を除去済み）
  data/sources.csv に出典とライセンスの記録を追記

注意（実データでの最初の実行はKaggle上で確認してください）:
  - データセット名・列名は、作成時にネットワークが無く実物を確認できていません。名前が違う/列名が違う場合は、
    読み込めなかった旨と列名を表示して、そのデータだけ飛ばします。--*_dataset / --*_local で差し替えられます。
  - ライセンスは、使う前に各データセットのページで確認してください。
    日本語Wikipedia: CC BY-SA（帰属表示・同条件継承）。CodeSearchNet: 元リポジトリごとにライセンスが異なる。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- 既定のデータ源（実物は未確認。候補を順に試す）
WIKI_CANDIDATES = [("wikimedia/wikipedia", "20231101.ja")]
CODE_CANDIDATES = [("Nan-Do/code-search-net-python", None), ("code-search-net/code_search_net", "python")]
VANILLA_CANDIDATES = [("izumi-lab/llm-japanese-dataset-vanilla", None)]
CODE_INSTR_CANDIDATES = [("iamtarun/python_code_instructions_18k_alpaca", None),
                         ("TokenBender/code_instructions_122k_alpaca_style", None)]

CODE_FIELDS = ("code", "func_code_string", "whole_func_string", "original_string", "content")
_URL_RE = re.compile(r"https?://")
_BLANKS_RE = re.compile(r"\n{3,}")


# --------------------------------------------------------------------------- 整形（純粋な関数。テスト対象）
def clean_wiki(text: str, min_chars: int = 200) -> str | None:
    """Wikipediaの記事本文を整える。短すぎる記事は捨てる。"""
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"=+\s*.*?\s*=+", ln)]  # 見出しのマークアップ行
    out = "\n".join(lines).strip()
    return out if len(out) >= min_chars else None


def clean_code(code: str, min_chars: int = 60, max_chars: int = 4000) -> str | None:
    """Pythonコードを整える（行末の空白を除き、長すぎる/短すぎるものを捨てる）。"""
    if not code:
        return None
    lines = [ln.rstrip() for ln in str(code).replace("\r\n", "\n").replace("\t", "    ").split("\n")]
    out = _BLANKS_RE.sub("\n\n", "\n".join(lines)).strip("\n")
    if not (min_chars <= len(out) <= max_chars):
        return None
    return out


def pick_field(row: dict, candidates=CODE_FIELDS) -> str | None:
    for k in candidates:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def is_python_row(row: dict) -> bool:
    lang = row.get("language") or row.get("lang")
    return lang is None or str(lang).lower() == "python"


def split_bucket(text: str, val_permille: int = 3) -> str:
    """文書の内容のハッシュで train/val を決める（毎回同じ結果になる）。"""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 1000
    return "val" if h < val_permille else "train"


def pair_from_alpaca(row: dict) -> tuple[str, str] | None:
    """instruction / input / output 形式の行から (質問, 回答) を作る。"""
    instr = str(row.get("instruction") or "").strip()
    extra = str(row.get("input") or "").strip()
    out = str(row.get("output") or row.get("response") or "").strip()
    if not instr or not out:
        return None
    q = instr + ("\n" + extra if extra else "")
    return q, out


def keep_pair(q: str, a: str, max_q: int, max_a: int) -> bool:
    if not q or not a or len(q) > max_q or len(a) > max_a:
        return False
    return not (_URL_RE.search(q) or _URL_RE.search(a))


# --------------------------------------------------------------------------- 物差し・調整用との重複除去
def _heldout_texts(yardstick_path: str) -> list[str]:
    texts: list[str] = []
    if os.path.exists(yardstick_path):
        with open(yardstick_path, encoding="utf-8") as f:
            texts += [json.loads(line)["input"] for line in f if line.strip()]
    try:
        from src.tune_guard import NEG_FAR, NEG_NEAR, NEG_SHORT
        texts += list(NEG_NEAR) + list(NEG_FAR) + list(NEG_SHORT)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 調整用データを読めませんでした（{type(e).__name__}）。物差しだけで重複を検査します")
    return texts


class LeakFilter:
    """文字2-gramのJaccard類似度が閾値以上、または正規化後に同一なら「漏れ」とみなす。
    大量のデータを速く検査するため、2-gramの転置インデックスで候補を絞ってから厳密に計算する。"""

    def __init__(self, heldout: list[str], threshold: float = 0.70):
        from src.paraphrase_bank import _bigrams, _norm
        self._bigrams, self._norm, self.threshold = _bigrams, _norm, threshold
        self.heldout_norm = [self._norm(t) for t in heldout]
        self.exact = set(self.heldout_norm)
        self.sets = [self._bigrams(n) for n in self.heldout_norm]
        self.index: dict[str, list[int]] = {}
        for i, s in enumerate(self.sets):
            for g in s:
                self.index.setdefault(g, []).append(i)

    def is_leak(self, text: str) -> bool:
        n = self._norm(text)
        if n in self.exact:
            return True
        b = self._bigrams(n)
        counts: Counter = Counter()
        for g in b:
            for i in self.index.get(g, ()):
                counts[i] += 1
        for i, shared in counts.items():
            # Jaccard >= t なら shared >= t * max(|A|,|B|) >= 0.5 * min(|A|,|B|)（この条件で候補を絞る）
            if shared < 0.5 * min(len(b), len(self.sets[i])):
                continue
            union = len(b) + len(self.sets[i]) - shared
            if shared / union >= self.threshold:
                return True
        return False


# --------------------------------------------------------------------------- データの読み込み
def _open_stream(candidates, local: str | None, name: str):
    """(行のイテレータ, 使った名前) を返す。読めなければ (None, None)。"""
    from datasets import load_dataset  # 遅延import（整形関数だけテストできるように）

    if local:
        ext = Path(local).suffix.lower()
        kind = "parquet" if ext == ".parquet" else "json"
        try:
            return load_dataset(kind, data_files=local, split="train", streaming=True), f"local:{local}"
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: ローカルファイルを読めませんでした ({type(e).__name__}: {e})")
            return None, None
    for repo, config in candidates:
        try:
            ds = (load_dataset(repo, config, split="train", streaming=True) if config
                  else load_dataset(repo, split="train", streaming=True))
            return ds, repo + (f"/{config}" if config else "")
        except Exception as e:  # noqa: BLE001
            print(f"[試行失敗] {name}: {repo} ({type(e).__name__}: {str(e)[:160]})")
    print(f"[skip] {name}: どの候補も読み込めませんでした。--{name}_dataset か --{name}_local で指定してください")
    return None, None


def _write_pretrain(out_dir: Path, docs_iter, label: str, writers: dict, stats: Counter, max_docs: int,
                    max_chars: int):
    n = 0
    seen: set[str] = set()
    for text in docs_iter:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        bucket = split_bucket(text)
        writers[bucket].write(json.dumps({"src": label, "text": text}, ensure_ascii=False) + "\n")
        stats[f"{label}_{bucket}_docs"] += 1
        stats[f"{label}_{bucket}_chars"] += len(text)
        n += 1
        if max_docs and n >= max_docs:
            break
        if max_chars and stats[f"{label}_train_chars"] + stats[f"{label}_val_chars"] >= max_chars:
            break


def _record_sources(rows: list[dict]) -> None:
    path = Path("data/sources.csv")
    if not path.parent.exists():
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    new = not existing
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["source_id", "url", "license", "source_type", "category", "notes"])
        for r in rows:
            if f"{r['source_id']}," not in existing:
                w.writerow([r["source_id"], r["url"], r["license"], r["source_type"], r["category"], r["notes"]])


def main() -> None:
    ap = argparse.ArgumentParser(description="p7用: 事前学習データと指示データの取得・整形")
    ap.add_argument("--out_dir", default="/kaggle/working/p7_data")
    ap.add_argument("--sft_dir", default="/kaggle/working/hf_import_p7")
    ap.add_argument("--yardstick", default="eval_sets/yardstick_v1.jsonl")
    ap.add_argument("--include_existing_hf", default=None,
                    help="既存の hf_import ディレクトリ（dolly/oasst）。指定すると、指示データに加える")
    for name in ("wiki", "code", "vanilla", "code_instr"):
        ap.add_argument(f"--{name}_dataset", default=None, help=f"{name} のHFデータセット名（候補を差し替える）")
        ap.add_argument(f"--{name}_local", default=None, help=f"{name} のローカルファイル（jsonl/parquet）")
    ap.add_argument("--wiki_config", default=None)
    ap.add_argument("--wiki_max_docs", type=int, default=0)
    ap.add_argument("--wiki_max_chars", type=int, default=0, help="Wikipediaの合計文字数の上限（0なら全部）")
    ap.add_argument("--code_max_docs", type=int, default=0)
    ap.add_argument("--code_max_chars", type=int, default=0)
    ap.add_argument("--sft_max_vanilla", type=int, default=60000)
    ap.add_argument("--sft_max_code", type=int, default=20000)
    ap.add_argument("--vanilla_max_q", type=int, default=200)
    ap.add_argument("--vanilla_max_a", type=int, default=300)
    ap.add_argument("--code_max_q", type=int, default=300)
    ap.add_argument("--code_max_a", type=int, default=800)
    ap.add_argument("--leak_threshold", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_pretrain", action="store_true")
    ap.add_argument("--skip_sft", action="store_true")
    args = ap.parse_args()

    out_dir, sft_dir = Path(args.out_dir), Path(args.sft_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # ------------------------------------------------------------------ 事前学習用
    if not args.skip_pretrain:
        stats: Counter = Counter()
        writers = {b: (out_dir / f"pretrain.{b}.jsonl").open("w", encoding="utf-8") for b in ("train", "val")}
        try:
            wiki_cands = [(args.wiki_dataset, args.wiki_config)] if args.wiki_dataset else WIKI_CANDIDATES
            ds, used = _open_stream(wiki_cands, args.wiki_local, "wiki")
            if ds is not None:
                print(f"日本語Wikipedia: {used}")
                first_cols = []

                def wiki_docs():
                    for row in ds:
                        if not first_cols:
                            first_cols.append(list(row.keys()))
                        t = clean_wiki(row.get("text") or pick_field(row, ("content", "body")) or "")
                        if t:
                            yield t
                _write_pretrain(out_dir, wiki_docs(), "wiki", writers, stats, args.wiki_max_docs, args.wiki_max_chars)
                if not stats["wiki_train_docs"]:
                    print(f"  [注意] Wikipediaから1件も取れませんでした。列名: {first_cols}")
                records.append(dict(source_id="wiki_ja", url=f"hf:{used}", license="CC BY-SA (帰属表示・同条件継承)",
                                    source_type="public", category="pretrain", notes="日本語Wikipedia本文。事前学習用"))

            code_cands = [(args.code_dataset, None)] if args.code_dataset else CODE_CANDIDATES
            ds, used = _open_stream(code_cands, args.code_local, "code")
            if ds is not None:
                print(f"CodeSearchNet(Python): {used}")
                first_cols = []

                def code_docs():
                    for row in ds:
                        if not first_cols:
                            first_cols.append(list(row.keys()))
                        if not is_python_row(row):
                            continue
                        t = clean_code(pick_field(row) or "")
                        if t:
                            yield t
                _write_pretrain(out_dir, code_docs(), "code", writers, stats, args.code_max_docs, args.code_max_chars)
                if not stats["code_train_docs"]:
                    print(f"  [注意] コードから1件も取れませんでした。列名: {first_cols}")
                records.append(dict(source_id="codesearchnet_python", url=f"hf:{used}",
                                    license="元リポジトリごとに異なる（要確認）", source_type="public",
                                    category="pretrain", notes="Python関数。事前学習用"))
        finally:
            for w in writers.values():
                w.close()
        print("\n【事前学習データ】")
        for src in ("wiki", "code"):
            print(f"  {src}: train {stats[f'{src}_train_docs']:,}件 / {stats[f'{src}_train_chars']:,}文字, "
                  f"val {stats[f'{src}_val_docs']:,}件 / {stats[f'{src}_val_chars']:,}文字")

    # ------------------------------------------------------------------ 指示学習用
    if not args.skip_sft:
        heldout = _heldout_texts(args.yardstick)
        leak = LeakFilter(heldout, args.leak_threshold)
        print(f"\n重複検査の対象（物差し＋調整用）: {len(heldout)}件")
        items: list[dict] = []
        seen_q: set[str] = set()
        dropped = Counter()

        def add_items(rows, source, category, max_n, max_q, max_a):
            kept = 0
            for row in rows:
                pr = pair_from_alpaca(row)
                if pr is None:
                    dropped[f"{source}:形式"] += 1
                    continue
                q, a = pr
                if not keep_pair(q, a, max_q, max_a):
                    dropped[f"{source}:長さ/URL"] += 1
                    continue
                if q in seen_q:
                    dropped[f"{source}:重複"] += 1
                    continue
                if leak.is_leak(q):
                    dropped[f"{source}:物差し/調整用と類似"] += 1
                    continue
                seen_q.add(q)
                items.append({"id": f"{source}_{kept:06d}", "input": q, "cot": "", "answer": a,
                              "meta": {"category": category, "source": source, "has_cot": False}})
                kept += 1
                if max_n and kept >= max_n:
                    break
            return kept

        v_cands = [(args.vanilla_dataset, None)] if args.vanilla_dataset else VANILLA_CANDIDATES
        ds, used = _open_stream(v_cands, args.vanilla_local, "vanilla")
        if ds is not None:
            n = add_items(ds, "vanilla", "qa", args.sft_max_vanilla, args.vanilla_max_q, args.vanilla_max_a)
            print(f"llm-japanese-dataset-vanilla({used}): {n:,}件を採用")
            if n == 0:
                print("  [注意] 1件も採用できませんでした。列名（instruction/input/output）を確認してください")
            records.append(dict(source_id="llm_japanese_vanilla", url=f"hf:{used}", license="要確認（データセットページ）",
                                source_type="public", category="qa", notes="日本語指示データ。指示学習用"))

        c_cands = [(args.code_instr_dataset, None)] if args.code_instr_dataset else CODE_INSTR_CANDIDATES
        ds, used = _open_stream(c_cands, args.code_instr_local, "code_instr")
        if ds is not None:
            n = add_items(ds, "code_instr", "code", args.sft_max_code, args.code_max_q, args.code_max_a)
            print(f"コード指示データ({used}): {n:,}件を採用")
            if n == 0:
                print("  [注意] 1件も採用できませんでした。列名（instruction/input/output）を確認してください")
            records.append(dict(source_id="code_instructions", url=f"hf:{used}", license="要確認（データセットページ）",
                                source_type="public", category="code", notes="コード指示データ。指示学習用"))

        if args.include_existing_hf:
            p = Path(args.include_existing_hf)
            for name in ("hf_import.train.jsonl", "hf_import.val.jsonl"):
                if (p / name).exists():
                    with (p / name).open(encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                it = json.loads(line)
                                if it["input"] not in seen_q:
                                    seen_q.add(it["input"])
                                    items.append(it)
            print(f"既存の指示データ(dolly/oasst)を追加: {args.include_existing_hf}")

        print("除外の内訳:", dict(dropped))
        if not items:
            print("変換できた指示データがありません。上の表示を確認してください。")
        else:
            import random
            random.Random(args.seed).shuffle(items)
            n_val = max(1, int(len(items) * args.val_ratio))
            val, train = items[:n_val], items[n_val:]
            for name, data in (("hf_import.train.jsonl", train), ("hf_import.val.jsonl", val)):
                with (sft_dir / name).open("w", encoding="utf-8") as f:
                    for it in data:
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
            with (sft_dir / "hf_corpus.txt").open("w", encoding="utf-8") as f:
                for it in items:
                    f.write(it["input"].replace("\n", " ") + "\n" + it["answer"].replace("\n", " ") + "\n")
            by_cat = Counter(it["meta"]["category"] for it in items)
            print(f"\n✓ 指示データを保存: {sft_dir}（train {len(train):,} / val {len(val):,}）カテゴリ内訳: {dict(by_cat)}")

    _record_sources(records)
    print("\n✅ 完了")


if __name__ == "__main__":
    main()
