"""
import_reading_data.py
「渡された文章だけを根拠に答える」力を育てる指示データを作る（読解・回答不能・要約）。

材料（ライセンスは使う前に確認済みだが、公開ページの記載が揺れているため出典を記録する）
  - JSQuAD（JGLUE, Wikipedia 2021-11-01 由来。CC BY-SA 4.0 とされる）… 読解。train を学習に、valid を評価専用にする
  - 日本語Wikipedia（CC BY-SA）… 記事の冒頭段落を「本文→要約」の教師にする（XL-Sum は非商用のため使わない）
  - 回答不能: 質問を「別の記事の文章」と組み合わせ、「文章には書かれていないため、答えられません。」と答えさせる（規則で作る）

出力（--out_dir）
  reading.train.jsonl / reading.val.jsonl   既存の hf_import と同じ形式（mix_data がそのまま読める）
  reading_eval.jsonl                       JSQuAD valid 由来。学習には混ぜない（評価専用）
  --merge_into DIR を付けると DIR/hf_import.train.jsonl / .val.jsonl の末尾に追記する（元ファイルは .bak に退避）

実行例（Kaggle, インターネットON）:
  python -m src.import_reading_data --out_dir /kaggle/working/reading_import --wiki_from_hf 30000
  python -m src.import_reading_data --jsquad_dir /kaggle/input/jsquad --wiki_jsonl /kaggle/working/p7_data/pretrain.train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

# 実物は未確認（このスクリプトを書いた環境からは開けなかった）。開けなければ --jsquad_dir で手元のファイルを指定する
JSQUAD_URLS = {
    "train": ["https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.1/train-v1.1.json"],
    "valid": ["https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.1/valid-v1.1.json"],
}
READ_PROMPT = "次の文章を読んで、質問に答えてください。文章に書かれていないことは答えないでください。\n文章：{ctx}\n質問：{q}"
SUM_PROMPT = "次の文章を、1〜2文で要約してください。\n文章：{body}"
UNANSWERABLE = "文章には書かれていないため、答えられません。"
_SEP_RE = re.compile(r"\s*\[SEP\]\s*")


# --------------------------------------------------------------------------- 純粋な関数（テスト対象）
def parse_jsquad(obj: dict) -> list[dict]:
    """SQuAD v1.1 形式のJSONから (title, context, question, answer, answer_start) を取り出す。
    JSQuAD の context は「記事タイトル [SEP] 本文」の形。タイトルは分離し、answer_start も本文基準にずらす。"""
    rows = []
    for art in obj.get("data", []):
        for para in art.get("paragraphs", []):
            ctx_full = para.get("context", "")
            title, body, shift = art.get("title", ""), ctx_full, 0
            m = _SEP_RE.search(ctx_full)
            if m:
                title, body, shift = ctx_full[:m.start()].strip(), ctx_full[m.end():], m.end()
            for qa in para.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                ans = (qa.get("answers") or [None])[0]
                if not ans or not ans.get("text"):
                    continue
                start = ans.get("answer_start", -1) - shift
                if start < 0 or body[start:start + len(ans["text"])] != ans["text"]:
                    continue  # 位置がずれているものは捨てる（根拠が文章内にあると保証したい）
                rows.append({"title": title, "context": body, "question": qa["question"].strip(),
                             "answer": ans["text"].strip(), "answer_start": start, "id": qa.get("id", "")})
    return rows


def window_context(context: str, start: int, length: int, max_chars: int = 450) -> tuple[str, int] | None:
    """答えを含むまま、文の切れ目（。）で max_chars 以内に切り出す。収まらなければ None。"""
    if len(context) <= max_chars:
        return context, start
    end = start + length
    sents, pos = [], 0
    for m in re.finditer(r"[^。]*。|[^。]+$", context):
        sents.append((m.start(), m.end()))
        pos = m.end()
    if not sents:
        return None
    k = next((i for i, (a, b) in enumerate(sents) if a <= start < b), None)
    if k is None or sents[k][1] < end:  # 答えが文をまたぐ
        return None
    lo = hi = k
    while True:
        grew = False
        if hi + 1 < len(sents) and sents[hi + 1][1] - sents[lo][0] <= max_chars:
            hi += 1; grew = True
        if lo - 1 >= 0 and sents[hi][1] - sents[lo - 1][0] <= max_chars:
            lo -= 1; grew = True
        if not grew:
            break
    if sents[hi][1] - sents[lo][0] > max_chars:
        return None
    a = sents[lo][0]
    return context[a:sents[hi][1]], start - a


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def overlap(small: str, big: str) -> float:
    """small の文字2-gramのうち、big に出てくる割合。"""
    a = _bigrams(small)
    return len(a & _bigrams(big)) / len(a) if a else 0.0


def make_item(item_id: str, prompt: str, answer: str, task: str, source: str) -> dict:
    return {"id": item_id, "input": prompt, "cot": "", "answer": answer,
            "meta": {"category": "qa", "source": source, "task": task, "has_cot": False}}


def make_reading_items(rows: list[dict], max_ctx: int = 450, prefix: str = "jsquad") -> list[dict]:
    items = []
    for i, r in enumerate(rows):
        w = window_context(r["context"], r["answer_start"], len(r["answer"]), max_ctx)
        if w is None:
            continue
        ctx, _ = w
        items.append(make_item(f"{prefix}_{i:06d}", READ_PROMPT.format(ctx=ctx, q=r["question"]),
                               r["answer"], "reading", "jsquad"))
        items[-1]["_title"] = r["title"]
        items[-1]["_ctx"] = ctx
        items[-1]["_q"] = r["question"]
    return items


def make_unanswerable_items(reading: list[dict], n: int, rng: random.Random, max_overlap: float = 0.12,
                            prefix: str = "unans") -> list[dict]:
    """質問を別記事の文章と組み合わせる。質問の2-gramが文章にほとんど現れない組だけ採用する。"""
    out, tries = [], 0
    if len(reading) < 2:
        return out
    while len(out) < n and tries < n * 30:
        tries += 1
        a, b = rng.sample(reading, 2)
        if a["_title"] == b["_title"] or overlap(a["_q"], b["_ctx"]) > max_overlap:
            continue
        out.append(make_item(f"{prefix}_{len(out):06d}", READ_PROMPT.format(ctx=b["_ctx"], q=a["_q"]),
                             UNANSWERABLE, "unanswerable", "jsquad_mismatch"))
    return out


def make_summary_items(docs, max_items: int, body_chars: int = 450, min_overlap: float = 0.30,
                       prefix: str = "sum") -> list[dict]:
    """記事の冒頭段落（要約的）を、続く本文から作る要約課題にする。冒頭の内容が本文に十分現れる組だけ採用する。"""
    out = []
    for text in docs:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        if len(paras) < 3:
            continue
        lead, body = paras[0], ""
        if not (40 <= len(lead) <= 160) or not lead.endswith("。"):
            continue
        for p in paras[1:]:
            if len(body) + len(p) > body_chars:
                break
            body += p
        if len(body) < 200 or overlap(lead, body) < min_overlap:
            continue
        out.append(make_item(f"{prefix}_{len(out):06d}", SUM_PROMPT.format(body=body), lead, "summary", "wiki_lead"))
        if len(out) >= max_items:
            break
    return out


def strip_private(it: dict) -> dict:
    return {k: v for k, v in it.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- 入出力
def _download(urls: list[str], dest: Path) -> bool:
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=60) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  [試行失敗] {u} ({type(e).__name__}: {str(e)[:120]})")
    return False


def load_jsquad(split: str, jsquad_dir: str | None, cache_dir: Path) -> list[dict]:
    names = {"train": "train-v1.1.json", "valid": "valid-v1.1.json"}
    path = None
    if jsquad_dir:
        hits = list(Path(jsquad_dir).rglob(names[split]))
        path = hits[0] if hits else None
    if path is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / names[split]
        if not path.exists() and not _download(JSQUAD_URLS[split], path):
            raise SystemExit(f"JSQuAD {split} を取得できませんでした。JGLUE の GitHub から {names[split]} を手元に置き、"
                             f"--jsquad_dir で指定してください。")
    with open(path, encoding="utf-8") as f:
        return parse_jsquad(json.load(f))


def iter_wiki_docs(args):
    if args.wiki_jsonl:
        with open(args.wiki_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("src", "wiki") == "wiki":
                        yield row["text"]
        return
    if args.wiki_from_hf:
        from src.import_pretrain_data import WIKI_CANDIDATES, _open_stream, clean_wiki
        ds, used = _open_stream(WIKI_CANDIDATES, None, "wiki")
        if ds is None:
            return
        print(f"日本語Wikipedia: {used}（先頭 {args.wiki_from_hf:,} 記事から要約課題を作る）")
        for i, row in enumerate(ds):
            if i >= args.wiki_from_hf:
                break
            t = clean_wiki(row.get("text") or "")
            if t:
                yield t


def split_by_hash(text: str, val_permille: int = 10) -> str:
    return "val" if int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 1000 < val_permille else "train"


def _heldout(yardstick: str) -> list[str]:
    from src.import_pretrain_data import _heldout_texts
    return _heldout_texts(yardstick)


def main() -> None:
    ap = argparse.ArgumentParser(description="読解・回答不能・要約の指示データを作る")
    ap.add_argument("--out_dir", default="/kaggle/working/reading_import")
    ap.add_argument("--merge_into", default=None, help="hf_import.* のあるフォルダ。指定すると末尾に追記する")
    ap.add_argument("--jsquad_dir", default=None, help="train-v1.1.json / valid-v1.1.json のあるフォルダ（無ければGitHubから取得）")
    ap.add_argument("--wiki_jsonl", default=None, help="pretrain.train.jsonl（src=wiki の行を使う）")
    ap.add_argument("--wiki_from_hf", type=int, default=0, help="HFのWikipediaを先頭N記事ストリームして使う")
    ap.add_argument("--max_reading", type=int, default=20000)
    ap.add_argument("--unans_ratio", type=float, default=0.25, help="読解件数に対する回答不能の比率")
    ap.add_argument("--max_summary", type=int, default=8000)
    ap.add_argument("--max_ctx", type=int, default=450)
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--yardstick", default="eval_sets/yardstick_v1.jsonl")
    ap.add_argument("--leak_threshold", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()

    # --- 読解（学習用: JSQuAD train / 評価専用: valid）
    train_rows = load_jsquad("train", args.jsquad_dir, out / "_src")
    rng.shuffle(train_rows)
    reading = make_reading_items(train_rows, args.max_ctx)[:args.max_reading]
    stats["reading"] = len(reading)
    unans = make_unanswerable_items(reading, int(len(reading) * args.unans_ratio), rng)
    stats["unanswerable"] = len(unans)

    eval_rows = load_jsquad("valid", args.jsquad_dir, out / "_src")
    rng.shuffle(eval_rows)
    eval_items = make_reading_items(eval_rows, args.max_ctx, prefix="jsquad_eval")[:args.max_eval]
    eval_unans = make_unanswerable_items(eval_items, len(eval_items) // 4, rng, prefix="unans_eval")

    # --- 要約（Wikipedia冒頭）
    summaries = make_summary_items(iter_wiki_docs(args), args.max_summary) if (args.wiki_jsonl or args.wiki_from_hf) else []
    stats["summary"] = len(summaries)
    if not summaries:
        print("  （要約課題は作りません。--wiki_jsonl か --wiki_from_hf を指定すると作ります）")

    # --- 物差し・調整用との重複除去（質問文・要約本文で検査）
    try:
        from src.import_pretrain_data import LeakFilter
        leak = LeakFilter(_heldout(args.yardstick), args.leak_threshold)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 重複検査を準備できませんでした（{type(e).__name__}: {e}）。このまま続けます")
        leak = None

    def keep(it: dict) -> bool:
        if leak is None:
            return True
        probe = it.get("_q") or it["input"].split("質問：")[-1][:200]
        if it["meta"]["task"] == "summary":
            probe = it["answer"]
        if leak.is_leak(probe):
            stats["物差しと類似で除外"] += 1
            return False
        return True

    all_items = [it for it in reading + unans + summaries if keep(it)]
    rng.shuffle(all_items)
    buckets = {"train": [], "val": []}
    for it in all_items:
        buckets[split_by_hash(it["input"])].append(strip_private(it))
    for name, rows in buckets.items():
        with open(out / f"reading.{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "reading_eval.jsonl", "w", encoding="utf-8") as f:
        for it in eval_items + eval_unans:
            f.write(json.dumps(strip_private(it), ensure_ascii=False) + "\n")

    # --- 出典の記録（ライセンスは公開ページの記載が揺れているため、そのまま残す）
    (out / "SOURCES.md").write_text(
        "- JSQuAD (JGLUE, yahoojapan/JGLUE): Wikipedia(2021-11-01)由来。ライセンスは CC BY-SA 4.0 と案内されているが、"
        "HFのメタデータは cc-by-4.0 と記載が食い違う。出典表示を付けること。\n"
        "- 日本語Wikipedia (wikimedia/wikipedia 20231101.ja): CC BY-SA。冒頭段落→要約課題に使用。\n"
        "- 回答不能課題: JSQuADの質問を別記事の文章と組み合わせて規則で生成（外部モデルは使っていない）。\n", encoding="utf-8")

    print("\n=== 結果 ===")
    print(f"  train {len(buckets['train']):,}件 / val {len(buckets['val']):,}件 / 評価専用 {len(eval_items) + len(eval_unans):,}件")
    print("  内訳:", dict(Counter(r["meta"]["task"] for r in buckets["train"])), dict(stats))

    if args.merge_into:
        d = Path(args.merge_into)
        for name in ("train", "val"):
            tgt, bak = d / f"hf_import.{name}.jsonl", d / f"hf_import.{name}.jsonl.bak"
            if not bak.exists():
                shutil.copy2(tgt, bak)          # 初回だけ元を退避
            shutil.copy2(bak, tgt)              # 毎回「元＋今回分」で作り直す（再実行しても二重にならない）
            with open(tgt, "a", encoding="utf-8") as f:
                for r in buckets[name]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ {d} の hf_import.* に追記しました（元は .bak に退避）。再実行しても二重にはなりません")


if __name__ == "__main__":
    main()
