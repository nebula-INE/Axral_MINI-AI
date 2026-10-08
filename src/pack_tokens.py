"""
pack_tokens.py
事前学習用データの2つの作業をまとめたスクリプト。

  1) corpus  トークナイザー学習用のコーパスを作る（各データから文字数の予算内でランダムに抜き出し、
             改行・インデントを記号化して、1行が長くなりすぎないよう分割する）
  2) pack    事前学習用の文書を、トークン番号の並び（uint16の.binファイル）に詰める。
             データの種類ごとに別ファイルにする（学習時に混ぜる比率を設定で決められるように）。

実行例:
  python -m src.pack_tokens corpus --pretrain_dir /kaggle/working/p7_data --sft_dir /kaggle/working/hf_import_p7 \\
      --extra_corpus data/v_tpl_p7_corpus.txt --out data/p7_tok_corpus.txt
  python -m src.train_tokenizer --corpus data/p7_tok_corpus.txt --output_dir data/ --vocab_size 32000 \\
      --model_prefix spm_p7 --code_aware --input_sentence_size 2000000
  python -m src.pack_tokens pack --pretrain_dir /kaggle/working/p7_data --tokenizer data/spm_p7.model \\
      --out_dir /kaggle/working/p7_tokens

出力（pack）:
  {out_dir}/train_wiki.bin, train_code.bin, val_wiki.bin, val_code.bin  uint16 のトークン番号
  {out_dir}/pack_meta.json                                              トークン数・1トークンあたりの文字数・語彙数
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

from src.code_text import NL, encode_text


# --------------------------------------------------------------------------- corpus
def chunk_text(encoded: str, max_len: int = 1500) -> list[str]:
    """記号化済みの文章を、できるだけ <nl> か「。」で区切って max_len 文字以内の塊にする。
    （SentencePieceは1行が長すぎる文を学習から捨てるため、1文書=1行にはできない）"""
    out = []
    s = encoded
    while len(s) > max_len:
        cut = -1
        i_nl = s.rfind(NL, 0, max_len)
        if i_nl >= 0:
            cut = i_nl + len(NL)
        i_period = s.rfind("。", 0, max_len)
        if i_period >= 0:
            cut = max(cut, i_period + 1)
        if cut < max_len // 4:  # 区切りが見つからない（または手前すぎる）ときは、長さで切る
            cut = max_len
        out.append(s[:cut])
        s = s[cut:]
    if s.strip():
        out.append(s)
    return out


def _iter_docs(path: Path, p_keep: float, rng: random.Random):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if p_keep >= 1.0 or rng.random() < p_keep:
                yield json.loads(line)


def build_corpus(args) -> None:
    rng = random.Random(args.seed)
    budgets = {"wiki": args.wiki_chars, "code": args.code_chars}
    pre = Path(args.pretrain_dir)
    n_lines = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for src, budget in budgets.items():
            path = pre / "pretrain.train.jsonl"
            if not path.exists():
                continue
            # このデータ種の総文字数を数えて、予算に収まる確率で文書を抜き出す
            total = 0
            with path.open(encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if d["src"] == src:
                        total += len(d["text"])
            if total == 0:
                print(f"  {src}: 文書がありません（スキップ）")
                continue
            p = min(1.0, budget / total)
            used = 0
            for d in _iter_docs(path, 1.0, rng):
                if d["src"] != src or (p < 1.0 and rng.random() >= p):
                    continue
                for ch in chunk_text(encode_text(d["text"]), args.max_line_chars):
                    fout.write(ch.replace("\n", " ") + "\n")
                    used += len(ch)
                    n_lines += 1
            print(f"  {src}: 総{total:,}文字のうち約{used:,}文字を使用（抜き出し確率 {p:.3f}）")
        # 指示データ（短いので全部）と、追加のコーパス（テンプレートデータ等）
        extras = [Path(args.sft_dir) / "hf_corpus.txt"] + [Path(x) for x in (args.extra_corpus or [])]
        for ex in extras:
            if not ex.exists():
                print(f"  ⚠ {ex} が見つかりません（スキップ）")
                continue
            c = 0
            with ex.open(encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line:
                        fout.write(encode_text(line).replace("\n", " ") + "\n")
                        c += 1
            n_lines += c
            print(f"  {ex.name}: {c:,}行")
    print(f"✓ {args.out}（{n_lines:,}行）")


# --------------------------------------------------------------------------- pack
def pack(args) -> None:
    import numpy as np
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    vocab = sp.GetPieceSize()
    if vocab > 65535:
        raise ValueError(f"語彙数{vocab}はuint16に収まりません（65535以下にしてください）")
    codec_on = sp.PieceToId(NL) != sp.unk_id()
    eos = sp.eos_id() if sp.eos_id() >= 0 else 2
    print(f"語彙数 {vocab} / 改行記号 {'あり' if codec_on else 'なし（コードの改行は失われます）'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"vocab_size": vocab, "eos_id": eos, "code_aware": codec_on, "files": {}}
    max_tokens = {"wiki": args.max_tokens_wiki, "code": args.max_tokens_code}

    for split in ("val", "train"):
        path = Path(args.pretrain_dir) / f"pretrain.{split}.jsonl"
        files = {src: open(out_dir / f"{split}_{src}.bin", "wb") for src in ("wiki", "code")}
        counts = {"wiki": 0, "code": 0}
        chars = {"wiki": 0, "code": 0}
        t0 = last_print = time.time()
        batch: dict[str, list[str]] = {"wiki": [], "code": []}

        def flush(src: str) -> None:
            texts = batch[src]
            if not texts:
                return
            enc_texts = [encode_text(t) if codec_on else t for t in texts]
            ids_list = sp.Encode(enc_texts)
            arr = []
            for ids in ids_list:
                arr.extend(ids)
                arr.append(eos)
            np.asarray(arr, dtype=np.uint16).tofile(files[src])
            counts[src] += len(arr)
            chars[src] += sum(len(t) for t in texts)
            batch[src] = []

        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                src = d["src"]
                if src not in files:
                    continue
                if split == "train" and max_tokens[src] and counts[src] >= max_tokens[src]:
                    continue
                batch[src].append(d["text"])
                if len(batch[src]) >= args.batch_docs:
                    flush(src)
                    if time.time() - last_print > 30:
                        last_print = time.time()
                        rate = sum(counts.values()) / max(last_print - t0, 1e-6)
                        print(f"  [{split}] {sum(counts.values()):,}トークン（{rate:,.0f}トークン/秒）", flush=True)
        for src in files:
            flush(src)
            files[src].close()
        for src in ("wiki", "code"):
            meta["files"][f"{split}_{src}"] = {
                "tokens": counts[src], "chars": chars[src],
                "chars_per_token": round(chars[src] / counts[src], 3) if counts[src] else None,
            }
        print(f"✓ {split}: " + ", ".join(f"{s} {counts[s]:,}トークン" for s in counts))

    with open(out_dir / "pack_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="トークナイザー用コーパスの作成 / 事前学習データのトークン化")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus")
    c.add_argument("--pretrain_dir", default="/kaggle/working/p7_data")
    c.add_argument("--sft_dir", default="/kaggle/working/hf_import_p7")
    c.add_argument("--extra_corpus", nargs="*", default=[], help="追加のコーパス（テンプレートデータの *_corpus.txt など）")
    c.add_argument("--out", default="data/p7_tok_corpus.txt")
    c.add_argument("--wiki_chars", type=int, default=60_000_000)
    c.add_argument("--code_chars", type=int, default=30_000_000)
    c.add_argument("--max_line_chars", type=int, default=1500)
    c.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("pack")
    p.add_argument("--pretrain_dir", default="/kaggle/working/p7_data")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out_dir", default="/kaggle/working/p7_tokens")
    p.add_argument("--batch_docs", type=int, default=256)
    p.add_argument("--max_tokens_wiki", type=int, default=0, help="trainに詰める上限（0なら全部）")
    p.add_argument("--max_tokens_code", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(getattr(args, "out", "") or ".") or ".", exist_ok=True)
    build_corpus(args) if args.cmd == "corpus" else pack(args)


if __name__ == "__main__":
    main()
