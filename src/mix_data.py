"""
mix_data.py
手作りテンプレートデータ（generate_data.pyの出力）と、公開データ（import_hf_data.pyの出力）を
1つの学習セットに混ぜる。

実行例:
  python -m src.mix_data --template_version v_tpl60 --hf_dir /kaggle/working/hf_import

出力（data/ 配下）:
  v_mix.train.jsonl / v_mix.val.jsonl   混合した学習用・検証用データ
  v_mix_corpus.txt                      トークナイザー学習用コーパス（両方のテキストを結合）

混ぜ方:
  テンプレートの学習データ件数が公開データより多い場合は、公開データと同数になるよう
  ランダムに間引く（約半分ずつにするため。--no_balance で無効化できる）。
  検証用データも同様に、テンプレート側が多ければ公開データと同数に揃えてから結合する。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _share(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        v = item.get("meta", {}).get(key, "template" if key == "source" else "unknown")
        out[v] = out.get(v, 0) + 1
    return out


def main():
    parser = argparse.ArgumentParser(description="テンプレートデータと公開データを混ぜる")
    parser.add_argument("--template_version", default="v_tpl60",
                        help="generate_data.py の --version（data/{version}.train.jsonl を読む）")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--hf_dir", default="/kaggle/working/hf_import")
    parser.add_argument("--out_version", default="v_mix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refusal_version", default=None,
                        help="拒否学習データ（refusal_data.py の --version。例: v_refusal）。指定すると学習・検証・コーパスに加える")
    parser.add_argument("--no_balance", action="store_true",
                        help="テンプレートを間引かず、全件使う")
    args = parser.parse_args()

    data_dir, hf_dir = Path(args.data_dir), Path(args.hf_dir)
    tv = args.template_version

    tpl_train = read_jsonl(data_dir / f"{tv}.train.jsonl")
    tpl_val = read_jsonl(data_dir / f"{tv}.val.jsonl")
    hf_train = read_jsonl(hf_dir / "hf_import.train.jsonl")
    hf_val = read_jsonl(hf_dir / "hf_import.val.jsonl")

    rng = random.Random(args.seed)
    if not args.no_balance and len(tpl_train) > len(hf_train):
        tpl_train = rng.sample(tpl_train, len(hf_train))
    # 検証データも同様に揃える（テンプレート側が多いと、検証損失・早期終了の判定が
    # テンプレートに偏る。生成ベースの評価は件数に比例して遅くもなる）。
    # ※ sanity_check用に、間引く前のテンプレート検証データ(data/{version}.val.jsonl)は別途そのまま残る。
    if not args.no_balance and len(tpl_val) > len(hf_val):
        tpl_val = rng.sample(tpl_val, len(hf_val))

    # 拒否学習データ（「分からない」を答えるデータ）。間引かず、そのまま加える。
    ref_train, ref_val = [], []
    if args.refusal_version:
        ref_train = read_jsonl(data_dir / f"{args.refusal_version}.train.jsonl")
        ref_val = read_jsonl(data_dir / f"{args.refusal_version}.val.jsonl")
        # 公開データの学習用の質問と同じ文面を拒否にすると矛盾するため、取り除く
        hf_inputs = {it["input"] for it in hf_train}
        before = len(ref_train)
        ref_train = [it for it in ref_train if it["input"] not in hf_inputs]
        if len(ref_train) != before:
            print(f"  ※ 公開データと同じ質問の拒否データ{before - len(ref_train)}件を除外しました")

    train = tpl_train + hf_train + ref_train
    val = tpl_val + hf_val + ref_val
    rng.shuffle(train)
    rng.shuffle(val)

    out_version = args.out_version
    write_jsonl(data_dir / f"{out_version}.train.jsonl", train)
    write_jsonl(data_dir / f"{out_version}.val.jsonl", val)

    # トークナイザー学習用コーパス: テンプレート側（CoTの文章を含む）＋公開データ側
    corpus_path = data_dir / f"{out_version}_corpus.txt"
    with corpus_path.open("w", encoding="utf-8") as fout:
        sources = [data_dir / f"{tv}_corpus.txt", hf_dir / "hf_corpus.txt"]
        if args.refusal_version:
            sources.append(data_dir / f"{args.refusal_version}_corpus.txt")
        for src in sources:
            with src.open(encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    print(f"✓ 混合データを保存しました（{out_version}）")
    print(f"  train: テンプレート{len(tpl_train)}件 + 公開データ{len(hf_train)}件 = {len(train)}件")
    print(f"  val  : テンプレート{len(tpl_val)}件 + 公開データ{len(hf_val)}件 = {len(val)}件")
    if args.refusal_version:
        print(f"  拒否学習データ: train {len(ref_train)}件 / val {len(ref_val)}件を含む")
    print(f"  trainのソース内訳: {_share(train, 'source')}")
    print(f"  trainのカテゴリ内訳: {_share(train, 'category')}")
    print(f"  コーパス: {corpus_path}")


if __name__ == "__main__":
    main()
