"""
train_tokenizer.py
corpus.txt から SentencePiece トークナイザーを学習する（plan §3.1 段階8 対応）。

実行:
    python src/train_tokenizer.py --corpus data/corpus.txt --output_dir data/ --vocab_size 16000

Kaggle上での実行例（データがKaggle Dataset側にある場合）:
    python src/train_tokenizer.py \
        --corpus /kaggle/input/vose-initial-llm-data-v001/corpus.txt \
        --output_dir /kaggle/working/data/ \
        --vocab_size 16000

学習済みモデル（spm_16k.model / .vocab）は、そのまま
kaggle_dataset/ にコピーして次回データセットバージョンに含めておくと、
以後は毎回学習し直す必要がなくなる（package_kaggle_dataset.pyのFILES_TO_PACKAGEに追加）。
"""
from __future__ import annotations

import argparse
import os
import sys

# `python src/train_tokenizer.py` で直接実行しても `from src.xxx import ...` が通るよう、プロジェクトルートを追加する
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="SentencePieceトークナイザー学習")
    parser.add_argument("--corpus", required=True, help="学習用コーパス（corpus.txt）")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--model_prefix", default="spm_16k", help="出力ファイル名の接頭辞")
    parser.add_argument("--model_type", default="bpe", choices=["bpe", "unigram", "char", "word"])
    parser.add_argument("--character_coverage", type=float, default=0.9999,
                        help="低頻度の文字（新しく追加したQAトピックの固有名詞・専門用語等）が"
                             "vocab_sizeの制約で切り捨てられ<unk>化するのを防ぐため、"
                             "デフォルトを0.9995→0.9999に引き上げ。日本語のような文字種が"
                             "多い言語では1.0に近いほど安全（実データで、沖縄/亜熱帯/エベレスト等"
                             "の低頻度語が<unk>化する問題が発生したため変更）。")
    parser.add_argument("--split_digits", type=lambda x: x.lower() != "false", default=True,
                        help="数字を1文字ずつ分割してトークン化するか（デフォルトTrue）。"
                             "算数タスクでは繰り上がりを含む筆算的な処理を学習しやすくするため強く推奨。"
                             "Falseにすると複数桁の数字が1つのBPEトークンにまとまり、"
                             "モデルが数値の組み合わせを丸暗記するだけになり計算を学習しにくくなる"
                             "（実データのsanity_checkで発覚: 45×4=240等、入力の数字は正しく"
                             "コピーできても掛け算自体を間違えるケースが頻発した）。")
    parser.add_argument("--code_aware", action="store_true",
                        help="コード対応モード。改行(<nl>)とインデント(<i>)を専用記号として登録し、空白の整理を無効にする。"
                             "コーパスは src/code_text.encode_text で変換済みであること（build_tokenizer_corpus が作る）。")
    parser.add_argument("--input_sentence_size", type=int, default=0,
                        help="学習に使う行数の上限（0なら全部）。大きなコーパスのときは200万程度にして、"
                             "ランダムに抜き出す（学習時間とメモリを抑える）。")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        raise FileNotFoundError(
            f"コーパスファイルが見つかりません: {args.corpus}\n"
            f"先に generate_data.py を実行して corpus.txt を作成してください。"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    model_prefix_path = os.path.join(args.output_dir, args.model_prefix)

    import sentencepiece as spm

    print(f"SentencePiece学習開始...")
    print(f"  コーパス: {args.corpus}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  model_type: {args.model_type}")
    print(f"  split_digits: {args.split_digits}")

    extra = {}
    if args.code_aware:
        from src.code_text import USER_SYMBOLS
        # max_sentence_length はバイト数。日本語は1文字3バイトなので、既定の4192だと約1400文字を超える行が捨てられる
        extra.update(user_defined_symbols=USER_SYMBOLS, remove_extra_whitespaces=False, max_sentence_length=16384)
        print(f"  code_aware: 改行・インデント記号 {USER_SYMBOLS} を登録（空白の整理は無効）")
    if args.input_sentence_size:
        extra.update(input_sentence_size=args.input_sentence_size, shuffle_input_sentence=True)

    spm.SentencePieceTrainer.Train(
        input=args.corpus,
        model_prefix=model_prefix_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        split_digits=args.split_digits,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        # コーパスが小さく、指定した語彙数に届かない場合にエラーで止まらず、
        # 作れる範囲の語彙数で学習を終えるようにする（実際の語彙数は学習後に確認すること）。
        hard_vocab_limit=False,
        **extra,
    )

    model_path = f"{model_prefix_path}.model"
    vocab_path = f"{model_prefix_path}.vocab"

    print(f"\n✅ 学習完了")
    print(f"   {model_path}")
    print(f"   {vocab_path}")

    # 動作確認（簡単なエンコード/デコードテスト）
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    test_text = "3人で1200円を割り勘すると1人いくら？"
    ids = sp.EncodeAsIds(test_text)
    decoded = sp.DecodeIds(ids)

    # SentencePieceはデフォルトでnmt_nfkc正規化を行うため、全角記号（？等）が
    # 半角（?）に変換されるのは仕様であり不具合ではない。unicodedata.normalize("NFKC", ...)
    # で同じ正規化をかけた上で比較する。
    import unicodedata
    normalized_input = unicodedata.normalize("NFKC", test_text).replace(" ", "")
    normalized_decoded = unicodedata.normalize("NFKC", decoded).replace(" ", "")
    is_match = normalized_decoded == normalized_input

    if args.code_aware:
        from src.code_text import CodecTokenizer
        codec = CodecTokenizer(sp)
        sample = "def add(a, b):\n    return a + b\n"
        back = codec.decode(codec.encode(sample))
        print(f"\n【コードの往復確認】{'✓ 一致' if back.strip() == sample.strip() else '✗ 不一致'}")
        print(repr(back))

    print(f"\n【動作確認】")
    print(f"  入力: {test_text}")
    print(f"  トークンID数: {len(ids)}")
    print(f"  デコード結果: {decoded}")
    if is_match:
        print(f"  一致: ✓（NFKC正規化後で一致。全角/半角記号の変換はSentencePieceの仕様であり問題なし）")
    else:
        print(f"  一致: ✗ 要確認（NFKC正規化後も入力と異なる。文字化けや語彙不足の可能性）")



if __name__ == "__main__":
    main()
