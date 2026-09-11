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


def main():
    parser = argparse.ArgumentParser(description="SentencePieceトークナイザー学習")
    parser.add_argument("--corpus", required=True, help="学習用コーパス（corpus.txt）")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--model_prefix", default="spm_16k", help="出力ファイル名の接頭辞")
    parser.add_argument("--model_type", default="bpe", choices=["bpe", "unigram", "char", "word"])
    parser.add_argument("--character_coverage", type=float, default=0.9995,
                        help="日本語のような文字種が多い言語では0.9995〜0.9999推奨")
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

    spm.SentencePieceTrainer.Train(
        input=args.corpus,
        model_prefix=model_prefix_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
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
    print(f"\n【動作確認】")
    print(f"  入力: {test_text}")
    print(f"  トークンID数: {len(ids)}")
    print(f"  デコード結果: {decoded}")
    print(f"  一致: {'✓' if decoded.replace(' ', '') == test_text.replace(' ', '') else '✗ (要確認)'}")


if __name__ == "__main__":
    main()
