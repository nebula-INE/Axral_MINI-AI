"""
infer.py
学習済みcheckpoint（checkpoint_best.pt）を使って、入力テキストに対する回答を生成する。

実行例:
  python -m src.infer --config configs/exp01_kaggle.yaml \
      --checkpoint results/checkpoints/p1_cot20/checkpoint_best.pt \
      --text "3人で1200円を割り勘すると1人いくら？"

  # 対話的に複数質問したい場合
  python -m src.infer --config configs/exp01_kaggle.yaml \
      --checkpoint results/checkpoints/p1_cot20/checkpoint_best.pt \
      --interactive
"""
from __future__ import annotations

import argparse

import torch

from src.model import TransformerLM
from src.utils import load_config


def load_model_and_tokenizer(config: dict, checkpoint_path: str, device: torch.device):
    """config・checkpointからモデルとSentencePieceトークナイザーを構築する。
    再利用しやすいよう関数化（他スクリプトからのimportも想定）。
    """
    import sentencepiece as spm

    tokenizer_path = config["data"].get("tokenizer_path")
    if not tokenizer_path:
        raise RuntimeError("configの data.tokenizer_path が未設定です。推論にはトークナイザーが必須です。")

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    tok_meta = {
        "pad_id": sp.pad_id() if sp.pad_id() >= 0 else 0,
        "bos_id": sp.bos_id() if sp.bos_id() >= 0 else 1,
        "eos_id": sp.eos_id() if sp.eos_id() >= 0 else 2,
    }

    model_cfg = dict(config["model"])
    model_cfg["vocab_size"] = config["data"].get("vocab_size", 16000)
    model = TransformerLM(model_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ checkpoint 読み込み完了: {checkpoint_path} (step={checkpoint.get('step', '?')})")

    return model, sp, tok_meta


@torch.no_grad()
def generate_answer(
    model,
    sp,
    tok_meta: dict,
    text: str,
    device: torch.device,
    max_new_tokens: int = 128,
    strategy: str = "greedy",
    temperature: float = 0.8,
) -> str:
    """入力テキストに対して [BOS] + input_ids を渡して自己回帰生成する。
    学習時と同じ構造（[BOS] input_ids cot answer [EOS]）を踏襲する。
    生成結果はEOSトークンで打ち切り、テキストにデコードして返す。
    """
    input_ids = [tok_meta["bos_id"]] + sp.EncodeAsIds(text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        eos_token_id=tok_meta["eos_id"],
        strategy=strategy,
        temperature=temperature,
    )
    new_tokens = generated[0, input_tensor.size(1):].tolist()
    if tok_meta["eos_id"] in new_tokens:
        new_tokens = new_tokens[: new_tokens.index(tok_meta["eos_id"])]
    return sp.DecodeIds(new_tokens)


def _main():
    parser = argparse.ArgumentParser(description="学習済みモデルで推論する")
    parser.add_argument("--config", required=True, help="configs/exp01_kaggle.yaml など")
    parser.add_argument("--checkpoint", required=True, help="checkpoint_best.pt のパス")
    parser.add_argument("--text", default=None, help="質問文（1回だけ推論する場合）")
    parser.add_argument("--interactive", action="store_true", help="対話的に複数質問する")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--strategy", default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling時のみ使用")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    model, sp, tok_meta = load_model_and_tokenizer(config, args.checkpoint, device)

    def ask(text: str) -> str:
        return generate_answer(
            model, sp, tok_meta, text, device,
            max_new_tokens=args.max_new_tokens,
            strategy=args.strategy,
            temperature=args.temperature,
        )

    if args.interactive:
        print("対話モード（'quit' または 'exit' で終了）")
        while True:
            text = input("\n質問> ").strip()
            if text.lower() in ("quit", "exit", ""):
                break
            print(f"生成: {ask(text)}")
    elif args.text:
        output = ask(args.text)
        print(f"\n入力: {args.text}")
        print(f"生成: {output}")
    else:
        parser.error("--text か --interactive のどちらかを指定してください")


if __name__ == "__main__":
    _main()
