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
import re

import torch

from src.model import TransformerLM
from src.utils import load_config


_CALC_EXPR_RE = re.compile(r"(\d+)\s*([×÷x*/])\s*(\d+)\s*=\s*(\d+)")


def apply_calculator_correction(text: str) -> str:
    """生成テキスト中の「A × B = C」のような式を実際に計算し直し、
    Cが誤っていれば正しい値に置き換える（末尾の重複した答えも合わせて修正する）。

    モデル自体に掛け算・割り算を正確に実行させるのは小規模モデルには荷が重いため
    （sanity_checkで、数字のコピーは正確でも計算結果だけ間違うケースが多発した）、
    決定的な後処理として計算機に肩代わりさせる。CoTのフォーマットが
    「A op B = C」に統一されていることを前提にした、実用的な折衷案。
    """
    op_map = {"×": "*", "x": "*", "*": "*", "÷": "/", "/": "/"}

    def _fix(match: re.Match) -> str:
        a, op, b, stated = match.groups()
        a_val, b_val = int(a), int(b)
        py_op = op_map.get(op, "*")
        if py_op == "*":
            correct = a_val * b_val
        else:
            if b_val == 0:
                return match.group(0)  # ゼロ除算は手を出さない
            correct = a_val // b_val if a_val % b_val == 0 else round(a_val / b_val, 2)

        if str(correct) == stated:
            return match.group(0)  # 既に正しい

        # 誤って述べられた数値を、後続の「答えは○○」等でも合わせて修正する
        text_container["wrong_to_correct"][stated] = str(correct)
        return f"{a} {op} {b} = {correct}"

    text_container = {"wrong_to_correct": {}}
    corrected = _CALC_EXPR_RE.sub(_fix, text)

    # 式の中だけでなく、末尾に単独で繰り返される誤答（例: "988円"）も置換する
    for wrong, correct in text_container["wrong_to_correct"].items():
        corrected = re.sub(rf"(?<!\d){re.escape(wrong)}(?!\d)", correct, corrected)

    return corrected


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
    use_calculator: bool = True,
) -> str:
    """入力テキストに対して [BOS] + input_ids を渡して自己回帰生成する。
    学習時と同じ構造（[BOS] input_ids cot answer [EOS]）を踏襲する。
    生成結果はEOSトークンで打ち切り、テキストにデコードして返す。

    use_calculator=True（デフォルト）の場合、生成テキスト中の「A × B = C」形式の
    計算式を検算し、誤っていれば正しい値に自動訂正する（apply_calculator_correction参照）。
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
    output = sp.DecodeIds(new_tokens)

    if use_calculator:
        output = apply_calculator_correction(output)

    return output


def _main():
    parser = argparse.ArgumentParser(description="学習済みモデルで推論する")
    parser.add_argument("--config", required=True, help="configs/exp01_kaggle.yaml など")
    parser.add_argument("--checkpoint", required=True, help="checkpoint_best.pt のパス")
    parser.add_argument("--text", default=None, help="質問文（1回だけ推論する場合）")
    parser.add_argument("--interactive", action="store_true", help="対話的に複数質問する")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--strategy", default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling時のみ使用")
    parser.add_argument("--no_calculator", action="store_true",
                        help="計算式の自動検算・訂正を無効化する（モデル本来の計算力を見たい場合）")
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
            use_calculator=not args.no_calculator,
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
