"""
eval.py
評価関数（修正版 plan_10 §4.4 に基づく実装）。

旧版の問題点（teacher-forcing下のargmaxをEM/F1として扱っていた）を修正し、
EM/F1/数学正答率は必ず model.generate() による自己回帰生成テキストで計算する。
loss/PPLのみ teacher-forcing で計算する。
"""
from __future__ import annotations

import re
from collections import Counter

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# テキスト正規化 & EM/F1
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "", s)  # 日本語は分かち書きしない前提で空白除去
    s = re.sub(r"[。、．，,\.]", "", s)
    return s


def compute_em(preds: list[str], refs: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(_normalize_text(p) == _normalize_text(r) for p, r in zip(preds, refs)) / len(preds)


def _char_f1(pred: str, ref: str) -> float:
    """日本語は単語分割が非自明なため、文字bag-of-charsでF1を近似する。"""
    pred_chars = list(_normalize_text(pred))
    ref_chars = list(_normalize_text(ref))
    if not pred_chars or not ref_chars:
        return float(pred_chars == ref_chars)

    common = Counter(pred_chars) & Counter(ref_chars)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_chars)
    recall = num_same / len(ref_chars)
    return 2 * precision * recall / (precision + recall)


def compute_f1(preds: list[str], refs: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(_char_f1(p, r) for p, r in zip(preds, refs)) / len(preds)


def extract_final_answer(generated_text: str) -> str:
    """生成テキストの末尾から最終回答らしき部分を抽出する簡易ヘルパー（算数向け）。
    データの answer フォーマット（例: "400円"）に合わせて正規表現は調整すること。
    """
    match = re.search(r"([-\d,\.]+\s*(?:円|個|人|%)?)\s*$", generated_text.strip())
    return match.group(1).strip() if match else generated_text.strip()


_ANSWER_MARKERS = ("答えは", "よって、", "したがって、", "よって", "したがって")


def extract_final_segment(generated_text: str) -> str:
    """QA/一般カテゴリ向けの解答抽出ヘルパー。

    修正前は生成テキスト全体（CoTを含む長文）を正解の短い語句とそのまま
    完全一致比較していたため、CoTが生成されている限りEMが恒常的に0になっていた。
    「答えは」等のマーカー以降を優先的に抜き出し、無ければ最後の文（句点区切り）を使う。
    """
    text = generated_text.strip()
    if not text:
        return text

    for marker in _ANSWER_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            break

    parts = [p.strip() for p in re.split("[。\n]", text) if p.strip()]
    result = parts[-1] if parts else text
    result = result.strip("。 　")

    # 「である/です/だ」等の文末表現を除去し、正解語句とEM比較しやすくする
    for suffix in ("である", "です", "だ"):
        if result.endswith(suffix):
            result = result[: -len(suffix)]
            break

    return result.strip()


# ---------------------------------------------------------------------------
# メイン評価関数
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, tokenizer=None, max_gen_tokens: int = 128) -> tuple[float, dict]:
    """バリデーション損失（teacher-forcing）+ 生成ベースのタスク別メトリクスを計算する。

    tokenizer は decode(ids) -> str を持つオブジェクト（例: sentencepiece.SentencePieceProcessor
    のラッパー）。None の場合は生成ベース評価をスキップし、loss/PPLのみ返す
    （torch/sentencepieceが揃わない環境でのユニットテスト用）。
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    generated_texts: list[str] = []
    reference_texts: list[str] = []
    categories_list: list[str] = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids)
        # 次トークン予測: logits[:, :-1] が labels[:, 1:] を予測する
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()
        n_batches += 1

        if tokenizer is not None:
            prompt_ids = batch["prompt_ids"].to(device)
            gen_ids = model.generate(prompt_ids, max_new_tokens=max_gen_tokens, strategy="greedy")
            for i in range(gen_ids.size(0)):
                # プロンプト部分を除いた生成分だけデコード
                new_tokens = gen_ids[i, prompt_ids.size(1):].tolist()
                generated_texts.append(tokenizer.decode(new_tokens))
                reference_texts.append(batch["answer_text"][i])
                categories_list.append(batch["category"][i])

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {
        "loss": avg_loss,
        "ppl": float(torch.exp(torch.tensor(avg_loss))),
    }

    if tokenizer is not None:
        qa_gen = [g for g, c in zip(generated_texts, categories_list) if c == "qa"]
        qa_ref = [r for r, c in zip(reference_texts, categories_list) if c == "qa"]
        if qa_gen:
            # 修正: 生成テキスト全体ではなく、抽出した最終回答部分で比較する
            qa_gen_extracted = [extract_final_segment(g) for g in qa_gen]
            metrics["qa_em"] = compute_em(qa_gen_extracted, qa_ref)
            metrics["qa_f1"] = compute_f1(qa_gen_extracted, qa_ref)
            metrics["em"] = metrics["qa_em"]
            metrics["f1"] = metrics["qa_f1"]
        else:
            metrics["em"] = 0.0
            metrics["f1"] = 0.0

        math_gen = [g for g, c in zip(generated_texts, categories_list) if c == "arithmetic"]
        math_ref = [r for r, c in zip(reference_texts, categories_list) if c == "arithmetic"]
        if math_gen:
            metrics["math_accuracy"] = sum(
                _normalize_text(extract_final_answer(g)) == _normalize_text(r)
                for g, r in zip(math_gen, math_ref)
            ) / len(math_gen)

        cot_count = sum(1 for c in categories_list if c == "cot")
        if cot_count:
            metrics["cot_count"] = cot_count
    else:
        metrics["em"] = 0.0
        metrics["f1"] = 0.0

    return avg_loss, metrics
