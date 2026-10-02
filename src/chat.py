"""
chat.py
Axral_MINI-AI と対話するための対話モード専用スクリプト。

infer.py の --interactive は「質問→回答」の一問一答だけだったので、
コマンド操作（ガードのON/OFF切替、ログ保存など）ができる、より実用的な
対話ループとして独立させた。generate_answer() 自体はinfer.pyのものを
そのまま再利用する（ロジックの二重管理を避けるため）。

実行例:
  python -m src.chat --config configs/exp01_kaggle.yaml \\
      --checkpoint results/checkpoints/p1_cot60_gate3/checkpoint_best.pt

  # やり取りをjsonlに保存したい場合
  python -m src.chat --config configs/exp01_kaggle.yaml \\
      --checkpoint results/checkpoints/p1_cot60_gate3/checkpoint_best.pt \\
      --log_path logs/chat_history.jsonl

対話中に使えるコマンド（先頭が "/" の入力はコマンドとして解釈する）:
  /help            コマンド一覧を表示
  /guard on|off    既知トピックガードのON/OFFを切り替える
  /calc on|off     計算機検算のON/OFFを切り替える
  /search on|off   未知トピックのWikipedia検索補完のON/OFF（要インターネット接続）
  /reason on|off   思考過程（まず、〜よって、〜）も表示するかを切り替える
  /topics          学習済み（既知）トピック数を表示
  /save <path>     これまでの対話ログをjsonlとして保存する
  1〜3             回答後に表示される関連トピックを番号で選んで深掘りする
  /reset           画面をクリアする代わりに区切り線を表示する（会話状態自体は元々ステートレス）
  /quit, /exit     終了する（Ctrl+Cでも終了できる）
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from src.infer import generate_answer, load_model_and_tokenizer
from src.utils import split_reasoning_answer
from src.knowledge_base import KNOWN_TOPIC_PAIRS, UNKNOWN_TOPIC_RESPONSE, suggest_followups
from src.utils import load_config

_HELP_TEXT = """\
コマンド一覧:
  /help            このヘルプを表示
  /guard on|off    既知トピックガード（未学習の話題への「分かりません」応答）のON/OFF
  /calc  on|off    計算式の自動検算・訂正のON/OFF
  /search on|off   未知トピックをWikipedia検索で補う（Kaggleは Settings→Internet をONに）
  /reason on|off   思考過程も表示（既定はOFF＝最終回答だけ）
  /topics          既知トピック（学習済みの話題）の登録数を表示
  /save <path>     これまでの対話ログをjsonl形式で保存
  /reset           区切り線を表示（モデル自体は1問1答でステートレスなため、履歴のリセットは不要）
  /quit, /exit     終了（Ctrl+Cでも終了可）
"""


def _print_banner(checkpoint_path: str, guard_on: bool, calc_on: bool, search_on: bool = False) -> None:
    print("=" * 60)
    print("Axral_MINI-AI 対話モード")
    print(f"checkpoint: {checkpoint_path}")
    print(f"既知トピックガード: {'ON' if guard_on else 'OFF'} / 計算機検算: {'ON' if calc_on else 'OFF'} / Web検索: {'ON' if search_on else 'OFF'}")
    print("'/help' でコマンド一覧、'/quit' または Ctrl+C で終了")
    print("=" * 60)


def _save_history(history: list[dict], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for turn in history:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")
    print(f"✓ 対話ログを保存しました: {out_path} ({len(history)}件)")


def run_chat(
    model,
    sp,
    tok_meta: dict,
    device: torch.device,
    checkpoint_path: str,
    max_new_tokens: int = 128,
    strategy: str = "greedy",
    temperature: float = 0.8,
    use_calculator: bool = True,
    use_topic_guard: bool = True,
    use_web_search: bool = False,
    log_path: str | None = None,
) -> None:
    """対話ループ本体。Ctrl+C / EOF / '/quit' のいずれでも安全に終了する。"""
    guard_on = use_topic_guard
    calc_on = use_calculator
    search_on = use_web_search
    show_reason = False  # 思考過程（CoT）も表示するか
    history: list[dict] = []
    pending: list[str] = []  # 直前の回答から提案した関連質問（番号で選択可能）

    _print_banner(checkpoint_path, guard_on, calc_on, search_on)

    while True:
        try:
            text = input("\n質問> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not text:
            continue

        # 直前に提示した関連トピックを番号で選んだ場合は、その質問文に置き換える
        if pending and text.isdigit() and 1 <= int(text) <= len(pending):
            text = pending[int(text) - 1]
            print(f"質問（選択）> {text}")
        pending = []

        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                print("終了します。")
                break
            if cmd == "/help":
                print(_HELP_TEXT)
                continue
            if cmd == "/guard":
                if arg == "on":
                    guard_on = True
                elif arg == "off":
                    guard_on = False
                print(f"既知トピックガード: {'ON' if guard_on else 'OFF'}")
                continue
            if cmd == "/calc":
                if arg == "on":
                    calc_on = True
                elif arg == "off":
                    calc_on = False
                print(f"計算機検算: {'ON' if calc_on else 'OFF'}")
                continue
            if cmd == "/search":
                if arg == "on":
                    search_on = True
                elif arg == "off":
                    search_on = False
                print(f"Web検索: {'ON' if search_on else 'OFF'}"
                      + ("（ガードがOFFのときは検索されません）" if search_on and not guard_on else ""))
                continue
            if cmd == "/reason":
                if arg == "on":
                    show_reason = True
                elif arg == "off":
                    show_reason = False
                print(f"思考過程の表示: {'ON' if show_reason else 'OFF'}")
                continue
            if cmd == "/topics":
                print(f"既知トピック登録数: {len(KNOWN_TOPIC_PAIRS)}件"
                      "（QA・技術・コード・会話カテゴリ合算。src/knowledge_base.py参照）")
                continue
            if cmd == "/save":
                if not arg:
                    print("使い方: /save <保存先パス>（例: /save logs/chat_history.jsonl）")
                elif not history:
                    print("まだ保存する対話がありません。")
                else:
                    _save_history(history, arg)
                continue
            if cmd == "/reset":
                print("-" * 60 + "\n（モデルは1問1答でステートレスなため、会話状態自体のリセットは不要です）")
                continue

            print(f"不明なコマンドです: {cmd}（'/help' で一覧を確認できます）")
            continue

        answer = generate_answer(
            model, sp, tok_meta, text, device,
            max_new_tokens=max_new_tokens,
            strategy=strategy,
            temperature=temperature,
            use_calculator=calc_on,
            use_topic_guard=guard_on,
            use_web_search=search_on,
        )
        if answer == UNKNOWN_TOPIC_RESPONSE or answer.startswith("【Web検索"):
            print(f"生成: {answer}")
        else:
            reasoning, final = split_reasoning_answer(answer)
            print(f"生成: {final}")
            if show_reason and reasoning:
                print(f"  （思考過程: {reasoning}）")

        # 生成文に登場する既知トピックから、話題を枝分かれさせる関連質問を提案する
        if answer != UNKNOWN_TOPIC_RESPONSE and not answer.startswith("【Web検索"):
            pending = suggest_followups(text, answer)
            if pending:
                print("  関連トピック（番号を入力すると深掘りできます）:")
                for i, q in enumerate(pending, 1):
                    print(f"    {i}) {q}")

        history.append({
            "timestamp": datetime.now().isoformat(),
            "input": text,
            "output": answer,
            "guard_on": guard_on,
            "calc_on": calc_on,
            "search_on": search_on,
        })

    if log_path and history:
        _save_history(history, log_path)


def _main():
    parser = argparse.ArgumentParser(description="Axral_MINI-AI 対話モード")
    parser.add_argument("--config", required=True, help="configs/exp01_kaggle.yaml など")
    parser.add_argument("--checkpoint", required=True, help="checkpoint_best.pt のパス")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--strategy", default="greedy", choices=["greedy", "sampling"])
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling時のみ使用")
    parser.add_argument("--no_calculator", action="store_true",
                        help="起動時点で計算機検算を無効化する（対話中に /calc on で再度有効化可能）")
    parser.add_argument("--no_topic_guard", action="store_true",
                        help="起動時点で既知トピックガードを無効化する（対話中に /guard on で再度有効化可能）")
    parser.add_argument("--web_search", action="store_true",
                        help="起動時点でWeb検索補完を有効化する（対話中に /search on|off で切替可能）")
    parser.add_argument("--log_path", default=None,
                        help="終了時に対話ログをjsonl形式で自動保存するパス（省略可。対話中の /save でも保存可能）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    model, sp, tok_meta = load_model_and_tokenizer(config, args.checkpoint, device)

    run_chat(
        model, sp, tok_meta, device,
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        strategy=args.strategy,
        temperature=args.temperature,
        use_calculator=not args.no_calculator,
        use_topic_guard=not args.no_topic_guard,
        use_web_search=args.web_search,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    _main()
