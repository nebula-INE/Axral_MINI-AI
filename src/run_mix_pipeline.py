"""
run_mix_pipeline.py
手作りテンプレートデータ＋公開日本語データ（dolly/oasst）を混ぜて、
トークナイザー（語彙16,000）の再学習から学習まで通しで実行するパイプライン。

Kaggleノートブックのセルで:
  !python /kaggle/working/Axral_MINI-AI/src/run_mix_pipeline.py

途中まで（学習の手前まで）で止めて確認したい場合:
  !python /kaggle/working/Axral_MINI-AI/src/run_mix_pipeline.py --skip_train

学習率・エポック数だけ変えて学習をやり直す場合（データ・トークナイザーは前回のものを再利用）:
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_mix_pipeline --reuse_data
  （既定は lr=3e-4, epochs=15, warmup=400, 実験名 p5_refusal。--lr/--epochs/--exp_name で変更可）

ステップ:
  1. 公開データの取得・変換（/kaggle/working/hf_import に既にあれば再利用）
  2. テンプレートデータ生成（cot_ratio=0.6。Gate3でcot60が最良だったため）
  3. テンプレートと公開データを混ぜる（約半分ずつ）
  4. トークナイザー再学習（語彙16,000）→ 実際の語彙数を確認
  5. configs/exp_mix.yaml を生成（語彙数・データパス・実験名を差し替え）
  6. 前処理（token_ids化）
  7. 学習
  8. checkpoint・トークナイザー・configを kaggle_dataset_mix/ に退避
  9. テンプレート側の検証データでsanity_check（失敗しても続行）
  10. 物差し yardstick_v1 で、未知の言い回しへの対応力を測る（失敗しても続行）

設計上の注意:
  - 前回のGate3パイプラインと同じく、各ステップの直後に生成物の存在確認を入れ、
    失敗したら即停止する。
  - テンプレートデータは新しいバージョン名（v_tpl60）で作るため、これまでの
    v_cot20/40/60 のデータは上書きされない。
  - トークナイザーとcheckpointは必ずセットで保存する（片方だけ消えると推論できなくなる）。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys


def run(cmd: list[str], step: str, cwd: str, fatal: bool = True) -> bool:
    print(f"\n{'=' * 60}\n▶ {step}\n{'=' * 60}")
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        msg = f"❌ ステップ「{step}」が失敗しました（終了コード {result.returncode}）。"
        if fatal:
            raise RuntimeError(msg + "ここで停止します。上のログを確認してください。")
        print("⚠️ " + msg + "（このステップは失敗しても続行します）")
        return False
    print(f"✓ {step} 完了")
    return True


def require_file(root: str, relpath: str, step: str) -> None:
    full = relpath if os.path.isabs(relpath) else os.path.join(root, relpath)
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"❌ 「{step}」の後に期待されるファイルが見つかりません: {full}\n"
            f"このステップが実際には失敗していた可能性があります。上のログを確認してください。"
        )
    print(f"  ✓ 確認: {full} ({os.path.getsize(full):,} bytes)")


def build_config(template_text: str, exp_name: str, tok_model: str,
                 vocab_size: int, version: str,
                 lr: float | None = None, epochs: int | None = None,
                 warmup_steps: int | None = None) -> str:
    """exp_cot60.yaml の中身から、混合データ用のconfig文字列を作る。"""
    replacements = [
        (r"^experiment_name:.*$", f"experiment_name: {exp_name}"),
        (r"^  train_path:.*$", f"  train_path: data/{version}_processed.train.jsonl"),
        (r"^  val_path:.*$", f"  val_path: data/{version}_processed.val.jsonl"),
        (r"^  tokenizer_path:.*$", f"  tokenizer_path: {tok_model}"),
        (r"^  vocab_size:.*$", f"  vocab_size: {vocab_size}"),
    ]
    if lr is not None:
        replacements.append((r"^  lr:.*$", f"  lr: {lr:g}"))
    if warmup_steps is not None:
        replacements.append((r"^  lr_warmup_steps:.*$", f"  lr_warmup_steps: {warmup_steps}"))
    if epochs is not None:
        replacements.append((r"^  epochs:.*$", f"  epochs: {epochs}"))
    text = template_text
    for pattern, repl in replacements:
        text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"config内に想定した行が見つかりません: {pattern}")
    # テンプレート由来の先頭コメント（「cot20」等の古い説明）は取り除く
    lines = text.split("\n")
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    header = "# 公開データ(dolly/oasst)＋テンプレートの混合学習。run_mix_pipeline.py が自動生成\n"
    return header + "\n".join(lines)


def count_params(vocab: int, d_model: int, n_layers: int, d_ff: int, max_seq: int) -> int:
    per_layer = (3 * d_model * d_model + 3 * d_model + d_model * d_model + d_model
                 + d_model * d_ff + d_ff + d_ff * d_model + d_model + 4 * d_model)
    return vocab * d_model + max_seq * d_model + n_layers * per_layer + 2 * d_model


def main():
    parser = argparse.ArgumentParser(description="公開データ＋テンプレートの混合学習パイプライン")
    parser.add_argument("--project_root", default="/kaggle/working/Axral_MINI-AI")
    parser.add_argument("--hf_dir", default="/kaggle/working/hf_import")
    parser.add_argument("--template_samples", type=int, default=14000,
                        help="生成するテンプレートデータの総数（公開データ約1.3万件に合わせた値）")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", default="p5_refusal",
                        help="実験名（checkpointの保存先）。前回(p2_mix_hf_11m)と別名にすること。"
                             "同名だと resume_from_checkpoint で前回の続きから始まってしまう")
    parser.add_argument("--tokenizer_prefix", default=None,
                        help="トークナイザーのファイル名（data/{prefix}.model）。既定は実験ごとに別名（spm_{exp_name}）。"
                             "同じ名前で作り直すと、以前のcheckpointが使えなくなるため。--reuse_data 時の既定は前回の spm_mix_16k")
    parser.add_argument("--no_refusal", action="store_true",
                        help="拒否学習データ（「分からない」を答えるデータ）を混ぜない（p4と同じ構成に戻す）")
    parser.add_argument("--arith_style", choices=["oneshot", "stepwise"], default="oneshot",
                        help="算数CoTの形式。stepwise=位ごとに分解した手順つき（p6 の施策）")
    parser.add_argument("--arith_ranges", choices=["default", "wide"], default="default",
                        help="stepwise時の数字の範囲（wide=広い範囲）")
    parser.add_argument("--lr", type=float, default=3e-4, help="最大学習率（前回は1e-4）")
    parser.add_argument("--epochs", type=int, default=15, help="エポック数（前回は10）")
    parser.add_argument("--warmup_steps", type=int, default=400, help="ウォームアップ（前回は1000）")
    parser.add_argument("--reuse_data", action="store_true",
                        help="前回作った混合データ・前処理済みデータ・トークナイザーを再利用し、"
                             "学習だけをやり直す（データ生成・トークナイザー学習・前処理を省略）")
    parser.add_argument("--skip_train", action="store_true", help="学習の手前（前処理まで）で止める")
    args = parser.parse_args()

    root = args.project_root
    os.chdir(root)
    assert os.path.exists("src") and os.path.exists("configs"), f"プロジェクトルートが不正です: {root}"
    py = sys.executable

    tpl_version, mix_version = "v_tpl60", "v_mix"
    tok_prefix = args.tokenizer_prefix or ("spm_mix_16k" if args.reuse_data else f"spm_{args.exp_name}")
    tok_model = f"data/{tok_prefix}.model"

    if not args.reuse_data:
        # 1. 公開データ
        hf_train = os.path.join(args.hf_dir, "hf_import.train.jsonl")
        if os.path.exists(hf_train):
            print(f"公開データは変換済みのため再利用します: {args.hf_dir}")
        else:
            run([py, "-m", "src.import_hf_data", "--output_dir", args.hf_dir],
                "公開データの取得・変換（要インターネット）", root)
        require_file(root, hf_train, "公開データ")
        require_file(root, os.path.join(args.hf_dir, "hf_corpus.txt"), "公開データ")

        run([py, "-m", "src.paraphrase_bank", "--check"], "言い換えバンクの検査（物差しとの重複チェック）", root)

        # 2. テンプレートデータ
        run([py, "src/generate_data.py", "--output_dir", "data/",
             "--num_samples", str(args.template_samples), "--seed", str(args.seed),
             "--cot_ratio", "0.6", "--version", tpl_version,
             "--arith_style", args.arith_style, "--arith_ranges", args.arith_ranges],
            "テンプレートデータ生成（cot_ratio=0.6）", root)
        for suffix in (".train.jsonl", ".val.jsonl", "_corpus.txt"):
            require_file(root, f"data/{tpl_version}{suffix}", "テンプレートデータ生成")

        # 2.5 拒否学習データ（既知の話題×未知の属性、未学習の話題 → 「分かりません」）
        if not args.no_refusal:
            run([py, "-m", "src.refusal_data", "--output_dir", "data/", "--version", "v_refusal",
                 "--seed", str(args.seed)], "拒否学習データの生成（物差し・調整用との重複検査つき）", root)
            for suffix in (".train.jsonl", ".val.jsonl", "_corpus.txt"):
                require_file(root, f"data/v_refusal{suffix}", "拒否学習データの生成")

        # 3. 混合
        run([py, "-m", "src.mix_data", "--template_version", tpl_version,
             "--hf_dir", args.hf_dir, "--out_version", mix_version, "--seed", str(args.seed)]
             + ([] if args.no_refusal else ["--refusal_version", "v_refusal"]),
            "テンプレート＋公開データの混合", root)
        for suffix in (".train.jsonl", ".val.jsonl", "_corpus.txt"):
            require_file(root, f"data/{mix_version}{suffix}", "混合")

        # 4. トークナイザー
        run([py, "src/train_tokenizer.py", "--corpus", f"data/{mix_version}_corpus.txt",
             "--output_dir", "data/", "--vocab_size", str(args.vocab_size),
             "--model_prefix", tok_prefix, "--character_coverage", "0.9999"],
            f"トークナイザー再学習（語彙{args.vocab_size}）", root)
        require_file(root, tok_model, "トークナイザー学習")
    else:
        # 学習だけやり直す: 前回作ったデータ・トークナイザーを再利用する（作り直すと比較条件が変わるため）
        print("--reuse_data: 前回の混合データ・前処理済みデータ・トークナイザーを再利用します")
        for rel in (tok_model, f"data/{mix_version}_processed.train.jsonl",
                    f"data/{mix_version}_processed.val.jsonl"):
            require_file(root, rel, "再利用するファイルの確認")
        if not os.path.exists(os.path.join(root, f"data/{tpl_version}.val.jsonl")):
            print(f"  ⚠ data/{tpl_version}.val.jsonl が無いため、最後のsanity_checkは失敗する可能性があります")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(root, tok_model))
    actual_vocab = sp.GetPieceSize()
    print(f"  実際の語彙数: {actual_vocab}（指定: {args.vocab_size}）")
    if actual_vocab < args.vocab_size:
        print("  ※ コーパスが小さく、指定した語彙数には届きませんでした。実際の語彙数でconfigを作ります。")

    # 5. config生成
    with open(os.path.join(root, "configs/exp_cot60.yaml"), encoding="utf-8") as f:
        config_text = build_config(f.read(), args.exp_name, tok_model, actual_vocab, mix_version,
                                   lr=args.lr, epochs=args.epochs, warmup_steps=args.warmup_steps)
    config_path = f"configs/exp_{args.exp_name}.yaml"
    with open(os.path.join(root, config_path), "w", encoding="utf-8") as f:
        f.write(config_text)
    print(f"✓ {config_path} を生成しました")

    import yaml
    cfg = yaml.safe_load(config_text)
    m = cfg["model"]
    total = count_params(actual_vocab, m["d_model"], m["n_layers"], m["d_ff"], m["max_seq_length"])
    print(f"  モデル規模: d_model={m['d_model']}, n_layers={m['n_layers']}, 語彙={actual_vocab} "
          f"→ 約{total / 1e6:.1f}M パラメータ")

    if not args.reuse_data:
        # 6. 前処理
        run([py, "-m", "src.preprocess",
             "--train_path", f"data/{mix_version}.train.jsonl",
             "--val_path", f"data/{mix_version}.val.jsonl",
             "--output_dir", "data/", "--tokenizer_path", tok_model],
            "前処理（token_ids化）", root)
        require_file(root, f"data/{mix_version}_processed.train.jsonl", "前処理")
        require_file(root, f"data/{mix_version}_processed.val.jsonl", "前処理")

    n_train = sum(1 for _ in open(os.path.join(root, f"data/{mix_version}_processed.train.jsonl"), encoding="utf-8"))
    steps_per_epoch = -(-n_train // cfg["data"]["batch_size"]) // cfg["training"]["grad_accum_steps"]
    print(f"  学習設定: lr={cfg['optimizer']['lr']:g}, warmup={cfg['optimizer']['lr_warmup_steps']}, "
          f"epochs={cfg['training']['epochs']} → 約{steps_per_epoch * cfg['training']['epochs']:,}ステップ"
          f"（学習データ{n_train:,}件）")

    if args.skip_train:
        print("\n--skip_train 指定のため、学習の手前で終了します。")
        print(f"学習するには: !cd {root} && python -m src.train --config {config_path}")
        return

    # 7. 学習
    run([py, "-m", "src.train", "--config", config_path], "学習", root)
    ckpt = f"results/checkpoints/{args.exp_name}/checkpoint_best.pt"
    require_file(root, ckpt, "学習")

    # 8. 退避（checkpoint・トークナイザー・configをセットで）
    out_dir = "/kaggle/working/kaggle_dataset_mix"
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(root, ckpt), os.path.join(out_dir, f"checkpoint_best_{args.exp_name}.pt"))
    for src_rel in (tok_model, tok_model.replace(".model", ".vocab"), config_path):
        shutil.copy(os.path.join(root, src_rel), os.path.join(out_dir, os.path.basename(src_rel)))
    print(f"\n✓ 退避しました: {out_dir}")
    for name in sorted(os.listdir(out_dir)):
        print(f"   {name} ({os.path.getsize(os.path.join(out_dir, name)):,} bytes)")
    print("  ↑ このフォルダの中身を、Notebookの「Save Version」→ Output から Dataset として保存してください。")
    print("    （checkpoint・トークナイザー・configの3種はセットで保存しないと推論できません）")

    # 9. sanity_check（テンプレート側の検証データ。公開データは自由文なので完全一致では測れない）
    run([py, "-m", "src.sanity_check", "--config", config_path, "--checkpoint", ckpt,
         "--val_path", f"data/{tpl_version}.val.jsonl", "--num_samples", "50"],
        "sanity_check（テンプレート側）", root, fatal=False)

    # 10. 物差し（学習で見ていない言い回しへの対応力。前回の結果があれば差分も表示）
    yardstick_cmd = [py, "-m", "src.eval_yardstick", "--config", config_path, "--checkpoint", ckpt]
    for prev in ("logs/yardstick_p3_mix_lr3e4.json", "logs/yardstick_p2_mix_hf_11m.json"):
        if os.path.exists(os.path.join(root, prev)):
            yardstick_cmd += ["--baseline", prev]
            break
    run(yardstick_cmd, "物差し yardstick_v1", root, fatal=False)

    print("\n" + "=" * 60)
    print("🎉 混合学習パイプライン、全ステップ完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
