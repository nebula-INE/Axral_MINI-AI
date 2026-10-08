"""
run_p7_pipeline.py
p7: 2段階学習（事前学習 → 指示学習）のパイプライン。Kaggleのセッション時間に合わせ、段階ごとに別のセルで実行する。

  # 1. データ準備（CPUで可。ネットワーク必須。数十分〜）
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_p7_pipeline --stage prepare
  # 2. 速度測定（GPU。数分）→ 表示された推奨サイズを見て決める
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_p7_pipeline --stage bench --budget_hours 20
  # 3. 事前学習（GPU。時間予算つき。セッションが切れても、同じコマンドで続きから再開）
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_p7_pipeline --stage pretrain \\
        --d_model 512 --n_layers 8 --n_heads 8 --pretrain_hours 18
  # 4. 指示学習＋評価（事前学習の重みから開始）
  !cd /kaggle/working/Axral_MINI-AI && python -m src.run_p7_pipeline --stage sft \\
        --d_model 512 --n_layers 8 --n_heads 8

注意:
  - 3 と 4 の --d_model/--n_layers/--n_heads は必ず同じ値にする（重みの形が合わなくなる）。
  - 各段階は、前の段階の生成物を確認してから始める（無ければ止まる）。
  - 指示学習の評価（sanity_check / 物差し）は、手順つき算数が途中で切れないよう max_new_tokens=384 で行う。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

TPL_VERSION, REF_VERSION, MIX_VERSION = "v_tpl_p7", "v_refusal_p7", "v_p7mix"
TOK_PREFIX = "spm_p7"


def run(cmd: list[str], step: str, cwd: str, fatal: bool = True) -> bool:
    print(f"\n{'=' * 60}\n▶ {step}\n{'=' * 60}\n$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        msg = f"❌ ステップ「{step}」が失敗しました（終了コード {r.returncode}）。"
        if fatal:
            raise RuntimeError(msg + "ここで停止します。上のログを確認してください。")
        print("⚠️ " + msg + "（失敗しても続行します）")
        return False
    print(f"✓ {step} 完了")
    return True


def require(root: str, rel: str, step: str) -> None:
    full = rel if os.path.isabs(rel) else os.path.join(root, rel)
    if not os.path.exists(full):
        raise FileNotFoundError(f"❌ 「{step}」に必要なファイルがありません: {full}\n前の段階が完了しているか確認してください。")
    print(f"  ✓ 確認: {full} ({os.path.getsize(full):,} bytes)")


def write_yaml(path: str, cfg: dict, header: str) -> None:
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"✓ {path} を生成しました")


def model_cfg(a) -> dict:
    return {"d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads,
            "d_ff": a.d_ff or 4 * a.d_model, "max_seq_length": a.block}


def main() -> None:
    ap = argparse.ArgumentParser(description="p7: 事前学習→指示学習のパイプライン")
    ap.add_argument("--stage", required=True, choices=["prepare", "bench", "pretrain", "sft"])
    ap.add_argument("--project_root", default="/kaggle/working/Axral_MINI-AI")
    ap.add_argument("--pretrain_dir", default="/kaggle/working/p7_data")
    ap.add_argument("--token_dir", default="/kaggle/working/p7_tokens")
    ap.add_argument("--sft_dir", default="/kaggle/working/hf_import_p7")
    ap.add_argument("--existing_hf_dir", default="/kaggle/working/hf_import",
                    help="既存の指示データ（dolly/oasst）。あれば指示データに加える")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=42)
    # prepare
    ap.add_argument("--template_samples", type=int, default=60000)
    ap.add_argument("--arith_ranges", choices=["default", "wide"], default="default")
    ap.add_argument("--import_args", nargs=argparse.REMAINDER, default=[],
                    help="import_pretrain_data にそのまま渡す引数（動作確認: --import_args --wiki_max_docs 2000 ...）")
    # モデル・事前学習
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=0, help="0なら 4×d_model")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--budget_hours", type=float, default=20.0, help="bench: 事前学習に使えるGPU時間")
    ap.add_argument("--pretrain_hours", type=float, default=18.0)
    ap.add_argument("--total_tokens", type=int, default=0, help="指定すると、時間予算の代わりに総トークン数で学習の長さを決める")
    ap.add_argument("--micro_bs", type=int, default=32)
    ap.add_argument("--batch_tokens", type=int, default=262144, help="1回の更新で使うトークン数")
    ap.add_argument("--lr", type=float, default=0.0, help="0なら d_model に応じた既定値（<=512: 6e-4, それ以上: 4e-4）")
    ap.add_argument("--mix_wiki", type=float, default=0.7)
    ap.add_argument("--mix_code", type=float, default=0.3)
    ap.add_argument("--exp_name", default="p7_pretrain")
    # sft
    ap.add_argument("--sft_exp_name", default="p7_sft")
    ap.add_argument("--sft_lr", type=float, default=1e-4)
    ap.add_argument("--sft_epochs", type=int, default=3)
    ap.add_argument("--pretrained", default=None, help="事前学習の重み（既定: results/checkpoints/{exp_name}/pretrained_final.pt）")
    ap.add_argument("--skip_eval", action="store_true")
    a = ap.parse_args()

    root = a.project_root
    os.chdir(root)
    assert os.path.exists("src") and os.path.exists("configs"), f"プロジェクトルートが不正です: {root}"
    py = sys.executable
    tok_model = f"data/{TOK_PREFIX}.model"

    # ------------------------------------------------------------------ prepare
    if a.stage == "prepare":
        imp = [py, "-m", "src.import_pretrain_data", "--out_dir", a.pretrain_dir, "--sft_dir", a.sft_dir]
        if os.path.exists(os.path.join(a.existing_hf_dir, "hf_import.train.jsonl")):
            imp += ["--include_existing_hf", a.existing_hf_dir]
        run(imp + a.import_args, "事前学習データ・指示データの取得と整形（要インターネット）", root)
        require(root, os.path.join(a.pretrain_dir, "pretrain.train.jsonl"), "データ取得")
        require(root, os.path.join(a.sft_dir, "hf_import.train.jsonl"), "データ取得")

        run([py, "-m", "src.paraphrase_bank", "--check"], "言い換えバンクの検査（物差しとの重複チェック）", root)
        run([py, "src/generate_data.py", "--output_dir", "data/", "--num_samples", str(a.template_samples),
             "--seed", str(a.seed), "--cot_ratio", "0.6", "--version", TPL_VERSION,
             "--arith_style", "stepwise", "--arith_ranges", a.arith_ranges],
            "テンプレートデータ生成（手順つき算数）", root)
        run([py, "-m", "src.refusal_data", "--output_dir", "data/", "--version", REF_VERSION, "--seed", str(a.seed)],
            "拒否学習データの生成", root)
        for v in (TPL_VERSION, REF_VERSION):
            for suf in (".train.jsonl", ".val.jsonl", "_corpus.txt"):
                require(root, f"data/{v}{suf}", "テンプレート/拒否データ生成")

        run([py, "-m", "src.pack_tokens", "corpus", "--pretrain_dir", a.pretrain_dir, "--sft_dir", a.sft_dir,
             "--extra_corpus", f"data/{TPL_VERSION}_corpus.txt", f"data/{REF_VERSION}_corpus.txt",
             "--out", "data/p7_tok_corpus.txt", "--seed", str(a.seed)], "トークナイザー用コーパスの作成", root)
        run([py, "src/train_tokenizer.py", "--corpus", "data/p7_tok_corpus.txt", "--output_dir", "data/",
             "--vocab_size", str(a.vocab_size), "--model_prefix", TOK_PREFIX, "--character_coverage", "0.9999",
             "--code_aware", "--input_sentence_size", "2000000"], f"トークナイザー学習（語彙{a.vocab_size}・コード対応）", root)
        require(root, tok_model, "トークナイザー学習")
        run([py, "-m", "src.pack_tokens", "pack", "--pretrain_dir", a.pretrain_dir, "--tokenizer", tok_model,
             "--out_dir", a.token_dir], "事前学習データのトークン化", root)
        require(root, os.path.join(a.token_dir, "pack_meta.json"), "トークン化")
        print("\n✅ prepare 完了。次は --stage bench で速度を測ってください。")

    # ------------------------------------------------------------------ bench
    elif a.stage == "bench":
        require(root, os.path.join(a.token_dir, "pack_meta.json"), "速度測定")
        import json
        meta = json.load(open(os.path.join(a.token_dir, "pack_meta.json"), encoding="utf-8"))
        avail = sum(v["tokens"] for k, v in meta["files"].items() if k.startswith("train_"))
        print(f"用意できる学習データ: {avail:,}トークン（語彙 {meta['vocab_size']}）")
        run([py, "-m", "src.bench_throughput", "--budget_hours", str(a.budget_hours),
             "--vocab_size", str(meta["vocab_size"]), "--block", str(a.block), "--micro_bs", str(a.micro_bs),
             "--available_tokens", str(avail)], "学習速度の測定", root)

    # ------------------------------------------------------------------ pretrain
    elif a.stage == "pretrain":
        require(root, os.path.join(a.token_dir, "pack_meta.json"), "事前学習")
        require(root, tok_model, "事前学習")
        lr = a.lr or (6e-4 if a.d_model <= 512 else 4e-4)
        accum = max(1, round(a.batch_tokens / (a.micro_bs * a.block)))
        training = {"micro_batch_size": a.micro_bs, "grad_accum_steps": accum, "max_grad_norm": 1.0, "amp": "auto",
                    "eval_every": 500, "eval_iters": 40, "log_every": 50, "checkpoint_every": 500, "resume": True}
        if a.total_tokens:
            training["total_tokens"] = a.total_tokens
        else:
            training["time_budget_hours"] = a.pretrain_hours
        cfg = {"experiment_name": a.exp_name,
               "model": {**model_cfg(a), "dropout": 0.0, "attention_dropout": 0.0, "residual_dropout": 0.0},
               "optimizer": {"lr": lr, "min_lr": lr * 0.1, "warmup_fraction": 0.02, "beta1": 0.9, "beta2": 0.95,
                             "eps": 1e-8, "weight_decay": 0.1},
               "data": {"tokenizer_path": tok_model},
               "pretrain": {"token_dir": a.token_dir, "mix": {"wiki": a.mix_wiki, "code": a.mix_code}},
               "training": training,
               "kaggle": {"max_session_time_minutes": 540},
               "results_dir": "results/checkpoints", "logs_dir": "logs"}
        cfg_path = f"configs/exp_{a.exp_name}.yaml"
        write_yaml(cfg_path, cfg, "# p7 事前学習。run_p7_pipeline.py が自動生成")
        run([py, "-m", "src.pretrain", "--config", cfg_path], "事前学習", root)
        final = f"results/checkpoints/{a.exp_name}/pretrained_final.pt"
        if os.path.exists(os.path.join(root, final)):
            out_dir = "/kaggle/working/kaggle_dataset_p7"
            os.makedirs(out_dir, exist_ok=True)
            for src_rel in (final, tok_model, tok_model.replace(".model", ".vocab"), cfg_path):
                shutil.copy(os.path.join(root, src_rel), os.path.join(out_dir, os.path.basename(src_rel)))
            print(f"\n✓ 退避しました: {out_dir}（重み・トークナイザー・configをセットで保存してください）")
        else:
            print("\n⚠ 事前学習はまだ完了していません（セッション上限で中断）。同じコマンドをもう一度実行すると続きから再開します。")

    # ------------------------------------------------------------------ sft
    else:
        pretrained = a.pretrained or f"results/checkpoints/{a.exp_name}/pretrained_final.pt"
        require(root, pretrained, "指示学習")
        require(root, tok_model, "指示学習")
        for suf in (".train.jsonl", ".val.jsonl"):
            require(root, f"data/{TPL_VERSION}{suf}", "指示学習")
        run([py, "-m", "src.mix_data", "--template_version", TPL_VERSION, "--hf_dir", a.sft_dir,
             "--out_version", MIX_VERSION, "--seed", str(a.seed), "--refusal_version", REF_VERSION],
            "テンプレート＋公開データ＋拒否データの混合", root)
        run([py, "-m", "src.preprocess", "--train_path", f"data/{MIX_VERSION}.train.jsonl",
             "--val_path", f"data/{MIX_VERSION}.val.jsonl", "--output_dir", "data/", "--tokenizer_path", tok_model],
            "前処理（新しいトークナイザーでtoken_ids化）", root)
        require(root, f"data/{MIX_VERSION}_processed.train.jsonl", "前処理")

        import json
        meta = json.load(open(os.path.join(a.token_dir, "pack_meta.json"), encoding="utf-8"))
        cfg = {"experiment_name": a.sft_exp_name,
               "model": {**model_cfg(a), "dropout": 0.1, "attention_dropout": 0.1, "residual_dropout": 0.1},
               "optimizer": {"lr": a.sft_lr, "lr_warmup_steps": 200, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
                             "weight_decay": 1e-2},
               "data": {"train_path": f"data/{MIX_VERSION}_processed.train.jsonl",
                        "val_path": f"data/{MIX_VERSION}_processed.val.jsonl",
                        "tokenizer_path": tok_model, "vocab_size": meta["vocab_size"],
                        "batch_size": 16, "eval_batch_size": 16},
               "training": {"epochs": a.sft_epochs, "grad_accum_steps": 4, "max_grad_norm": 1.0,
                            "label_smoothing": 0.1, "eval_frequency": 500, "checkpoint_frequency": 1000,
                            "early_stopping_patience": 5, "resume_from_checkpoint": True, "amp": "auto",
                            "init_from": pretrained},
               "kaggle": {"max_session_time_minutes": 540},
               "wandb": {"enabled": False, "project": "vose-initial-llm"},
               "results_dir": "results/checkpoints", "logs_dir": "logs"}
        cfg_path = f"configs/exp_{a.sft_exp_name}.yaml"
        write_yaml(cfg_path, cfg, "# p7 指示学習（事前学習の重みから開始）。run_p7_pipeline.py が自動生成")
        run([py, "-m", "src.train", "--config", cfg_path], "指示学習", root)
        ckpt = f"results/checkpoints/{a.sft_exp_name}/checkpoint_best.pt"
        require(root, ckpt, "指示学習")

        out_dir = "/kaggle/working/kaggle_dataset_p7"
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(os.path.join(root, ckpt), os.path.join(out_dir, f"checkpoint_best_{a.sft_exp_name}.pt"))
        for src_rel in (tok_model, tok_model.replace(".model", ".vocab"), cfg_path):
            shutil.copy(os.path.join(root, src_rel), os.path.join(out_dir, os.path.basename(src_rel)))
        print(f"\n✓ 退避しました: {out_dir}")

        if not a.skip_eval:
            run([py, "-m", "src.sanity_check", "--config", cfg_path, "--checkpoint", ckpt,
                 "--val_path", f"data/{TPL_VERSION}.val.jsonl", "--num_samples", "50", "--max_new_tokens", "384"],
                "sanity_check（max_new_tokens=384）", root, fatal=False)
            cmd = [py, "-m", "src.eval_yardstick", "--config", cfg_path, "--checkpoint", ckpt,
                   "--max_new_tokens", "384", "--out", f"logs/yardstick_{a.sft_exp_name}.json"]
            for prev in ("logs/yardstick_p5_refusal.json", "logs/yardstick_p3_mix_lr3e4.json"):
                if os.path.exists(os.path.join(root, prev)):
                    cmd += ["--baseline", prev]
                    break
            run(cmd, "物差し yardstick_v1（max_new_tokens=384）", root, fatal=False)
        print("\n🎉 p7 完了")


if __name__ == "__main__":
    main()
