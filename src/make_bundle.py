"""
make_bundle.py
学習の成果物（checkpoint・トークナイザー・config）を1つのzipにまとめる。

Kaggleノートブックのセルで:
  # 推論に必要な最小セット（既定）
  !cd /kaggle/working/Axral_MINI-AI && python -m src.make_bundle

  # データ・ログ・ソースコードも全部入り
  !cd /kaggle/working/Axral_MINI-AI && python -m src.make_bundle --full

出力: /kaggle/working/axral_mix_bundle.zip
  Notebook右側の Output に表示される。ダウンロード、または Save Version で保存できる。

zipの中身:
  既定:   checkpoint / トークナイザー(.model, .vocab) / config / README.txt
  --full: 上記 + data/（混合データ・前処理済み・検証データ）, hf_import/, logs/, src/, configs/
"""
from __future__ import annotations

import argparse
import os
import zipfile

_README = """\
Axral_MINI-AI 成果物バンドル（{exp}）

【中身】
  checkpoint_best_{exp}.pt   学習済みモデル
  spm_mix_16k.model/.vocab   トークナイザー（checkpointとセットで必要。語彙が違うと動かない）
  {cfg}   学習時のconfig

【推論のしかた（Kaggleノートブックのセル。パスは展開先に合わせて書き換える）】
  import sys; sys.path.insert(0, "/kaggle/working/Axral_MINI-AI")
  import torch
  from src.infer import load_model_and_tokenizer
  from src.chat import run_chat
  from src.utils import load_config

  B = "/kaggle/working/bundle"   # このzipを展開した場所
  config = load_config(B + "/{cfg}")
  config["data"]["tokenizer_path"] = B + "/spm_mix_16k.model"   # ← configのパスを展開先に差し替える
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, sp, tok_meta = load_model_and_tokenizer(
      config, B + "/checkpoint_best_{exp}.pt", device)
  run_chat(model, sp, tok_meta, device, checkpoint_path="{exp}", use_web_search=True)

※ コード（src/）は別途必要です。--full で作ったzipには src/ と configs/ も入っています。
"""


def _add(zf: zipfile.ZipFile, path: str, arcname: str) -> None:
    # .pt/.model は既に圧縮されにくいので無圧縮で格納（時間短縮）
    stored = path.endswith((".pt", ".model"))
    zf.write(path, arcname, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    print(f"  + {arcname} ({os.path.getsize(path):,} bytes)")


def _add_dir(zf: zipfile.ZipFile, root: str, rel: str) -> None:
    base = os.path.join(root, rel)
    if not os.path.isdir(base):
        print(f"  - {rel}/ は見つかりません（スキップ）")
        return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", ".pytest_cache")]
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, name)
            _add(zf, full, os.path.join("full", os.path.relpath(full, root)))


def main():
    parser = argparse.ArgumentParser(description="成果物をzipにまとめる")
    parser.add_argument("--project_root", default="/kaggle/working/Axral_MINI-AI")
    parser.add_argument("--exp_name", default="p3_mix_lr3e4",
                        help="前回の実験は p2_mix_hf_11m（configは configs/exp_mix.yaml）")
    parser.add_argument("--tokenizer_prefix", default="spm_mix_16k")
    parser.add_argument("--config", default=None,
                        help="既定は configs/exp_{exp_name}.yaml")
    parser.add_argument("--hf_dir", default="/kaggle/working/hf_import")
    parser.add_argument("--out", default="/kaggle/working/axral_mix_bundle.zip")
    parser.add_argument("--full", action="store_true", help="data/・logs/・src/なども含める")
    args = parser.parse_args()

    root, exp = args.project_root, args.exp_name
    if args.config is None:
        args.config = f"configs/exp_{exp}.yaml"
    ckpt = os.path.join(root, f"results/checkpoints/{exp}/checkpoint_best.pt")
    if not os.path.exists(ckpt):  # 退避フォルダ側のコピーにフォールバック
        alt = f"/kaggle/working/kaggle_dataset_mix/checkpoint_best_{exp}.pt"
        ckpt = alt if os.path.exists(alt) else ckpt

    required = {
        "checkpoint": ckpt,
        "トークナイザー(.model)": os.path.join(root, f"data/{args.tokenizer_prefix}.model"),
        "トークナイザー(.vocab)": os.path.join(root, f"data/{args.tokenizer_prefix}.vocab"),
        "config": os.path.join(root, args.config),
    }
    missing = [f"{k}: {v}" for k, v in required.items() if not os.path.exists(v)]
    if missing:
        raise SystemExit("❌ 必須ファイルが見つかりません（セッションが切れて消えた可能性があります）:\n  "
                         + "\n  ".join(missing))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"zipを作成します: {args.out}")
    with zipfile.ZipFile(args.out, "w") as zf:
        _add(zf, ckpt, f"checkpoint_best_{exp}.pt")
        _add(zf, required["トークナイザー(.model)"], f"{args.tokenizer_prefix}.model")
        _add(zf, required["トークナイザー(.vocab)"], f"{args.tokenizer_prefix}.vocab")
        _add(zf, required["config"], os.path.basename(args.config))
        zf.writestr("README.txt", _README.format(exp=exp, cfg=os.path.basename(args.config)))
        print("  + README.txt")

        if args.full:
            for rel in ("data", "logs", "src", "configs"):
                _add_dir(zf, root, rel)
            _add_dir_abs = args.hf_dir
            if os.path.isdir(_add_dir_abs):
                for name in sorted(os.listdir(_add_dir_abs)):
                    _add(zf, os.path.join(_add_dir_abs, name), os.path.join("full", "hf_import", name))

    print(f"\n✅ 完了: {args.out} ({os.path.getsize(args.out):,} bytes)")
    print("Notebook右側の Output に表示されます。ダウンロード、または Save Version で保存してください。")


if __name__ == "__main__":
    main()
