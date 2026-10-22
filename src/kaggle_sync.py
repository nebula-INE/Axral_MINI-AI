"""
kaggle_sync.py
Kaggle のセッションが切れても学習結果を失わないための、バックアップ／復元／セッション間の受け渡し。

  # 1) 学習中の自動バックアップ（別プロセスで動かす。run_p8_pipeline の pretrain ステージが自動で起動する）
  python -m src.kaggle_sync loop --watch results/checkpoints/p8_pretrain --slug axral-p8-ckpt --user sekoia29 --interval 20

  # 2) 新しいセッションの最初に、バックアップから復元（Dataset を Add Input しておく）
  python -m src.kaggle_sync restore --dest results/checkpoints/p8_pretrain

  # 3) CPUセッションで作ったデータ（トークン列・トークナイザー等）を、GPUセッションへ渡す
  python -m src.kaggle_sync bundle-make --slug axral-p8-prep --user sekoia29      # CPUセッションで
  python -m src.kaggle_sync bundle-restore                                          # GPUセッションで（Add Input 済み）

仕組み:
  - バックアップは、同じ Dataset の「新しいバージョン」として上げる（古いバージョンは消す）。
    最初の1回だけ Dataset を新規作成する。作成直後の処理中は新しいバージョンを上げられないことがあるが、
    失敗しても学習は止めず、次の回で再試行する。
  - 認証は Kaggle CLI のもの（~/.kaggle/kaggle.json または 環境変数 KAGGLE_USERNAME / KAGGLE_KEY）。
    無ければ、Notebook の Add-ons → Secrets に KAGGLE_USERNAME / KAGGLE_KEY を登録しておくと自動で読む。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

STATE_NAME = ".kaggle_sync_state.json"


# --------------------------------------------------------------------------- 認証・CLI
def ensure_auth() -> bool:
    """Kaggle CLI が使える認証があるかを確認する。無ければ Notebook の Secrets から環境変数に入れる。"""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        return True
    try:
        from kaggle_secrets import UserSecretsClient  # Kaggle Notebook の中だけにある
        c = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = c.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = c.get_secret("KAGGLE_KEY")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"⚠ Kaggle の認証が見つかりません（{type(e).__name__}）。"
              f"Add-ons → Secrets に KAGGLE_USERNAME / KAGGLE_KEY を登録してください。")
        return False


def _kaggle(args: list[str], timeout: int = 3600) -> tuple[int, str]:
    try:
        r = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "kaggle コマンドが見つかりません（pip install kaggle）"
    except subprocess.TimeoutExpired:
        return 124, "タイムアウト"
    return r.returncode, (r.stdout + r.stderr).strip()


def _stage(files: dict[str, str], stage_dir: Path) -> None:
    """アップロード用の平らなフォルダを作る（サブフォルダは Kaggle に上がらないので使わない）。同じディスク上ならリンクで速く。"""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for name, src in files.items():
        dst = stage_dir / name
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def push(files: dict[str, str], slug: str, user: str, title: str, message: str, stage_dir: str) -> bool:
    """files = {Dataset内のファイル名: 元のパス}。Dataset が無ければ作成、あれば新バージョンとして上げる。"""
    if not files:
        print("  （上げるファイルがありません）")
        return False
    if not ensure_auth():
        return False
    st = Path(stage_dir)
    _stage(files, st)
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (st / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    code, out = _kaggle(["datasets", "version", "-p", str(st), "-m", message, "-d"])
    if code != 0 and any(k in out.lower() for k in ("404", "not found", "does not exist", "could not find")):
        print("  Dataset がまだ無いので新規作成します")
        code, out = _kaggle(["datasets", "create", "-p", str(st)])
    shutil.rmtree(st, ignore_errors=True)
    if code == 0:
        size = sum(os.path.getsize(p) for p in files.values()) / 1e6
        print(f"  ✓ バックアップ完了: {slug}（{size:,.0f}MB / {time.time() - t0:.0f}秒）")
        return True
    print(f"  ⚠ バックアップに失敗しました（次の回に再試行します）: {out[-300:]}")
    return False


# --------------------------------------------------------------------------- チェックポイントのバックアップ
def collect_ckpt_files(watch: str, logs_glob: str | None = None) -> dict[str, str]:
    """最新の再開用 checkpoint 1つ・完成した重み・ログを、Dataset に上げる名前で返す。"""
    files: dict[str, str] = {}
    cks = sorted(glob.glob(os.path.join(watch, "checkpoint_[0-9]*.pt")))
    if cks:
        files[os.path.basename(cks[-1])] = cks[-1]
    for name in ("pretrained_final.pt", "checkpoint_best.pt", "model_choice.json"):  # model_choice: 再開時に重みの形を合わせるため
        p = os.path.join(watch, name)
        if os.path.exists(p):
            files[name] = p
    for p in glob.glob(logs_glob) if logs_glob else []:
        files[os.path.basename(p)] = p
    return files


def _signature(files: dict[str, str]) -> list:
    sig = []
    for name, p in sorted(files.items()):
        st = os.stat(p)
        sig.append([name, st.st_size, int(st.st_mtime)])
    return sig


def backup_once(watch: str, slug: str, user: str, logs_glob: str | None, state_path: str, stage_dir: str) -> bool:
    files = collect_ckpt_files(watch, logs_glob)
    ckpts = [n for n in files if n.startswith("checkpoint_") or n.endswith("final.pt") or n == "checkpoint_best.pt"]
    if not ckpts:
        return False
    sig = _signature({n: p for n, p in files.items() if n in ckpts})  # ログだけの変化では上げ直さない
    try:
        last = json.load(open(state_path)).get("sig")
    except Exception:  # noqa: BLE001
        last = None
    if last == sig:
        return False
    newest = sorted(ckpts)[-1]
    ok = push(files, slug, user, f"{slug}", f"{newest} ({time.strftime('%m-%d %H:%M')})", stage_dir)
    if ok:
        json.dump({"sig": sig}, open(state_path, "w"))
    return ok


def loop(a) -> None:
    print(f"[バックアップ] {a.watch} を {a.interval}分ごとに {a.user}/{a.slug} へ（Ctrl+C か親プロセス終了まで）", flush=True)
    state = os.path.join(a.stage_root, STATE_NAME)
    os.makedirs(a.stage_root, exist_ok=True)
    stage = os.path.join(a.stage_root, "_ckpt_stage")
    while True:
        try:
            backup_once(a.watch, a.slug, a.user, a.logs_glob, state, stage)
        except Exception as e:  # noqa: BLE001  バックアップの失敗で学習を止めない
            print(f"  ⚠ バックアップ中に例外: {type(e).__name__}: {e}", flush=True)
        if a.once:
            return
        time.sleep(a.interval * 60)


# --------------------------------------------------------------------------- 復元
def restore(a) -> None:
    """/kaggle/input 以下のバックアップから、最新の checkpoint と完成した重みを dest に戻す。"""
    found: dict[str, list[str]] = {}
    for pat in ("checkpoint_[0-9]*.pt", "pretrained_final.pt", "checkpoint_best.pt", "model_choice.json", "*_pretrain.jsonl"):
        for p in glob.glob(os.path.join(a.input_root, "**", pat), recursive=True):
            if a.slug and a.slug not in p:
                continue
            found.setdefault(pat, []).append(p)
    os.makedirs(a.dest, exist_ok=True)
    n = 0
    for pat, paths in found.items():
        if pat == "checkpoint_[0-9]*.pt":  # 一番新しい step のものだけ
            paths = [sorted(paths, key=lambda p: os.path.basename(p))[-1]]
        for p in paths:
            dst = os.path.join(a.logs_dest if p.endswith(".jsonl") else a.dest, os.path.basename(p))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(p):
                shutil.copy2(p, dst)
                print(f"  復元: {p} → {dst}")
                n += 1
    print(f"✓ {n}ファイルを復元しました" if n else "（復元できるバックアップは見つかりませんでした。最初から始めます）")


# --------------------------------------------------------------------------- CPU→GPU セッションの受け渡し
BUNDLE_NAME = "prep_small.tar"  # 小さいファイル（指示データ・トークナイザー等）だけをまとめたtar。大きい .bin は tar に入れず、そのまま上げる


def bundle_items(work: str, root: str) -> list[tuple[str, str]]:
    """小さいもの: (tarの中の名前, 実パス)。/kaggle/working からの相対パスで入れる。"""
    items = []
    paths = [os.path.join(work, "hf_import_p7"), os.path.join(work, "hf_import")]
    for pat in ("spm_p7.model", "spm_p7.vocab", "v_tpl_p7*", "v_refusal_p7*", "sources.csv"):
        paths += glob.glob(os.path.join(root, "data", pat))
    for p in paths:
        if os.path.exists(p):
            items.append((os.path.relpath(p, work), p))
    return items


def bundle_make(a) -> None:
    """トークン列(.bin)と pack_meta.json は平らなファイルのまま上げる（tarに複製するとディスクが足りなくなる。
    アップロード用の置き場はハードリンクなので、追加のディスクを使わない）。"""
    items = bundle_items(a.work, a.root)
    os.makedirs(a.stage_root, exist_ok=True)
    small = os.path.join(a.stage_root, BUNDLE_NAME)
    print("束ねるもの（小さいもの → " + BUNDLE_NAME + "）:")
    with tarfile.open(small, "w") as tf:
        for arc, p in items:
            size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)
            print(f"  {size / 1e6:>9,.1f} MB  {arc}")
            tf.add(p, arcname=arc)
    files = {BUNDLE_NAME: small}
    tok_dir = os.path.join(a.work, "p7_tokens")
    for p in sorted(glob.glob(os.path.join(tok_dir, "*.bin")) + glob.glob(os.path.join(tok_dir, "pack_meta.json"))):
        files[os.path.basename(p)] = p
        print(f"  {os.path.getsize(p) / 1e6:>9,.1f} MB  {os.path.basename(p)}（そのまま）")
    if not push(files, a.slug, a.user, a.slug, f"prep bundle {time.strftime('%m-%d %H:%M')}",
                os.path.join(a.stage_root, "_bundle_stage")):
        print("⚠ アップロードに失敗しました。上のエラーを確認して、もう一度 bundle-make を実行してください（データは消えていません）")
        sys.exit(1)
    os.remove(small)


def bundle_restore(a) -> None:
    hits = sorted(glob.glob(os.path.join(a.input_root, "**", BUNDLE_NAME), recursive=True), key=os.path.getmtime)
    if not hits:
        sys.exit(f"❌ {BUNDLE_NAME} が {a.input_root} に見つかりません。Add Input でデータ受け渡し用の Dataset を追加してください")
    src_dir = os.path.dirname(hits[-1])
    print(f"展開: {hits[-1]} → {a.work}")
    with tarfile.open(hits[-1]) as tf:
        try:
            tf.extractall(a.work, filter="data")  # Python 3.12+: 安全な展開
        except TypeError:
            tf.extractall(a.work)
    # 大きい .bin は、コピーせずリンクを張る（/kaggle/working のディスクを節約。読み取りだけなので問題ない）
    tok_dir = os.path.join(a.work, "p7_tokens")
    os.makedirs(tok_dir, exist_ok=True)
    n = 0
    for p in glob.glob(os.path.join(src_dir, "*.bin")) + glob.glob(os.path.join(src_dir, "pack_meta.json")):
        dst = os.path.join(tok_dir, os.path.basename(p))
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(p, dst)
        except OSError:
            shutil.copy2(p, dst)
        n += 1
    print(f"✓ 完了（トークン列 {n}ファイル）")


# --------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle のバックアップ・復元・セッション間の受け渡し")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--user", default="sekoia29")
        p.add_argument("--stage_root", default="/kaggle/working/_sync")

    p = sub.add_parser("loop", help="定期バックアップ")
    common(p)
    p.add_argument("--watch", required=True)
    p.add_argument("--slug", default="axral-p8-ckpt")
    p.add_argument("--interval", type=float, default=20, help="分")
    p.add_argument("--logs_glob", default=None)
    p.add_argument("--once", action="store_true", help="1回だけ上げて終了")
    p.set_defaults(fn=loop)

    p = sub.add_parser("restore", help="バックアップから復元")
    p.add_argument("--input_root", default="/kaggle/input")
    p.add_argument("--dest", required=True)
    p.add_argument("--logs_dest", default="logs")
    p.add_argument("--slug", default="", help="パスにこの文字列を含むものだけ（空なら全部）")
    p.set_defaults(fn=restore)

    p = sub.add_parser("bundle-make", help="CPUセッション: データを束ねてDatasetに上げる")
    common(p)
    p.add_argument("--slug", default="axral-p8-prep")
    p.add_argument("--work", default="/kaggle/working")
    p.add_argument("--root", default="/kaggle/working/Axral_MINI-AI")
    p.set_defaults(fn=bundle_make)

    p = sub.add_parser("bundle-restore", help="GPUセッション: 束ねたデータを展開")
    p.add_argument("--input_root", default="/kaggle/input")
    p.add_argument("--work", default="/kaggle/working")
    p.set_defaults(fn=bundle_restore)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
