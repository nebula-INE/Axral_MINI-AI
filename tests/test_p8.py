"""p8（日本語Web追加・複数GPU・自動バックアップ）のテスト。

  pytest -q tests/test_p8.py            # torch が無くても動く部分（データ整形・バックアップの仕組み）
  pytest -q -s tests/test_p8.py -k ddp  # GPUセッションの最初に。torch があれば、小さなモデルで複数プロセスの学習→再開を実際に回す
"""
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- データ整形（torch不要）
def test_clean_web_filters():
    from src.import_pretrain_data import clean_web
    ja = "これは日本語のWeb記事です。" * 30
    assert clean_web(ja)
    assert clean_web("短い") is None
    assert clean_web("English only text. " * 40) is None  # 日本語が少ない
    assert clean_web(ja + " http://a http://b http://c http://d http://e http://f") is None  # URLだらけ
    assert clean_web("あ" * 40000) is None  # 長すぎる


def test_build_corpus_single_pass_and_budget(tmp_path):
    from src import pack_tokens as pt
    with open(tmp_path / "pretrain.train.jsonl", "w", encoding="utf-8") as f:
        for i in range(300):
            for src in ("wiki", "web", "code"):
                text = ("これはテスト文章です。" * 20 if src != "code" else "def f(x):\n    return x\n" * 10) + str(i)
                f.write(json.dumps({"src": src, "text": text}, ensure_ascii=False) + "\n")
    (tmp_path / "sft").mkdir()
    (tmp_path / "sft" / "hf_corpus.txt").write_text("q\na\n", encoding="utf-8")
    a = types.SimpleNamespace(seed=1, wiki_chars=20000, web_chars=10 ** 9, code_chars=3000, pretrain_dir=str(tmp_path),
                              out=str(tmp_path / "out.txt"), max_line_chars=1000, sft_dir=str(tmp_path / "sft"),
                              extra_corpus=[])
    pt.build_corpus(a)
    lines = (tmp_path / "out.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) > 100
    web_lines = sum(1 for ln in lines if "テスト文章" in ln)
    assert web_lines >= 300  # webは予算が大きいので全部入る（wikiは抜き出し）


# --------------------------------------------------------------------------- バックアップの仕組み（偽の kaggle コマンドで）
@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    created = tmp_path / "created"
    script = bindir / "kaggle"
    script.write_text(f"""#!/bin/bash
echo "$@" >> {log}
if [ "$1 $2" = "datasets version" ]; then
  if [ ! -f {created} ]; then echo "404 - Not Found"; exit 1; fi
  exit 0
fi
if [ "$1 $2" = "datasets create" ]; then touch {created}; exit 0; fi
exit 2
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    return log


def test_backup_creates_then_versions_and_skips_unchanged(tmp_path, fake_kaggle):
    from src.kaggle_sync import backup_once
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "checkpoint_0000200.pt").write_text("a")
    (ck / "checkpoint_0000400.pt").write_text("b")
    state, stage = str(tmp_path / "state.json"), str(tmp_path / "stage")
    assert backup_once(str(ck), "slug", "u", None, state, stage) is True       # 初回: version が404 → create
    assert backup_once(str(ck), "slug", "u", None, state, stage) is False      # 変化なし: 上げない
    (ck / "checkpoint_0000600.pt").write_text("c")
    assert backup_once(str(ck), "slug", "u", None, state, stage) is True       # 新しいcheckpoint: 新バージョン
    calls = fake_kaggle.read_text().splitlines()
    assert [c.split()[1] for c in calls] == ["version", "create", "version"]
    assert "-d" in calls[0].split()  # 古いバージョンを消す


def test_backup_failure_does_not_raise_and_retries(tmp_path, monkeypatch):
    from src.kaggle_sync import backup_once
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "kaggle").write_text("#!/bin/bash\necho 'boom'; exit 1\n")
    (bindir / "kaggle").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "checkpoint_0000200.pt").write_text("a")
    state = tmp_path / "state.json"
    assert backup_once(str(ck), "s", "u", None, str(state), str(tmp_path / "st")) is False
    assert not state.exists()  # 失敗したので「上げた」と記録されず、次の回に再試行される


def test_restore_picks_newest_checkpoint(tmp_path):
    from src.kaggle_sync import restore
    src = tmp_path / "in" / "axral-p8-ckpt"
    src.mkdir(parents=True)
    for n in ("checkpoint_0000200.pt", "checkpoint_0000800.pt", "pretrained_final.pt", "model_choice.json", "x_pretrain.jsonl"):
        (src / n).write_text(n)
    a = types.SimpleNamespace(input_root=str(tmp_path / "in"), dest=str(tmp_path / "d"),
                              logs_dest=str(tmp_path / "l"), slug="axral-p8-ckpt")
    restore(a)
    assert sorted(os.listdir(tmp_path / "d")) == ["checkpoint_0000800.pt", "model_choice.json", "pretrained_final.pt"]
    assert os.listdir(tmp_path / "l") == ["x_pretrain.jsonl"]


def test_bundle_roundtrip(tmp_path, fake_kaggle):
    from src import kaggle_sync as ks
    w = tmp_path / "w"
    for d in ("p7_tokens", "hf_import_p7", "Axral_MINI-AI/data"):
        (w / d).mkdir(parents=True)
    (w / "p7_tokens" / "train_wiki.bin").write_bytes(b"1234")
    (w / "hf_import_p7" / "hf_import.train.jsonl").write_text("{}")
    (w / "Axral_MINI-AI" / "data" / "spm_p7.model").write_text("m")
    (w / "Axral_MINI-AI" / "data" / "other.txt").write_text("not bundled")
    a = types.SimpleNamespace(work=str(w), root=str(w / "Axral_MINI-AI"), stage_root=str(tmp_path / "s"),
                              slug="prep", user="u")
    ks.bundle_make(a)
    tar = tmp_path / "s" / ks.BUNDLE_NAME
    assert tar.exists()
    import shutil
    shutil.rmtree(w / "p7_tokens")
    shutil.rmtree(w / "Axral_MINI-AI" / "data")
    (tmp_path / "in" / "prep").mkdir(parents=True)
    shutil.copy(tar, tmp_path / "in" / "prep" / ks.BUNDLE_NAME)
    ks.bundle_restore(types.SimpleNamespace(input_root=str(tmp_path / "in"), work=str(w)))
    assert (w / "p7_tokens" / "train_wiki.bin").read_bytes() == b"1234"
    assert (w / "Axral_MINI-AI" / "data" / "spm_p7.model").exists()
    assert not (w / "Axral_MINI-AI" / "data" / "other.txt").exists()


# --------------------------------------------------------------------------- 複数プロセスの事前学習（torch が必要）
def _make_tiny_run(tmp: Path, total_tokens: int) -> Path:
    import numpy as np
    import yaml
    tok = tmp / "tokens"
    tok.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        for src in ("wiki", "web", "code"):
            rng.integers(3, 200, size=n, dtype=np.uint16).tofile(tok / f"{split}_{src}.bin")
    (tok / "pack_meta.json").write_text(json.dumps({"vocab_size": 200}))
    cfg = {"experiment_name": "tiny", "results_dir": str(tmp / "ck"), "logs_dir": str(tmp / "logs"),
           "model": {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 256, "max_seq_length": 32,
                     "dropout": 0.0, "attention_dropout": 0.0, "residual_dropout": 0.0},
           "optimizer": {"lr": 1e-3, "min_lr": 1e-4, "warmup_fraction": 0.1},
           "data": {}, "pretrain": {"token_dir": str(tok), "mix": {"wiki": 0.4, "web": 0.4, "code": 0.2}},
           "training": {"micro_batch_size": 4, "grad_accum_steps": 2, "total_tokens": total_tokens, "amp": "off",
                        "eval_every": 5, "eval_iters": 2, "log_every": 1, "checkpoint_every": 5, "resume": True}}
    p = tmp / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_ddp_pretrain_two_processes_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    n = 2
    cfg = _make_tiny_run(tmp_path, total_tokens=4 * 2 * 32 * n * 10)  # 10ステップ
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}",
           "--module", "src.pretrain", "--config", str(cfg)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ckdir = tmp_path / "ck" / "tiny"
    assert (ckdir / "pretrained_final.pt").exists()
    assert "プロセス数) 2" in r.stdout and r.stdout.count("事前学習完了") == 1  # rank 0 だけが報告する
    log = [json.loads(ln) for ln in (tmp_path / "logs" / "tiny_pretrain.jsonl").read_text().splitlines()]
    losses = [x["loss"] for x in log if "loss" in x]
    assert len(losses) >= 5 and all(l == l for l in losses)

    # 同じ重みか（2プロセスの重みが同期していること）: 最後のcheckpointを読めて、全部有限
    ck = torch.load(ckdir / "pretrained_final.pt", map_location="cpu")
    assert all(torch.isfinite(v).all() for v in ck["model_state_dict"].values() if v.is_floating_point())

    # 再開: 総トークンを増やして同じコマンドを実行すると、続きから進む
    cfg2 = _make_tiny_run(tmp_path, total_tokens=4 * 2 * 32 * n * 16)
    r2 = subprocess.run(cmd[:-1] + [str(cfg2)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stdout[-2000:] + r2.stderr[-2000:]
    assert "再開:" in r2.stdout
