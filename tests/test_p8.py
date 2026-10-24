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
    assert clean_web(ja + "wwwwwwwwwwwwwwww") is None  # 繰り返し（掲示板のコメント）
    assert clean_web(ja + "もっと見る" * 3) is None  # 定型文の切れ端


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
    assert not (tmp_path / "s" / ks.BUNDLE_NAME).exists()  # アップロード後に消える（ディスクを空ける）
    staged = fake_kaggle.read_text()
    assert "datasets create" in staged
    # Dataset の中身を再現: 小さいtar + 平らな .bin（ハードリンクの置き場は消えているので作り直す）
    import shutil
    (tmp_path / "in" / "prep").mkdir(parents=True)
    with __import__("tarfile").open(tmp_path / "in" / "prep" / ks.BUNDLE_NAME, "w") as tf:
        for arc, p in ks.bundle_items(str(w), str(w / "Axral_MINI-AI")):
            tf.add(p, arcname=arc)
    shutil.copy(w / "p7_tokens" / "train_wiki.bin", tmp_path / "in" / "prep" / "train_wiki.bin")
    shutil.rmtree(w / "p7_tokens")
    shutil.rmtree(w / "Axral_MINI-AI" / "data")
    ks.bundle_restore(types.SimpleNamespace(input_root=str(tmp_path / "in"), work=str(w)))
    assert (w / "p7_tokens" / "train_wiki.bin").read_bytes() == b"1234"
    assert (w / "Axral_MINI-AI" / "data" / "spm_p7.model").exists()
    assert not (w / "Axral_MINI-AI" / "data" / "other.txt").exists()


# --------------------------------------------------------------------------- パイプラインの組み立て（学習は実行しない）
def test_pipeline_pretrain_stage_builds_commands(tmp_path, monkeypatch):
    """pretrain ステージ: 設定ファイル・model_choice.json の作成、2GPU用の起動コマンド、バックアップの起動と終了。"""
    import yaml
    from src import run_p7_pipeline as rp
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "data").mkdir()
    (root / "data" / "spm_p7.model").write_text("x")
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "pack_meta.json").write_text("{}")
    calls, pops = [], []
    monkeypatch.setattr(rp, "run", lambda cmd, step, cwd, fatal=True: (calls.append(cmd), True)[1])

    class FakeP:
        def __init__(self, cmd, **kw):
            pops.append(list(cmd))

        def terminate(self):
            pops.append("TERMINATED")

        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(rp.subprocess, "Popen", FakeP)
    monkeypatch.setattr(rp, "gpu_count", lambda n: 2 if n == 0 else n)
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "pretrain", "--project_root", str(root), "--token_dir", str(tok),
                                     "--d_model", "640", "--n_layers", "10", "--n_heads", "10", "--pretrain_hours", "7.5"])
    rp.main()
    cfg = yaml.safe_load((root / "configs" / "exp_p8_pretrain.yaml").read_text())
    assert cfg["training"]["grad_accum_steps"] == 8 and cfg["pretrain"]["mix"]["web"] == 0.4
    choice = json.loads((root / "results" / "checkpoints" / "p8_pretrain" / "model_choice.json").read_text())
    assert choice["d_model"] == 640 and choice["n_layers"] == 10
    assert any("--nproc_per_node=2" in " ".join(c) and "src.pretrain" in c for c in calls)
    assert "loop" in pops[0] and "src.kaggle_sync" in pops[0] and pops[-1] == "TERMINATED"  # バックアップを学習の前に起動し、後で止める


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


def test_save_ckpt_frees_old_when_disk_low_and_skips_when_hopeless(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import pretrain as P
    d = tmp_path / "ck"
    d.mkdir()
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for n in (1, 2):
        assert P.save_ckpt(str(d / f"checkpoint_{n:07d}.pt"), model, opt, None, n, 0, 0.0, None)
    size = (d / "checkpoint_0000002.pt").stat().st_size
    # 空きが足りない: 古い方(1)を先に消してから、3を保存する
    free = {"v": size}                      # 1個分しか空きがない状態を作る
    def fake_free(_p):
        v = free["v"]
        free["v"] += size                  # 1個消すごとに空きが増える
        return v
    monkeypatch.setattr(P, "_free_bytes", fake_free)
    assert P.save_ckpt(str(d / "checkpoint_0000003.pt"), model, opt, None, 3, 0, 0.0, None)
    names = sorted(x.name for x in d.glob("checkpoint_*.pt"))
    assert "checkpoint_0000003.pt" in names and "checkpoint_0000001.pt" not in names
    # どうやっても足りない: 保存を見送るが例外は出さず、最新のcheckpointは残る
    monkeypatch.setattr(P, "_free_bytes", lambda _p: 0)
    assert P.save_ckpt(str(d / "checkpoint_0000004.pt"), model, opt, None, 4, 0, 0.0, None) is False
    assert (d / "checkpoint_0000003.pt").exists() and not (d / "checkpoint_0000004.pt").exists()
    assert "警告" in capsys.readouterr().out


# --------------------------------------------------------------------------- 読解データ・デコード比較・評価軽量化
def _jsquad_obj():
    ctx1 = "山田太郎 [SEP] " + "富士山は日本で最も高い山である。標高は3776メートルである。" + "静岡県と山梨県にまたがる。" * 3
    ctx2 = "海の生き物 [SEP] " + "クジラは海に住む哺乳類である。体長は最大で30メートルに達する。" + "プランクトンを食べる種もいる。" * 3
    def qa(i, q, ctx, ans):
        body = ctx.split(" [SEP] ", 1)[1]
        return {"id": str(i), "question": q, "is_impossible": False,
                "answers": [{"text": ans, "answer_start": ctx.index(body) + body.index(ans)}]}
    return {"data": [
        {"title": "山田太郎", "paragraphs": [{"context": ctx1, "qas": [qa(1, "富士山の標高は？", ctx1, "3776メートル")]}]},
        {"title": "海の生き物", "paragraphs": [{"context": ctx2, "qas": [qa(2, "クジラの体長は最大でどれくらい？", ctx2, "30メートル")]}]},
    ]}


def test_parse_jsquad_strips_title_and_keeps_answer_position():
    from src.import_reading_data import parse_jsquad
    rows = parse_jsquad(_jsquad_obj())
    assert len(rows) == 2 and rows[0]["title"] == "山田太郎"
    for r in rows:
        assert "[SEP]" not in r["context"]
        assert r["context"][r["answer_start"]:r["answer_start"] + len(r["answer"])] == r["answer"]


def test_window_context_keeps_answer_and_respects_limit():
    from src.import_reading_data import window_context
    ctx = "".join(f"これは{i}番目の文である。" for i in range(60)) + "答えは赤色である。" + "".join(f"後ろの{i}文目。" for i in range(60))
    start = ctx.index("赤色")
    w = window_context(ctx, start, 2, max_chars=120)
    assert w is not None
    text, s = w
    assert len(text) <= 120 and text[s:s + 2] == "赤色" and text.endswith("。")
    assert window_context("短い。", 0, 1, 450) == ("短い。", 0)


def test_reading_unanswerable_and_summary_items():
    import random
    from src.import_reading_data import (UNANSWERABLE, make_reading_items, make_summary_items,
                                         make_unanswerable_items, parse_jsquad, strip_private)
    reading = make_reading_items(parse_jsquad(_jsquad_obj()))
    assert len(reading) == 2 and reading[0]["answer"] == "3776メートル"
    assert "文章：" in reading[0]["input"] and "質問：富士山の標高は？" in reading[0]["input"]
    un = make_unanswerable_items(reading, 2, random.Random(0))
    assert un and all(u["answer"] == UNANSWERABLE for u in un)
    for u in un:   # 質問と文章は別記事（文章は質問の元記事ではない）
        q = u["input"].split("質問：")[-1]
        src = next(r for r in reading if r["_q"] == q)
        assert src["_ctx"] not in u["input"]
    assert "_ctx" not in strip_private(reading[0])
    lead = "富士山は、静岡県と山梨県にまたがる日本で最も高い山であり、標高は3776メートルである。"
    body1 = "富士山は静岡県と山梨県にまたがる成層火山で、標高は3776メートルであり日本で最も高い。" * 3
    doc = "\n".join([lead, body1, "古くから信仰の対象とされ、多くの登山者が訪れる。" * 5])
    sums = make_summary_items([doc, "短い。\n短い。"], 10)
    assert len(sums) == 1 and sums[0]["answer"] == lead and sums[0]["meta"]["task"] == "summary"


def test_import_reading_cli_end_to_end(tmp_path, monkeypatch):
    import json as _j
    src = tmp_path / "js"; src.mkdir()
    for n in ("train-v1.1.json", "valid-v1.1.json"):
        (src / n).write_text(_j.dumps(_jsquad_obj(), ensure_ascii=False), encoding="utf-8")
    hf = tmp_path / "hf"; hf.mkdir()
    for n in ("train", "val"):
        (hf / f"hf_import.{n}.jsonl").write_text(_j.dumps({"id": "x", "input": "既存", "cot": "", "answer": "a",
                                                           "meta": {}}, ensure_ascii=False) + "\n", encoding="utf-8")
    from src import import_reading_data as M
    out = tmp_path / "out"
    argv = ["x", "--out_dir", str(out), "--jsquad_dir", str(src), "--merge_into", str(hf),
            "--yardstick", str(tmp_path / "none.jsonl"), "--unans_ratio", "1.0"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(M, "_heldout", lambda p: ["全く関係のない質問です"])
    class _NoLeak:
        def __init__(self, *a, **k): pass
        def is_leak(self, t): return False
    monkeypatch.setitem(sys.modules, "src.import_pretrain_data",
                        types.SimpleNamespace(LeakFilter=_NoLeak, _heldout_texts=lambda p: []))
    M.main()
    M.main()   # 2回走らせても二重に追記されない
    n_tr = sum(1 for _ in open(hf / "hf_import.train.jsonl", encoding="utf-8"))
    n_va = sum(1 for _ in open(hf / "hf_import.val.jsonl", encoding="utf-8"))
    n_new = sum(1 for _ in open(out / "reading.train.jsonl", encoding="utf-8")) + \
        sum(1 for _ in open(out / "reading.val.jsonl", encoding="utf-8"))
    assert n_tr + n_va == 2 + n_new and n_new >= 2
    assert (out / "reading_eval.jsonl").exists() and (hf / "hf_import.train.jsonl.bak").exists()


def test_decode_compare_metrics_and_prompts(tmp_path):
    from src.decode_compare import keyword_hit, load_prompts, rep_ratio, summarize
    assert rep_ratio("氷、氷、氷、氷、氷、氷、氷、氷、氷、氷") > 0.5
    assert rep_ratio("日本の首都は東京で、人口は約1400万人です。") < 0.1
    assert rep_ratio("短い") == 0.0
    assert keyword_hit("答えは窒素です", ["窒素"]) is True and keyword_hit("x", None) is None
    y = tmp_path / "y.jsonl"
    y.write_text(json.dumps({"input": "空気の主成分は？", "expect_words": ["窒素"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    p = load_prompts(str(y), 5)
    assert p == [{"input": "空気の主成分は？", "keywords": ["窒素"]}]
    assert len(load_prompts(str(tmp_path / "none"), 3)) == 3
    rows = summarize({"A": [{"q": "q", "text": "窒素", "rep": 0.0, "hit": True}]})
    assert rows[0]["正解語を含む率"] == 1.0


def test_limit_loader_and_best_checkpoint_is_weights_only(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import limit_loader, save_checkpoint
    ds = list(range(1000))
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=lambda b: b)
    small = limit_loader(dl, 100)
    assert len(small.dataset) == 100 and sum(len(b) for b in small) == 100
    assert small.dataset[0] == 0 and small.dataset[1] == 10      # 等間隔
    assert limit_loader(dl, 0) is dl and limit_loader(dl, 5000) is dl
    m = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(m.parameters())
    d = str(tmp_path / "ck")
    save_checkpoint(1000, m, opt, None, d)
    save_checkpoint(2000, m, opt, None, d)
    save_checkpoint(3000, m, opt, None, d)
    save_checkpoint(3000, m, opt, None, d, is_best=True)
    names = sorted(os.listdir(d))
    assert names == ["checkpoint_002000.pt", "checkpoint_003000.pt", "checkpoint_best.pt"]   # 古いものは消える
    best = torch.load(os.path.join(d, "checkpoint_best.pt"))
    assert "model_state_dict" in best and "optimizer_state_dict" not in best
    assert os.path.getsize(os.path.join(d, "checkpoint_best.pt")) < os.path.getsize(os.path.join(d, "checkpoint_003000.pt"))
