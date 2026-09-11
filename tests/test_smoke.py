"""
test_smoke.py
Kaggle Notebook等、torch/sentencepieceが使える環境で最初に実行するスモークテスト。
plan §5 段階3「1ステップ実行テスト」に対応。

実行方法:
    cd SEKOIA-vose-initial-llm
    python -m pytest tests/test_smoke.py -v
"""
import pytest

from src.eval_cot import evaluate_cot_quality

# ---------------------------------------------------------------------------
# torch非依存: 常に実行される
# ---------------------------------------------------------------------------

def test_eval_cot_arithmetic_correct_adopts_or_revises():
    r = evaluate_cot_quality(
        "3人で1200円を割り勘すると1人いくら？",
        "合計を確認する。1200円ある。人数で割る。1200 ÷ 3 = 400。答えは400円。",
        "400円",
        "arithmetic",
    )
    assert r.breakdown["logic"] == 1.0
    assert r.status in ("adopt", "revise")


def test_eval_cot_arithmetic_wrong_rejects():
    r = evaluate_cot_quality("2+2は？", "2+2 = 5", "5", "arithmetic")
    assert r.status == "reject"
    assert r.breakdown["logic"] == 0.0


def test_eval_cot_qa_flags_manual_review():
    r = evaluate_cot_quality("東京の人口は？", "統計局によると約1400万人。", "約1400万人", "qa")
    assert r.needs_manual_review is True


def test_verify_arithmetic_tolerates_floor_division():
    """回帰テスト: 2299×22÷100=505.78→505（切り捨て）を正しい計算として扱う。
    Kaggle実データで発覚した誤検出バグ（v1では reject=1815件）の再発防止。
    """
    from src.eval_cot import verify_arithmetic

    assert verify_arithmetic("2299 × 22 ÷ 100 = 505", "505円") == 1.0
    # 本物の誤りは引き続き検出できること
    assert verify_arithmetic("2+2 = 5", "5") == 0.0


# ---------------------------------------------------------------------------
# torch依存: torchが無い環境（このサンドボックス等）では自動的にskipされる
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _tiny_config():
    return {
        "vocab_size": 64,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 2,
        "d_ff": 32,
        "max_seq_length": 32,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
    }


def test_model_forward_shape():
    from src.model import TransformerLM

    model = TransformerLM(_tiny_config())
    input_ids = torch.randint(0, 64, (2, 10))
    out = model(input_ids)
    assert out.logits.shape == (2, 10, 64)


def test_model_generate_extends_sequence():
    from src.model import TransformerLM

    model = TransformerLM(_tiny_config())
    input_ids = torch.randint(0, 64, (1, 5))
    generated = model.generate(input_ids, max_new_tokens=4)
    assert generated.shape[1] == 9  # 5 (prompt) + 4 (new tokens)


def test_train_one_step_runs(tmp_path):
    """1ステップだけ学習ループを回し、lossがNaNにならないことを確認する。"""
    import json
    import yaml
    from src.train import train

    # ダミーのtoken_ids付きjsonlを作る（tokenizerなし＝generation評価スキップ）
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "val"):
        path = data_dir / f"v001.{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in range(8):
                item = {
                    "id": f"dummy_{i}",
                    "input": "dummy",
                    "answer": "dummy",
                    "token_ids": [3, 4, 5, 6],
                    "cot_token_ids": [],
                    "answer_token_ids": [7, 8],
                    "meta": {"category": "qa"},
                }
                f.write(json.dumps(item) + "\n")

    config = {
        "experiment_name": "smoke_test",
        "model": _tiny_config(),
        "optimizer": {"lr": 1e-3, "lr_warmup_steps": 1},
        "data": {
            "train_path": str(data_dir / "v001.train.jsonl"),
            "val_path": str(data_dir / "v001.val.jsonl"),
            "tokenizer_path": None,
            "vocab_size": 64,
            "batch_size": 4,
            "eval_batch_size": 4,
        },
        "training": {
            "epochs": 1,
            "grad_accum_steps": 1,
            "max_grad_norm": 1.0,
            "eval_frequency": 1,
            "checkpoint_frequency": 1000,
            "early_stopping_patience": 100,
            "resume_from_checkpoint": False,
        },
        "kaggle": {"max_session_time_minutes": 540},
        "wandb": {"enabled": False},
        "results_dir": str(tmp_path / "checkpoints"),
        "logs_dir": str(tmp_path / "logs"),
    }
    config_path = tmp_path / "exp_smoke.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    result = train(str(config_path))
    assert result["final_step"] >= 1
    assert result["best_val_loss"] == result["best_val_loss"]  # NaNでないことの簡易チェック
