"""p7（事前学習→指示学習）のデータ周りの部品のテスト。torch が無い環境でも動く部分だけ。"""
import sys
import types


def test_code_text_roundtrip():
    from src.code_text import decode_text, encode_text
    code = "def add(a, b):\n    if a:\n        return a + b\n    return b\n"
    enc = encode_text(code)
    assert "\n" not in enc and "<nl>" in enc and "<i>" in enc
    assert decode_text(enc) == code
    # SentencePieceが記号の直後・先頭に付ける空白は取り除かれる
    assert decode_text(" " + enc.replace("<nl>", "<nl> ")) == code
    assert encode_text("a\n\tb") == "a<nl><i>b"          # タブは4スペース扱い
    assert decode_text("こんにちは") == "こんにちは"          # 記号が無ければ何もしない


def test_chunk_text_boundaries():
    from src.code_text import NL, encode_text
    from src.pack_tokens import chunk_text
    t = encode_text("def f(x):\n    return x\n" * 200)
    ch = chunk_text(t, 300)
    assert all(len(c) <= 300 for c in ch) and "".join(ch) == t
    assert all(c.endswith(NL) for c in ch[:-1])
    j = "これは文です。" * 500
    ch2 = chunk_text(j, 100)
    assert all(len(c) <= 100 for c in ch2) and "".join(ch2) == j
    assert "".join(chunk_text("あ" * 1000, 100)) == "あ" * 1000


def test_import_cleaners_and_leak_filter():
    from src.import_pretrain_data import (LeakFilter, clean_code, clean_wiki, is_python_row, pair_from_alpaca,
                                          split_bucket)
    assert clean_wiki("短い") is None
    w = clean_wiki("== 概要 ==\n" + "東京は日本の首都である。" * 30 + "\n\n\n== 歴史 ==\n江戸と呼ばれた。")
    assert w and "==" not in w and "\n\n" not in w
    c = clean_code("def f(x):\t\n\treturn x+1   \n\n\n\n\ndef g():\n    pass\n" * 3)
    assert c and "\t" not in c and "\n\n\n" not in c and clean_code("x=1") is None
    assert pair_from_alpaca({"instruction": "a", "input": "b", "output": "c"}) == ("a\nb", "c")
    assert pair_from_alpaca({"instruction": "a", "output": ""}) is None
    assert split_bucket("abc") in ("train", "val") and split_bucket("abc") == split_bucket("abc")
    assert is_python_row({"language": "python"}) and not is_python_row({"language": "java"})
    lf = LeakFilter(["日本の首都はどこですか？", "縦12m、横8mの長方形の面積は？"])
    assert lf.is_leak("日本の首都はどこですか") and lf.is_leak("縦35m、横9mの長方形の面積は？")
    assert not lf.is_leak("Pythonでリストを逆順にする方法を教えてください")


def test_lr_schedule_shape():
    for m in ("torch", "torch.nn", "torch.nn.functional"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["torch"].Tensor = object
    try:
        from src.pretrain import lr_at
    except Exception:  # torch が無い環境では utils/model の import が通らないので、式だけ取り出して検査
        import math
        src = open("src/pretrain.py", encoding="utf-8").read()
        body = src[src.index("def lr_at"):src.index("def pick_amp")]
        ns = {"math": math}
        exec(body, ns)
        lr_at = ns["lr_at"]
    assert lr_at(0.0, 1.0, 0.1, 0.02) == 0.0
    assert abs(lr_at(0.02, 1.0, 0.1, 0.02) - 1.0) < 1e-9
    assert abs(lr_at(1.0, 1.0, 0.1, 0.02) - 0.1) < 1e-9
    vals = [lr_at(p / 100, 1.0, 0.1, 0.02) for p in range(2, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))   # 単調に下がる
