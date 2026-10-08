"""
code_text.py
SentencePieceで「改行」と「インデント」を失わずにコードを扱うための変換。

背景:
  SentencePieceは学習・符号化のとき、改行や連続する空白を1つの空白に潰す。
  Pythonはインデントが文法なので、潰れたコードを学習すると、学習してもコードとして壊れた形でしか出てこない。
  そこで、トークナイザーに渡す前に「改行→<nl>」「行頭の4スペース→<i>」へ置き換え、
  生成のあとで元に戻す。トークナイザーにこの記号（<nl>, <i>）が登録されていなければ何もしない
  （従来のトークナイザー・checkpointの動作は変わらない）。

制限（意図した割り切り）:
  - タブは4スペースとして扱う。
  - 行頭の空白が4の倍数でないとき、端数（1〜3個）は落とす（2スペースのインデントは潰れる）。
  - 行頭以外の連続空白は、SentencePieceの通常の扱いに従う。
"""
from __future__ import annotations

import re

NL = "<nl>"
IND = "<i>"
USER_SYMBOLS = [NL, IND]

_ARTIFACT_SPACE = re.compile(r"(<nl>|<i>) ")


def encode_text(text: str) -> str:
    """改行とインデントを記号に置き換える。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = []
    for line in text.split("\n"):
        n = 0
        while line.startswith("    "):
            line = line[4:]
            n += 1
        lines.append(IND * n + line.lstrip(" "))
    return NL.join(lines)


def decode_text(text: str) -> str:
    """記号を改行とインデント（4スペース）に戻す。"""
    if NL not in text and IND not in text:
        return text
    # SentencePieceが記号の直後に付ける空白（▁由来）を取り除く
    text = _ARTIFACT_SPACE.sub(r"\1", text)
    text = text.lstrip(" ")
    text = text.replace(IND, "    ").replace(NL, "\n")
    return text


class CodecTokenizer:
    """SentencePieceProcessor の薄いラッパー。記号が登録されていれば自動で変換する。"""

    def __init__(self, sp):
        self.sp = sp
        unk = sp.unk_id()
        self.enabled = sp.PieceToId(NL) != unk and sp.PieceToId(IND) != unk

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self.sp.EncodeAsIds(encode_text(text) if self.enabled else text)

    def decode(self, ids: list[int]) -> str:
        out = self.sp.DecodeIds(list(ids))
        return decode_text(out) if self.enabled else out

    # 既存コード（infer.py など）が SentencePieceProcessor と同じ名前で呼べるようにする
    def EncodeAsIds(self, text: str) -> list[int]:
        return self.encode(text)

    def DecodeIds(self, ids: list[int]) -> str:
        return self.decode(ids)

    def __getattr__(self, name):  # それ以外（GetPieceSize など）は元のsentencepieceにそのまま渡す
        return getattr(self.sp, name)
