"""
knowledge_base.py
既知トピック判定（infer.pyの安全弁）。

generate_data.QA_KNOWLEDGE_PAIRS（学習データ生成にも使われる54件のQAペア）を
単一の情報源として再利用し、モデルが学習していないはずの話題について
自信満々で誤った内容を生成してしまう問題への対処として、
「既知の54トピックに一致しない質問には、無理に答えず分かりませんと言わせる」
判定ロジックを提供する。

背景: sanity_checkで「日本の面積は？」のような未学習の話題に対し、
モデルが尤もらしい嘘を生成するケースが確認された。300万パラメータの
モデルに全世界の知識を学習させるのは非現実的なため、まずは
「知らないことを知らないと言える」ことを優先する（Web検索連携はPhase 2以降）。

判定方式について:
単純に「日本」のような主題（entity）1語だけの一致で「既知」と判定すると、
「日本の首都は？」は学習していても「日本の面積は？」まで既知扱いしてしまい、
本来防ぎたかった「未学習の話題への自信満々な誤答」を防げない
（面積は54トピックのどこにも含まれていないため）。
そのため、各QAペアの質問文を (主題entity, 属性attr) に分解し、
入力テキストに entity と attr の両方が含まれる場合のみ「既知」と判定する
（entityだけの単語質問、例:「富士山」「Python」は attr=None として単独一致でよい）。
"""
from __future__ import annotations

import re

from src.generate_data import QA_KNOWLEDGE_PAIRS

# attr（属性部分）の末尾からそぎ落とす定型的な語尾・助詞。
# 長いものから順に試し、除去後も2文字以上残る場合のみ除去する
# （除去しすぎて空文字や無意味な断片になるのを防ぐ）。
_TRAILING_PARTICLES = (
    "でしょうか", "ますか", "ですか", "ました", "します", "でしょう",
    "です", "ます", "か", "は", "が", "を", "に", "で", "と",
)

# attr（属性部分）の先頭側の区切り。属性部分に助詞が含まれる場合は、
# そこより前の名詞部分だけをattrとして採用する（例:「人口はおよそ何人」→「人口」）。
# これにより「日本の人口はおよそ何人？」で学習していても、より自然な言い回しの
# 「日本の人口は？」でも既知トピックとして拾えるようにする（過度に厳格な
# フレーズ一致による再現率低下を防ぐ）。
_ATTR_BOUNDARY_RE = re.compile(r"[はがをにでと]")

# 算数・割合・距離などの計算問題は、話題知識の有無ではなく計算力の問題であり、
# infer.py側の計算機検算（apply_calculator_correction）が別途担当する。
# 既知トピック判定の対象外として、数字を含む入力は常に「既知」扱いにする。
_HAS_DIGIT_RE = re.compile(r"\d")


def _clean_attr(attr: str) -> str:
    """属性部分の末尾から、意味を持たない定型語尾を最大3回まで剥がす。"""
    for _ in range(3):
        for particle in _TRAILING_PARTICLES:
            if attr.endswith(particle) and len(attr) - len(particle) >= 2:
                attr = attr[: -len(particle)]
                break
        else:
            break
    return attr


def _split_entity_attr(question: str) -> tuple[str, str | None]:
    """質問文を (主題entity, 属性attr) に分解する。

    - 「Xとは...」型（例:「ピタゴラスの定理とは？」）を最優先で扱う
      （定義を尋ねる質問はentityそのものが主題であることが多いため）。
    - 次に「Xの...」型（例:「静岡県の名産品は？」）を扱う。
    - 次に「Xは...」型（例:「水は何度で氷になりますか？」）を扱う。
    - どれにも当てはまらない場合（例:「富士山」単体）はentity=質問文全体、attr=Noneとする。

    attrが短すぎる/空の場合はNoneとし、その場合entity単独一致で「既知」と判定する
    （単語だけの質問や、定義を問うだけの質問を想定）。
    """
    base = question.rstrip("？")

    idx_towa = base.find("とは")
    idx_no = base.find("の")
    idx_ha = base.find("は")

    if idx_towa > 0:
        entity, attr_raw = base[:idx_towa], base[idx_towa + 2:]
    elif idx_no != -1 and (idx_ha == -1 or idx_no < idx_ha):
        entity, attr_raw = base[:idx_no], base[idx_no + 1:]
    elif idx_ha != -1:
        entity, attr_raw = base[:idx_ha], base[idx_ha + 1:]
    else:
        entity, attr_raw = base, ""

    if len(entity) < 2:
        # entityが短すぎて分解に失敗した場合は、質問文全体をentityとして扱う
        # （attrによる絞り込みなしの、より緩い一致になる）。
        return base, None

    m = _ATTR_BOUNDARY_RE.search(attr_raw)
    if m and m.start() >= 2:
        # attr_rawの先頭に助詞が現れる位置があれば、そこまでの名詞部分だけを使う
        # （残りは「はどこ」「はおよそ何人」等の質問の型であり、topic判定には不要）。
        attr = attr_raw[: m.start()]
    else:
        attr = _clean_attr(attr_raw)

    return entity, (attr if len(attr) >= 2 else None)


def _build_known_topic_pairs() -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for question, _explanation, _answer in QA_KNOWLEDGE_PAIRS:
        pairs.append(_split_entity_attr(question))
    return pairs


# 54件のQAペアから抽出した (主題, 属性) のペア一覧。
# import時に一度だけ構築する（QA_KNOWLEDGE_PAIRS自体は不変なので毎回計算不要）。
KNOWN_TOPIC_PAIRS: list[tuple[str, str | None]] = _build_known_topic_pairs()

UNKNOWN_TOPIC_RESPONSE = (
    "すみません、その話題については学習していないため、正確にはお答えできません。"
)


def is_known_topic(text: str) -> bool:
    """入力テキストが既知の54トピックのいずれかに関連するかを判定する。

    数字を含む入力（算数・割合問題など）は判定をスキップし、常にTrueを返す
    （calculatorによる検算の対象であり、話題知識の範囲外のため）。

    それ以外は、(主題entity, 属性attr) のペアごとに、entityが入力テキストに
    含まれ、かつ（attrがある場合は）attrも含まれるかで判定する。
    entity単語1つだけの一致で「既知」とはしない（例:「日本」だけで、
    54トピックにない「日本の面積は？」まで既知扱いしてしまう誤判定を防ぐため）。
    """
    if not text.strip():
        return True  # 空文字はガードの対象外（呼び出し側で別途弾かれる想定）

    if _HAS_DIGIT_RE.search(text):
        return True

    for entity, attr in KNOWN_TOPIC_PAIRS:
        if entity not in text:
            continue
        if attr is None or attr in text:
            return True

    return False
