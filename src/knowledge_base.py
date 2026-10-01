"""
knowledge_base.py
既知トピック判定（infer.pyの安全弁）。

generate_data.py の学習データ（QA_KNOWLEDGE_PAIRS・TECH_SAMPLES・CODE_SAMPLES・
CONVERSATION_SAMPLES）を単一の情報源として再利用し、モデルが学習していないはずの
話題について自信満々で誤った内容を生成してしまう問題への対処として、
「学習済みの話題に一致しない質問には、無理に答えず分かりませんと言わせる」
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

from src.generate_data import (
    CODE_SAMPLES,
    CONVERSATION_SAMPLES,
    QA_KNOWLEDGE_PAIRS,
    TECH_SAMPLES,
)

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

# 数字を含んでいても、学習済みの算数テンプレート（割り勘・単価×個数・割合・時速・面積）に
# 該当する語句が無ければ「未知」として扱う。
# 以前は数字が1文字でもあれば無条件で通していたため、「150の素因数分解」のような
# 未学習の算数問題がガードをすり抜け、もっともらしい誤答（幻覚）を生成していた。
_ARITH_HINT_RE = re.compile(r"人で|円|個|%|時速|km|時間|縦|横|長方形|面積|割り勘")


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
    """既知トピック一覧を構築する。

    QA_KNOWLEDGE_PAIRS（地理・科学・IT・歴史の一般知識、54件）だけでなく、
    TECH_SAMPLES（技術文書）・CODE_SAMPLES（コード）・CONVERSATION_SAMPLES（挨拶等の会話）
    も合わせて対象にする。

    2025-09-XX 修正: 当初QA_KNOWLEDGE_PAIRSのみを対象にしていたため、
    「こんにちは」のような挨拶や「JSONファイルの読み込み方法は？」のような
    技術質問まで「未学習トピック」と誤判定してガードがブロックしてしまう
    退行バグが発生した（これらは元々別カテゴリとしてモデルが学習済みのため）。
    generate_data.pyの全カテゴリ（算数を除く）を既知トピック判定の対象に含めることで解消する。
    算数（arithmetic）は含めない: 数字を含む入力は is_known_topic() 側で
    そもそも判定をスキップし常に「既知」扱いにしているため、ここに加える必要がない。
    """
    pairs: list[tuple[str, str | None]] = []
    for question, _explanation, _answer in (
        *QA_KNOWLEDGE_PAIRS,
        *TECH_SAMPLES,
        *CODE_SAMPLES,
        *CONVERSATION_SAMPLES,
    ):
        pairs.append(_split_entity_attr(question))
    return pairs


# 既知トピック（QA・技術・コード・会話の全カテゴリ）から抽出した (主題, 属性) のペア一覧。
# import時に一度だけ構築する（元データはいずれも不変なので毎回計算不要）。
KNOWN_TOPIC_PAIRS: list[tuple[str, str | None]] = _build_known_topic_pairs()

UNKNOWN_TOPIC_RESPONSE = (
    "すみません、その話題については学習していないため、正確にはお答えできません。"
)


def is_known_topic(text: str) -> bool:
    """入力テキストが学習済みの既知トピック（QA・技術・コード・会話）のいずれかに
    関連するかを判定する。

    数字を含み、かつ学習済みの算数テンプレートの語句（円・個・時速・面積など）を含む入力は
    判定をスキップしてTrueを返す（calculatorによる検算の対象のため）。

    それ以外は、(主題entity, 属性attr) のペアごとに、entityが入力テキストに
    含まれ、かつ（attrがある場合は）attrも含まれるかで判定する。
    entity単語1つだけの一致で「既知」とはしない（例:「日本」だけで、
    54トピックにない「日本の面積は？」まで既知扱いしてしまう誤判定を防ぐため）。
    """
    if not text.strip():
        return True  # 空文字はガードの対象外（呼び出し側で別途弾かれる想定）

    if _HAS_DIGIT_RE.search(text) and _ARITH_HINT_RE.search(text):
        return True

    for entity, attr in KNOWN_TOPIC_PAIRS:
        if entity not in text:
            continue
        if attr is None or attr in text:
            return True

    return False


# ---------------------------------------------------------------------------
# 話題の枝分かれ（関連トピック提案）
# ---------------------------------------------------------------------------
_QA_QUESTIONS: list[tuple[str, str, str | None]] = [
    (q, *_split_entity_attr(q)) for q, _e, _a in QA_KNOWLEDGE_PAIRS
]


def suggest_followups(user_input: str, generated: str, limit: int = 3) -> list[str]:
    """生成された文章に登場する既知トピックから、次に聞けそうな学習済み質問を提案する。

    モデルは1問1答でステートレスなため、「生成文から話題を広げる」機能は
    モデル側ではなく知識ベース側で担う。生成文（CoT含む全文）に主題(entity)が
    現れる学習済みQAを探し、そのまま質問文として提示する
    （学習済みの質問文なので、選べば確実に答えられる）。

    除外条件:
      - 主題がユーザーの入力に既に含まれている（同じ話題の繰り返しになるため）
      - 属性(attr)が生成文に含まれている（今まさに答えた内容のため）
    """
    seen: set[str] = set()
    out: list[str] = []
    for question, entity, attr in _QA_QUESTIONS:
        if len(entity) < 2 or entity not in generated or entity in user_input:
            continue
        if attr is not None and attr in generated:
            continue
        if question in seen:
            continue
        seen.add(question)
        out.append(question)
        if len(out) >= limit:
            break
    return out
