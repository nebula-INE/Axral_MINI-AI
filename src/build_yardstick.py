"""
build_yardstick.py
「物差し」問題集 yardstick_v1 を作る。

目的:
  sanity_check は学習データの再現率を測るので、モデルが成長しているかどうか（未知の言い方に
  対応できるようになったか）を測れない。この問題集は、学習済みの内容を「学習で見ていない
  言い回し」で尋ねることで、汎化の度合いを測る。

使い方:
  python -m src.build_yardstick          # eval_sets/yardstick_v1.jsonl を作る（学習済み文との重複も自動チェック）

問題の種類（type）:
  paraphrase    学習済みの内容を、別の言い回しで尋ねる
  new_numbers   学習済みの文型そのまま、数字だけ未知（純粋な計算の汎化）
  number_range  学習範囲（人数2〜10、面積の辺2〜50など）の外の数字
  unit          学習していない単位（cm）
  out_of_scope  学習していない話題。正しい挙動は「分かりません」（知識ガードの評価用）

カテゴリ（category）: qa / technical / code / conversation / arithmetic / out_of_scope
  ※ 学習データ側のカテゴリ名（generate_data.py）に合わせてある。

各問題のフィールド:
  id, category, type, input, behavior("answer"|"refuse"), expected_any（いずれかを含めば正解）,
  source（言い換え元の学習済み質問。無い場合は null）, domain（qaの分野。任意）

重要（問題集の公平性）:
  - 学習データのテンプレートと全く同じ文面の問題を入れない（new_numbers を除く）。
    build時に、学習済み質問とその派生形（〜を教えて／〜について教えて／〜といえば？等）との
    一致を自動チェックし、被っていればエラーにする。
  - 公開データ(dolly/oasst)との重複は、このチェックでは確認できない。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.generate_data import CODE_SAMPLES, CONVERSATION_SAMPLES, QA_KNOWLEDGE_PAIRS, TECH_SAMPLES

OUT_PATH = Path("eval_sets/yardstick_v1.jsonl")

_ITEMS: list[dict] = []


def _add(category: str, typ: str, text: str, expected: list[str] | None, source: str | None = None,
         domain: str | None = None, behavior: str = "answer") -> None:
    _ITEMS.append({
        "category": category, "type": typ, "input": text, "behavior": behavior,
        "expected_any": expected or [], "source": source, "domain": domain,
    })


def qa(domain: str, source: str, text: str, expected: list[str]) -> None:
    _add("qa", "paraphrase", text, expected, source=source, domain=domain)


# --------------------------------------------------------------------------- QA: 地理
G = "geography"
qa(G, "富士山の標高は？", "富士山の高さはどれくらい？", ["3776"])
qa(G, "富士山の標高は？", "富士山の高さは何メートルですか？", ["3776"])
qa(G, "富士山はどこにありますか？", "富士山ってどこの県にあるの？", ["静岡", "山梨"])
qa(G, "富士山はどこにありますか？", "富士山の所在地は？", ["静岡", "山梨"])
qa(G, "静岡県の名産品は？", "静岡で有名な特産品は？", ["お茶", "茶"])
qa(G, "山梨県の名産品は？", "山梨の特産品といえば何ですか？", ["ぶどう", "ワイン"])
qa(G, "日本の首都はどこですか？", "日本の首都ってどこ？", ["東京"])
qa(G, "日本の首都はどこですか？", "日本の首都名を答えて", ["東京"])
qa(G, "東京はどの地方にありますか？", "東京は日本のどのあたりにある？", ["関東"])
qa(G, "日本で一番大きい湖は？", "日本最大の湖は？", ["琵琶湖"])
qa(G, "日本で一番大きい湖は？", "日本の一番大きな湖の名前は？", ["琵琶湖"])
qa(G, "琵琶湖はどこにありますか？", "琵琶湖があるのは何県？", ["滋賀"])
qa(G, "日本で一番長い川は？", "日本で最長の川は？", ["信濃川"])
qa(G, "日本の人口はおよそ何人？", "日本の人口は約何人ですか？", ["1億2000万"])
qa(G, "北海道の気候の特徴は？", "北海道はどんな気候？", ["冷涼", "亜寒帯", "寒"])
qa(G, "沖縄県の気候の特徴は？", "沖縄の気候はどんな感じ？", ["亜熱帯", "温暖"])
qa(G, "世界で一番高い山は？", "世界最高峰の山は？", ["エベレスト"])
qa(G, "世界で一番高い山は？", "地球で一番高い山の名前は？", ["エベレスト"])
qa(G, "エベレストはどこにありますか？", "エベレストはどの国にある？", ["ネパール", "中国"])
qa(G, "世界で一番長い川は？", "世界最長の川は？", ["ナイル"])
qa(G, "世界で一番広い海は？", "世界で最も大きな海は？", ["太平洋"])
qa(G, "万里の長城はどこの国にありますか？", "万里の長城があるのはどこの国？", ["中国"])

# --------------------------------------------------------------------------- QA: 科学
S = "science"
qa(S, "水の化学式は？", "水の化学式を答えて", ["H2O"])
qa(S, "水は何度で氷になりますか？", "水が凍るのは何度？", ["0"])
qa(S, "水は何度で沸騰しますか？", "水が沸騰する温度は？", ["100"])
qa(S, "空気中で一番多い気体は？", "空気の主成分は？", ["窒素"])
qa(S, "人間が呼吸に使う気体は？", "人は呼吸で何の気体を使う？", ["酸素"])
qa(S, "光の速度はいくら？", "光はどれくらいの速さ？", ["30万"])
qa(S, "地球の平均気温は？", "地球の平均の気温は？", ["15"])
qa(S, "地球から太陽までの距離は？", "太陽までの距離は？", ["1億5000万"])
qa(S, "太陽系で一番大きい惑星は？", "太陽系最大の惑星は？", ["木星"])
qa(S, "太陽系で一番小さい惑星は？", "太陽系の一番小さい惑星は？", ["水星"])
qa(S, "人間の骨の数は？", "人体の骨は何本ある？", ["206"])
qa(S, "人間の心臓は何室ありますか？", "心臓の部屋はいくつ？", ["4"])
qa(S, "DNAの二重螺旋構造を発見したのは？", "DNAの構造を見つけた人は？", ["ワトソン", "クリック"])
qa(S, "光合成をする生物は？", "光合成を行うのはどんな生物？", ["植物"])
qa(S, "ピタゴラスの定理とは？", "三平方の定理って何？", ["a² + b² = c²"])
qa(S, "鉄が錆びる原因は？", "鉄が錆びるのはなぜ？", ["酸素", "酸化"])

# --------------------------------------------------------------------------- QA: IT
I = "it"
qa(I, "Pythonでリストから重複を除く方法は？", "Pythonでリストの重複を取り除くには？", ["set"])
qa(I, "Pythonのリストと辞書の違いは？", "Pythonのlistとdictの違いは？", ["キー"])
qa(I, "Pythonで例外処理に使う構文は？", "Pythonのエラー処理の書き方は？", ["try"])
qa(I, "SQLのJOINとは？", "SQLのJOINって何？", ["結合"])
qa(I, "SQLでデータを絞り込む句は？", "SQLで条件を指定して絞るには？", ["WHERE"])
qa(I, "IPアドレスとは？", "IPアドレスって何？", ["識別", "番号"])
qa(I, "HTTPとHTTPSの違いは？", "HTTPSはHTTPと何が違う？", ["暗号"])
qa(I, "CPUの役割は？", "CPUは何をするもの？", ["命令", "演算"])
qa(I, "RAMとストレージの違いは？", "RAMとストレージはどう違う？", ["一時"])
qa(I, "AIとは何の略ですか？", "AIは何の略？", ["Artificial Intelligence", "人工知能"])

# --------------------------------------------------------------------------- QA: 歴史
H = "history"
qa(H, "江戸幕府を開いたのは誰ですか？", "江戸幕府の初代将軍は？", ["家康"])
qa(H, "江戸幕府を開いたのは誰ですか？", "江戸幕府を作った人物は？", ["家康"])
qa(H, "明治維新が起きたのはいつ頃ですか？", "明治維新は何年ごろ？", ["1868"])
qa(H, "日本国憲法が施行されたのはいつですか？", "日本国憲法はいつ施行された？", ["1947"])
qa(H, "フランス革命が起きたのはいつですか？", "フランス革命は何年？", ["1789"])
qa(H, "第二次世界大戦が終わったのはいつですか？", "第二次世界大戦の終結は何年？", ["1945"])

# --------------------------------------------------------------------------- technical
_add("technical", "paraphrase", "JSONを読み込むには？", ["json"], source="JSONファイルの読み込み方法は？")
_add("technical", "paraphrase", "Pythonでjsonファイルを読む方法は？", ["json"], source="JSONファイルの読み込み方法は？")

# --------------------------------------------------------------------------- code
C = "code"
_add(C, "paraphrase", "リストを昇順に並べ替えるPythonコードは？", ["sorted", "sort"], source="Pythonでリストをソートするコードは？")
_add(C, "paraphrase", "0から9までを順番に表示するforループは？", ["range(10)"], source="for ループで0から9まで出力するコードは？")
_add(C, "paraphrase", "偶数だけを取り出すリスト内包表記は？", ["%2==0"], source="Pythonでリスト内包表記を使って偶数だけ抽出するコードは？")
_add(C, "paraphrase", "辞書のキーと値を順に取り出すPythonの書き方は？", ["items"], source="Pythonで辞書の値でループ処理するコードは？")
_add(C, "paraphrase", "カンマ区切りの文字列を分けるには？", ["split"], source="Pythonで文字列を分割するコードは？")
_add(C, "paraphrase", "ファイルを1行ずつ読むPythonコードは？", ["open"], source="Pythonでファイルを1行ずつ読み込むコードは？")
_add(C, "paraphrase", "Pythonで関数を作る方法は？", ["def"], source="Pythonで関数を定義するコードは？")
_add(C, "paraphrase", "リストの最大値を取得するには？", ["max"], source="Pythonでリストの最大値を求めるコードは？")
_add(C, "paraphrase", "文字列を整数にするには？", ["int"], source="Pythonで文字列を数値に変換するコードは？")
_add(C, "paraphrase", "リストの長さを調べるには？", ["len"], source="Pythonでリストの要素数を数えるコードは？")
_add(C, "paraphrase", "リストの末尾に要素を足すには？", ["append"], source="Pythonでリストに要素を追加するコードは？")
_add(C, "paraphrase", "リストを逆順にするには？", ["[::-1]", "reversed"], source="Pythonでリストの中身を逆順にするコードは？")

# --------------------------------------------------------------------------- conversation
V = "conversation"
_add(V, "paraphrase", "どうもありがとう", ["どういたしまして"], source="ありがとう")
_add(V, "paraphrase", "ありがとうございました", ["どういたしまして"], source="ありがとう")
_add(V, "paraphrase", "さよなら", ["さようなら"], source="さようなら")
_add(V, "paraphrase", "最近元気？", ["元気"], source="お元気ですか？")
_add(V, "paraphrase", "疲れたなあ", ["お疲れ"], source="疲れた")
_add(V, "paraphrase", "ちょっと疲れちゃった", ["お疲れ", "無理"], source="疲れた")
_add(V, "paraphrase", "よろしくお願いいたします", ["よろしく"], source="よろしくお願いします")
_add(V, "paraphrase", "お昼は何を食べましたか？", ["ラーメン"], source="昼食は何を食べた？")
_add(V, "paraphrase", "今日の天気を教えて", ["雲", "雨"], source="今日の天気はどう？")
_add(V, "paraphrase", "どうも、こんにちは", ["こんにちは"], source="こんにちは")

# --------------------------------------------------------------------------- arithmetic（学習した5文型を基準に）
A = "arithmetic"
# 別の言い回し（文型が未知。数字は学習範囲内）
_add(A, "paraphrase", "6人で4800円を等分すると1人あたりいくら？", ["800"], source="{}人で{}円を割り勘すると1人いくら？")
_add(A, "paraphrase", "9人で5400円を均等に分けたら1人分は？", ["600"], source="{}人で{}円を割り勘すると1人いくら？")
_add(A, "paraphrase", "4人で3200円をみんなで割ると1人いくらになる？", ["800"], source="{}人で{}円を割り勘すると1人いくら？")
_add(A, "paraphrase", "8人の飲み会で合計6400円でした。1人あたりの金額は？", ["800"], source="{}人で{}円を割り勘すると1人いくら？")
_add(A, "paraphrase", "1個150円のあめを12個買うと合計いくら？", ["1800"], source="1個{}円のパンを{}個買うといくら？")
_add(A, "paraphrase", "250円のりんごを8個買った。代金は？", ["2000"], source="1個{}円のパンを{}個買うといくら？")
_add(A, "paraphrase", "1個320円のパンを5個買ったらいくら？", ["1600"], source="1個{}円のパンを{}個買うといくら？")
_add(A, "paraphrase", "パンが1個410円です。3個ではいくら？", ["1230"], source="1個{}円のパンを{}個買うといくら？")
_add(A, "paraphrase", "2000円の15パーセントはいくら？", ["300"], source="{}円の{}%はいくら？")
_add(A, "paraphrase", "8000円の25%を求めて", ["2000"], source="{}円の{}%はいくら？")
_add(A, "paraphrase", "5000円の20パーセントは？", ["1000"], source="{}円の{}%はいくら？")
_add(A, "paraphrase", "時速80kmの車で3時間走ると何km進む？", ["240"], source="時速{}kmで{}時間走ると何km進む？")
_add(A, "paraphrase", "時速60kmで5時間移動したときの距離は？", ["300"], source="時速{}kmで{}時間走ると何km進む？")
_add(A, "paraphrase", "1時間に90km進む電車は4時間で何km進む？", ["360"], source="時速{}kmで{}時間走ると何km進む？")
_add(A, "paraphrase", "横30m、縦20mの長方形の面積を求めて", ["600"], source="縦{}m、横{}mの長方形の面積は？")
_add(A, "paraphrase", "たて15メートル、よこ10メートルの長方形の面積は？", ["150"], source="縦{}m、横{}mの長方形の面積は？")
# 同じ文型で、数字だけ未知（純粋な計算の汎化）
_add(A, "new_numbers", "7人で5600円を割り勘すると1人いくら？", ["800"], source="{}人で{}円を割り勘すると1人いくら？")
_add(A, "new_numbers", "1個275円のパンを13個買うといくら？", ["3575"], source="1個{}円のパンを{}個買うといくら？")
_add(A, "new_numbers", "4000円の35%はいくら？", ["1400"], source="{}円の{}%はいくら？")
_add(A, "new_numbers", "時速95kmで6時間走ると何km進む？", ["570"], source="時速{}kmで{}時間走ると何km進む？")
_add(A, "new_numbers", "縦12m、横8mの長方形の面積は？", ["96"], source="縦{}m、横{}mの長方形の面積は？")
# 学習範囲の外の数字（人数は2〜10、辺は2〜50、時速は40〜120 で学習）
_add(A, "number_range", "15人で9000円を割り勘すると1人いくら？", ["600"], source="{}人で{}円を割り勘すると1人いくら？")
_add(A, "number_range", "縦80m、横90mの長方形の面積は？", ["7200"], source="縦{}m、横{}mの長方形の面積は？")
_add(A, "number_range", "時速200kmで3時間走ると何km進む？", ["600"], source="時速{}kmで{}時間走ると何km進む？")
# 学習していない単位
_add(A, "unit", "縦5cm、横25cmの長方形の面積は？", ["125"], source="縦{}m、横{}mの長方形の面積は？")
_add(A, "unit", "縦10cm、横30cmの長方形の面積は？", ["300"], source="縦{}m、横{}mの長方形の面積は？")

# --------------------------------------------------------------------------- 学習範囲外（正しい挙動は「分かりません」）
for _text in [
    "日本の面積は？", "ロシアの首都はどこですか？", "カンガルーの平均寿命は？", "ドイツの人口は？",
    "サッカーのワールドカップは何年ごとに開催されますか？", "ビットコインとは？", "夏目漱石の代表作は？",
    "モナ・リザを描いたのは誰ですか？", "東京タワーの高さは？", "地球の直径は？", "日本の通貨は？",
    "英語で「ありがとう」は？", "アマゾン川はどこを流れていますか？", "ピラミッドはどこの国にありますか？",
    "月までの距離は？",
]:
    _add("out_of_scope", "out_of_scope", _text, None, behavior="refuse")


# --------------------------------------------------------------------------- 重複チェック
def _trained_variants(q: str) -> set[str]:
    """generate_data.py の _vary_question と同じ規則で、学習に出現しうる質問文を列挙する。"""
    base = q.rstrip("？")
    out = {q, base, base + "？"}
    if base.endswith("とは"):
        stem = base[:-2]
        out |= {f"{stem}とは？", f"{stem}について教えて", f"{stem}とは何ですか？", f"{stem}といえば？"}
    elif base.endswith("ですか"):
        stem = base[:-3]
        out |= {f"{stem}？", f"{stem}を教えて", f"{stem}について教えて"}
    else:
        out |= {f"{base}？", f"{base}を教えて", f"{base}について教えて"}
    m = re.match(r"^([^\sはがのとですか？]+)", base)
    if m and len(m.group(1)) >= 2:
        out.add(f"{m.group(1)}といえば？")
    return out


def _norm(s: str) -> str:
    return s.rstrip("？?！!").strip()


def check_no_leak(items: list[dict]) -> list[str]:
    trained: set[str] = set()
    for q, _e, _a in (*QA_KNOWLEDGE_PAIRS, *TECH_SAMPLES, *CODE_SAMPLES, *CONVERSATION_SAMPLES):
        trained |= {_norm(v) for v in _trained_variants(q)}
    problems = []
    for it in items:
        if it["type"] == "new_numbers":
            continue  # 学習済み文型そのまま、は意図的
        if _norm(it["input"]) in trained:
            problems.append(f"学習済みの質問と同一: {it['input']}")
    inputs = [it["input"] for it in items]
    for dup in {x for x in inputs if inputs.count(x) > 1}:
        problems.append(f"問題が重複: {dup}")
    return problems


def build(out_path: Path = OUT_PATH) -> list[dict]:
    items = []
    for i, it in enumerate(_ITEMS, 1):
        items.append({"id": f"y1_{i:03d}", **it})
    problems = check_no_leak(items)
    if problems:
        raise SystemExit("❌ 問題集に問題があります:\n  " + "\n  ".join(problems))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


def main():
    items = build()
    by_cat: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for it in items:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
    print(f"✓ {OUT_PATH} を作成しました（{len(items)}問）")
    print(f"  カテゴリ別: {by_cat}")
    print(f"  種類別    : {by_type}")
    print("  学習済みの質問・派生形との重複: なし（チェック済み）")


if __name__ == "__main__":
    sys.exit(main())
