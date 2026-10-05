"""
tune_guard.py
知識ガード(knowledge_base.guard_score)の閾値と事前重みを、「調整用データ」だけで決める。

重要: 物差し yardstick_v1 は採点専用。ここで使うのは次の3種類だけ。
  答えられる側  言い換えバンクを1件ずつ外し、残りの言い回しだけで判定する（＝未知の言い換えの再現）
  答えられない側 (a) 同じ話題で属性だけ未知の「紛らわしい問題」  (b) 全く別の話題  (c) 短い単語だけの入力
これらは物差しの問題と重ならないよう、実行時に自動チェックする。

注意: 「答えられる側」は、同じ事実の別の言い回しが索引に残るため、本物の未知の言い換えより易しい。
      実際の受け入れ率は、この数字より低い可能性がある。

使い方:  python -m src.tune_guard
"""
from __future__ import annotations

import json
import sys

from src.knowledge_base import GUARD_THRESHOLD, _GUARD_PRIOR, build_guard_index, guard_entries, guard_score
from src.paraphrase_bank import _norm, similarity

NEG_NEAR = [
    "富士山の噴火はいつ？", "富士山の面積は？", "東京の人口は？", "琵琶湖の深さは？", "エベレストの初登頂は誰？",
    "水の密度は？", "太陽の温度は？", "地球の半径は？", "Pythonのバージョンは？", "SQLのINSERTとは？",
    "北海道の人口は？", "沖縄県の面積は？", "静岡県の人口は？", "山梨県の気候は？", "日本の国土面積の順位は？",
    "日本の祝日の数は？", "江戸幕府は何年間続いた？", "明治維新の中心人物は？", "フランス革命の指導者は？",
    "第二次世界大戦の始まりはいつ？", "万里の長城の長さは？", "ナイル川の長さは？", "太平洋の深さは？",
    "木星の衛星の数は？", "水星の公転周期は？", "人間の血液の量は？", "HTTPのポート番号は？",
    "CPUのクロック周波数とは？", "RAMの容量の目安は？", "AIの歴史は？",
]
NEG_FAR = [
    "今日の株価は？", "おすすめの映画は？", "寿司の作り方は？", "カレーのレシピを教えて", "犬の飼い方は？",
    "英語の勉強法は？", "野球のルールは？", "京都の観光名所は？", "結婚式のマナーは？", "肩こりの解消法は？",
    "日本の歴代首相は？", "宇宙の始まりは？", "ブラックホールとは？", "恐竜はなぜ絶滅した？", "ピアノの弾き方は？",
    "税金の計算方法は？", "免許の取り方は？", "睡眠時間はどのくらい必要？", "日本の歴史を教えて", "世界の国の数は？",
]
NEG_SHORT = ["日本", "世界", "地球", "太陽", "関東", "中国", "ロシア", "ドイツ", "オートミール", "政治家"]


def _check_no_overlap_with_yardstick(path: str = "eval_sets/yardstick_v1.jsonl") -> None:
    with open(path, encoding="utf-8") as f:
        yard = [json.loads(line)["input"] for line in f if line.strip()]
    for q in NEG_NEAR + NEG_FAR + NEG_SHORT:
        for y in yard:
            if _norm(q) == _norm(y) or similarity(q, y) >= 0.7:
                sys.exit(f"❌ 調整用の問題が物差しと重複・類似しています: 「{q}」 ≈ 「{y}」")


def main():
    _check_no_overlap_with_yardstick()
    entries = guard_entries()
    originals = {q for q, _ in entries[:0]}  # （未使用。LOOは言い換え＝元の質問以外のみ）
    from src.paraphrase_bank import PARAPHRASES
    paraphrase_set = {p for ps in PARAPHRASES.values() for p in ps}
    full = build_guard_index(entries)
    loo = []
    for i, (q, _fid) in enumerate(entries):
        if q in paraphrase_set and q not in PARAPHRASES:
            loo.append((q, build_guard_index(entries[:i] + entries[i + 1:])))
    print(f"調整用: 答えられる側(言い換えを1件外す) {len(loo)}件 / 紛らわしい {len(NEG_NEAR)}件 / 別話題 {len(NEG_FAR)}件 / 短い単語 {len(NEG_SHORT)}件")

    best = None
    print(f"\n{'prior':>5}{'閾値':>7}{'受け入れ':>9}{'紛らわしい誤':>11}{'別話題誤':>9}{'短い単語誤':>10}")
    for prior in (2.0, 4.0, 6.0, 8.0):
        pos = [guard_score(q, idx, prior)[0] for q, idx in loo]
        near = [guard_score(q, full, prior)[0] for q in NEG_NEAR]
        far = [guard_score(q, full, prior)[0] for q in NEG_FAR]
        short = [guard_score(q, full, prior)[0] for q in NEG_SHORT]
        for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
            r = lambda a: sum(x >= t for x in a) / len(a)
            row = (r(pos), r(near), r(far), r(short))
            print(f"{prior:5.1f}{t:7.2f}{row[0]:9.2f}{row[1]:11.2f}{row[2]:9.2f}{row[3]:10.2f}")
            if row[1] <= 0.10 and row[2] == 0 and row[3] == 0 and (best is None or row[0] > best[0]):
                best = (row[0], prior, t, row)
    print()
    if best:
        print(f"推奨: prior={best[1]}, 閾値={best[2]}  （受け入れ{best[3][0]:.2f} / 紛らわしい誤{best[3][1]:.2f} / 別話題誤0 / 短い単語誤0）")
    else:
        print("条件（紛らわしい誤≤10%、別話題・短い単語の誤0）を満たす組み合わせがありませんでした。")
    print(f"現在の設定: prior={_GUARD_PRIOR}, 閾値={GUARD_THRESHOLD}")


if __name__ == "__main__":
    main()
