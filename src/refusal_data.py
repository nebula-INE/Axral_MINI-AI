"""
refusal_data.py
「分からない」を、モデル自身に学習させるためのデータを作る（拒否学習）。

背景:
  知識ガード（knowledge_base）は語の一致で判定するため、「同じ話題で属性だけ未知」
  （例:「日本の面積は？」）のような紛らわしい問題を、紙一重でしか見分けられない。
  そこで、モデル自身に「学習していないことは断る」を学習させる。

作るデータ（答えはすべて UNKNOWN_TOPIC_RESPONSE の定型文。思考過程は付けない）:
  near  既知の話題 × 未知の属性   例:「富士山の登山シーズンは？」「Pythonの開発者は？」
  far   学習していない話題         例:「フランスとは？」「ライオンについて教えて」

公平さ（物差し・調整用データとの分離）:
  物差し yardstick_v1 と、ガード調整用データ（tune_guard.py）に使った属性・話題は、ここでは使わない。
  （例: 面積・人口・深さ・噴火・密度・ロシア・ドイツ・カンガルー など。）
  これにより、「拒否の学習に出てこなかった属性・話題でも、モデルが断れるか」を、物差しで
  本当の汎化として測れる。重複・類似は、生成時に自動で検査し、あればエラーにする。

安全装置:
  - 学習済みの質問（元の質問・言い換えバンク）と同じ文面を、拒否の問題にしない（矛盾した教師信号の防止）。
  - 検証用(val)は、学習に使わなかった属性・話題から作る。

使い方:
  python -m src.refusal_data              # data/v_refusal.{train,val}.jsonl, v_refusal_corpus.txt
  python -m src.refusal_data --check      # 重複・類似の検査だけ行う
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from src.knowledge_base import UNKNOWN_TOPIC_RESPONSE
from src.paraphrase_bank import PARAPHRASES, _norm, similarity

# --------------------------------------------------------------------------- 既知の話題 × 未知の属性
# 各リストの「最後の属性」は検証用(val)に取っておく（学習には使わない）。
NEAR_MISS: dict[str, list[str]] = {
    "日本": ["国花", "国鳥", "国歌", "時差", "公用語", "国旗の由来", "世界遺産の数"],
    "富士山": ["登山シーズン", "初登頂の年", "山頂の気温", "名前の由来", "登山道の数", "世界遺産登録の年"],
    "東京": ["地下鉄の路線数", "名前の由来", "主な観光地", "年間降水量", "区の数", "オリンピック開催年"],
    "琵琶湖": ["透明度", "生息する魚", "名前の由来", "周囲の長さ", "湖の成り立ち"],
    "エベレスト": ["登山にかかる日数", "山頂の気温", "名前の由来", "年間の登頂者数", "ベースキャンプの標高"],
    "北海道": ["道庁所在地", "主な農産物", "積雪量", "最北端の地名", "観光名所", "市町村の数"],
    "沖縄県": ["伝統料理", "方言", "台風の数", "主な産業", "県庁所在地", "観光名所"],
    "静岡県": ["県庁所在地", "主な観光地", "お茶の生産量", "市町村の数", "名物料理"],
    "山梨県": ["県庁所在地", "主な観光地", "ワイナリーの数", "名物料理", "市町村の数"],
    "水": ["比熱", "分子量", "色", "味", "pH", "硬度"],
    "空気": ["重さ", "酸素の割合", "圧力", "においの有無", "湿度の単位"],
    "人間": ["体温", "寿命", "血液型の種類", "歯の数", "筋肉の数", "脳の重さ"],
    "光": ["色の種類", "波長", "屈折率", "粒子の名前", "発見した人"],
    "音": ["周波数の単位", "伝わる仕組み", "反射の性質", "高低の決まり方", "測る機械"],
    "地球": ["自転周期", "年齢", "質量", "海の割合", "衛星の名前", "磁場の向き"],
    "太陽": ["年齢", "質量", "寿命", "種類", "表面の模様", "光が届く時間"],
    "DNA": ["塩基の種類", "名前の由来", "構成する塩基の数", "ヒトの染色体の数", "複製の仕組み"],
    "光合成": ["必要な光の色", "行われる場所", "化学反応式", "発見した人", "一日の量"],
    "鉄": ["融点", "元素記号", "原子番号", "磁石につく理由", "主な用途"],
    "Python": ["開発者", "名前の由来", "最初のリリース年", "実行速度", "インストール方法", "主な用途"],
    "SQL": ["読み方", "主な方言", "UPDATE文", "DELETE文", "GROUP BY句"],
    "IPアドレス": ["桁数", "種類", "IPv6", "割り当て方", "確認方法"],
    "HTTP": ["ステータスコードの意味", "メソッドの種類", "誕生した年", "ヘッダー", "クッキーとの関係"],
    "CPU": ["コア数の見方", "発熱", "主な製造会社", "キャッシュ", "消費電力"],
    "RAM": ["種類", "速度", "価格", "増設の方法", "読み方"],
    "AI": ["種類", "主な活用例", "誕生した年", "限界", "倫理的な課題"],
    "江戸幕府": ["滅亡の原因", "将軍の人数", "政治の仕組み", "鎖国政策", "身分制度"],
    "明治維新": ["背景", "主な改革", "影響", "廃藩置県", "薩長同盟"],
    "日本国憲法": ["三大原則", "条文の数", "改正の手続き", "第9条の内容", "公布された日"],
    "フランス革命": ["原因", "結果", "人権宣言", "バスチーユ襲撃", "ナポレオンとの関係"],
    "第二次世界大戦": ["原因", "戦死者の数", "主な戦場", "参戦国", "戦後処理"],
    "万里の長城": ["建設の目的", "建設に使われた材料", "世界遺産登録の年", "築かれた時代", "見学のしかた"],
    "ピタゴラスの定理": ["証明方法", "発見された時代", "逆の定理", "応用例", "名前の由来"],
    "太平洋": ["島の数", "名前の由来", "主な海流", "形成の歴史", "漁業の特徴"],
}
NEAR_FORMS = ["{E}の{A}は？", "{E}の{A}を教えてください", "{E}の{A}について知りたい",
              "{E}の{A}はどうなっていますか？", "{E}の{A}って何？"]

# --------------------------------------------------------------------------- 学習していない話題
# 最後の10%は検証用(val)に取っておく。
FAR_TOPICS = [
    "ポルトガル", "ブラジル", "インド", "エジプト", "カナダ", "オーストラリア", "韓国", "スペイン", "イタリア", "メキシコ",
    "パリ", "ロンドン", "大阪", "ニューヨーク", "シドニー", "北京", "ソウル", "ベルリン", "札幌", "福岡",
    "ピザ", "ラーメン", "天ぷら", "チョコレート", "コーヒー", "紅茶", "ワイン", "パスタ", "うどん", "味噌汁",
    "ライオン", "ゾウ", "ペンギン", "パンダ", "イルカ", "ワシ", "カメ", "ウサギ", "クジラ", "ハチ",
    "ラグビー", "テニス", "水泳", "バスケットボール", "マラソン", "柔道", "スキー", "ゴルフ",
    "ナポレオン", "ベートーベン", "アインシュタイン", "ニュートン", "ガリレオ", "ダーウィン", "織田信長", "坂本龍馬",
    "自転車", "電話", "冷蔵庫", "テレビ", "飛行機", "新幹線", "スマートフォン", "電子レンジ",
    "音楽", "演劇", "経済", "法律", "宗教", "哲学", "天文学", "地震", "台風", "火山",
    "ピカソ", "シェイクスピア", "モーツァルト", "ゴッホ", "ヒマラヤ", "サハラ砂漠", "ナイアガラの滝", "グランドキャニオン",
    "石油", "電気", "ダイヤモンド", "プラスチック", "ガラス", "紙", "絹", "綿",
    "ハロウィン", "クリスマス", "お正月", "七夕", "オリンピック", "万博", "ノーベル賞", "国連",
]
FAR_FORMS = ["{E}とは？", "{E}について教えて", "{E}について説明してください", "{E}ってどんなもの？", "{E}の特徴は？"]


def _known_questions() -> set[str]:
    from src.generate_data import CODE_SAMPLES, CONVERSATION_SAMPLES, QA_KNOWLEDGE_PAIRS, TECH_SAMPLES
    known = {q for q, _e, _a in (*QA_KNOWLEDGE_PAIRS, *TECH_SAMPLES, *CODE_SAMPLES, *CONVERSATION_SAMPLES)}
    known |= {p for ps in PARAPHRASES.values() for p in ps}
    return {_norm(q) for q in known}


def _yardstick_and_dev_inputs(path: str = "eval_sets/yardstick_v1.jsonl") -> list[str]:
    from src.tune_guard import NEG_FAR, NEG_NEAR, NEG_SHORT
    with open(path, encoding="utf-8") as f:
        yard = [json.loads(line)["input"] for line in f if line.strip()]
    return yard + NEG_NEAR + NEG_FAR + NEG_SHORT


def check_leaks(questions: list[str], yardstick_path: str = "eval_sets/yardstick_v1.jsonl",
                threshold: float = 0.70) -> list[str]:
    known = _known_questions()
    heldout = _yardstick_and_dev_inputs(yardstick_path)
    problems = []
    for q in questions:
        if _norm(q) in known:
            problems.append(f"[学習済みの質問と同一] {q}")
        for h in heldout:
            if _norm(q) == _norm(h):
                problems.append(f"[物差し/調整用と同一] {q}")
            elif similarity(q, h) >= threshold:
                problems.append(f"[物差し/調整用に類似 {similarity(q, h):.2f}] {q} ≈ {h}")
    return problems


def check_topics_not_heldout(yardstick_path: str = "eval_sets/yardstick_v1.jsonl") -> list[str]:
    """文面の類似度では見逃す「同じ話題・同じ属性」を、より厳密に検査する。
      far : 学習に使う話題が、物差し・調整用の問題文に含まれていないこと
      near: 学習に使う（話題, 属性）の組が、物差し・調整用の同じ問題文に両方含まれていないこと
    """
    heldout = [_norm(h) for h in _yardstick_and_dev_inputs(yardstick_path)]
    problems = []
    for topic in FAR_TOPICS:
        for h in heldout:
            if len(_norm(topic)) >= 2 and _norm(topic) in h:
                problems.append(f"[話題が物差し/調整用に登場] 「{topic}」 in 「{h}」")
    for entity, attrs in NEAR_MISS.items():
        for attr in attrs:
            for h in heldout:
                if _norm(entity) in h and _norm(attr) in h:
                    problems.append(f"[話題×属性が物差し/調整用に登場] 「{entity}」×「{attr}」 in 「{h}」")
    return problems


def all_questions() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(学習用, 検証用) の (質問, 種類) 一覧（全組み合わせ）。"""
    train, val = [], []
    for entity, attrs in NEAR_MISS.items():
        for i, attr in enumerate(attrs):
            target = val if i == len(attrs) - 1 else train
            target += [(f.format(E=entity, A=attr), "near") for f in NEAR_FORMS]
    cut = int(len(FAR_TOPICS) * 0.9)
    for i, topic in enumerate(FAR_TOPICS):
        target = train if i < cut else val
        target += [(f.format(E=topic), "far") for f in FAR_FORMS]
    return train, val


def _to_items(pairs: list[tuple[str, str]], prefix: str) -> list[dict]:
    return [{"id": f"{prefix}_{i:05d}", "input": q, "cot": "", "answer": UNKNOWN_TOPIC_RESPONSE,
             "meta": {"category": "refusal", "source": "refusal", "kind": kind, "has_cot": False}}
            for i, (q, kind) in enumerate(pairs)]


def generate(n_near: int, n_far: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train_all, val_all = all_questions()

    def pick(pairs: list[tuple[str, str]], kind: str, n: int) -> list[tuple[str, str]]:
        pool = [p for p in pairs if p[1] == kind]
        rng.shuffle(pool)
        # 組み合わせの総数を超える場合は重複を許す（拒否の文面は同じ答えなので、重複しても矛盾しない）
        return [pool[i % len(pool)] for i in range(n)] if n > len(pool) else pool[:n]

    count = lambda pairs, kind: sum(1 for p in pairs if p[1] == kind)
    n_near = n_near or count(train_all, "near")
    n_far = n_far or count(train_all, "far")
    train = pick(train_all, "near", n_near) + pick(train_all, "far", n_far)
    val = pick(val_all, "near", min(count(val_all, "near"), max(1, n_near // 5))) \
        + pick(val_all, "far", min(count(val_all, "far"), max(1, n_far // 5)))
    rng.shuffle(train)
    rng.shuffle(val)
    return _to_items(train, "refusal_train"), _to_items(val, "refusal_val")


def main():
    parser = argparse.ArgumentParser(description="拒否学習データの生成")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--version", default="v_refusal")
    parser.add_argument("--n_near", type=int, default=0, help="既知の話題×未知の属性の件数（0なら全候補を1回ずつ）")
    parser.add_argument("--n_far", type=int, default=0, help="学習していない話題の件数（0なら全候補を1回ずつ）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true", help="重複・類似の検査だけ行う")
    args = parser.parse_args()

    train_all, val_all = all_questions()
    problems = check_leaks([q for q, _ in train_all + val_all]) + check_topics_not_heldout()
    if problems:
        print(f"❌ {len(problems)}件の問題があります（物差し・調整用データとの重複／学習済み質問との衝突）:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"✓ 検査OK: 質問の候補{len(train_all) + len(val_all)}件（学習{len(train_all)} / 検証{len(val_all)}）は、"
          f"物差し・調整用データと重複・類似せず、学習済みの質問とも衝突しません")
    if args.check:
        return

    train, val = generate(args.n_near, args.n_far, args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in ((f"{args.version}.train.jsonl", train), (f"{args.version}.val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    # トークナイザー用コーパス: 質問は1回ずつ、定型の答えは1回だけ（同じ文を何千回も入れて偏らせない）
    with (out / f"{args.version}_corpus.txt").open("w", encoding="utf-8") as f:
        for q in sorted({it["input"] for it in train + val}):
            f.write(q + "\n")
        f.write(UNKNOWN_TOPIC_RESPONSE + "\n")
    kinds = {k: sum(1 for it in train if it["meta"]["kind"] == k) for k in ("near", "far")}
    print(f"✓ {out}/{args.version}.train.jsonl（{len(train)}件: {kinds}）/ val {len(val)}件")
    print(f"  例: {train[0]['input']} → {train[0]['answer']}")


if __name__ == "__main__":
    main()
