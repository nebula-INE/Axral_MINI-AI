"""
generate_data.py
合成学習データを大規模に生成する。plan §2.1 のデータ種別比率に従う。

実行: python src/generate_data.py --output_dir data/ --num_samples 50000
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta

# テンプレートベースの合成データ生成（LLMなし、100%再現可能）


try:
    from src.paraphrase_bank import ARITH_PHRASINGS, PARAPHRASES
except ModuleNotFoundError:  # `python src/generate_data.py` で直接実行した場合（プロジェクトルートがsys.pathに無い）
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.paraphrase_bank import ARITH_PHRASINGS, PARAPHRASES

# 言い換えバンクの質問を使う確率（残りは元の文面）。元の文面も一定割合残し、基本形も学習させる。
PARAPHRASE_PROB = 0.7


def vary_question(q: str) -> str:
    """元の質問文、または言い換えバンクの言い換えをランダムに返す。"""
    bank = PARAPHRASES.get(q)
    if bank and random.random() < PARAPHRASE_PROB:
        return random.choice(bank)
    return q


def _arith_q(kind: str, original_template: str, a: int, b: int) -> str:
    """算数の質問文。元の文型、または言い換えの書式から選ぶ（数字の順序は元の文型と同じ）。"""
    if random.random() < 1 - PARAPHRASE_PROB:
        return original_template.format(a, b)
    return random.choice(ARITH_PHRASINGS[kind]).format(a, b)


def generate_arithmetic_data(count: int) -> list[dict]:
    """数学・論理問題（CoT付き、30%）。
    接続詞（まず/次に/よって）を明示的に入れ、count_reasoning_steps()が
    複数ステップとして認識できるようにしている（Gate1採用率対策）。
    """
    templates = [
        # 割り算: (input_template, cot_template, answer_template)
        ("{}人で{}円を割り勘すると1人いくら？",
         "まず、合計額を確認する。{}円である。次に、人数を確認する。{}人である。"
         "よって、割り勘額を計算する。{} ÷ {} = {}。したがって、答えは{}円である。",
         "{}円"),
        # 掛け算
        ("1個{}円のパンを{}個買うといくら？",
         "まず、単価を確認する。{}円である。次に、数量を確認する。{}個である。"
         "よって、合計額を計算する。{} × {} = {}。したがって、答えは{}円である。",
         "{}円"),
        # 割合
        ("{}円の{}%はいくら？",
         "まず、基準額を確認する。{}円である。次に、割合を確認する。{}%である。"
         "よって、割合金額を計算する。{} × {} ÷ 100 = {}。したがって、答えは{}円である。",
         "{}円"),
        # 時間・距離
        ("時速{}kmで{}時間走ると何km進む？",
         "まず、速度を確認する。時速{}kmである。次に、時間を確認する。{}時間である。"
         "よって、距離を計算する。{} × {} = {}。したがって、答えは{}kmである。",
         "{}km"),
        # 面積
        ("縦{}m、横{}mの長方形の面積は？",
         "まず、縦の長さを確認する。{}mである。次に、横の長さを確認する。{}mである。"
         "よって、面積を計算する。{} × {} = {}。したがって、答えは{}m²である。",
         "{}m²"),
    ]

    data = []
    for i in range(count):
        template_input, template_cot, template_ans = random.choice(templates)
        # テンプレートに合わせた具体値を生成（cot末尾に答え確認の1引数を追加）
        if "割り勘" in template_input:
            # 修正: 以前は amount // people（切り捨て）を正解にしつつ、CoT文では
            # 「amount ÷ people = 切り捨て値」とまるで割り切れるかのように書いており、
            # 数学的に矛盾していた（例: 2882 ÷ 8 は実際には360.25で360ではない）。
            # digit分割トークナイザー導入でモデルの掛け算・割り算精度が上がった結果、
            # モデルが正しく360.25を計算してしまい「不正解」判定される矛盾が表面化した。
            # 常に割り切れる組み合わせ（peopleの倍数をamountにする）に限定して解消する。
            people = random.randint(2, 10)
            per_person = random.randint(120, 1000)
            amount = people * per_person
            result = amount // people
            input_text = _arith_q("split", template_input, people, amount)
            cot_text = template_cot.format(amount, people, amount, people, result, result)
            answer = template_ans.format(result)
        elif "パン" in template_input:
            price = random.randint(100, 500)
            qty = random.randint(2, 20)
            result = price * qty
            input_text = _arith_q("price", template_input, price, qty)
            cot_text = template_cot.format(price, qty, price, qty, result, result)
            answer = template_ans.format(result)
        elif "%" in template_input:
            base = random.randint(100, 10000)
            pct = random.randint(5, 50)
            result = base * pct // 100
            input_text = _arith_q("percent", template_input, base, pct)
            cot_text = template_cot.format(base, pct, base, pct, result, result)
            answer = template_ans.format(result)
        elif "時速" in template_input:
            speed = random.randint(40, 120)
            hours = random.randint(1, 8)
            result = speed * hours
            input_text = _arith_q("speed", template_input, speed, hours)
            cot_text = template_cot.format(speed, hours, speed, hours, result, result)
            answer = template_ans.format(result)
        else:  # 面積
            height = random.randint(2, 50)
            width = random.randint(2, 50)
            result = height * width
            input_text = _arith_q("area", template_input, height, width)
            cot_text = template_cot.format(height, width, height, width, result, result)
            answer = template_ans.format(result)

        data.append({
            "id": f"arithmetic_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": input_text,
            "cot": cot_text,
            "answer": answer,
            "meta": {
                "source": "synthetic",
                "lang": "ja",
                "category": "arithmetic",
                "date_created": datetime.now().isoformat(),
                # cot_quality_score はここでは仮置きしない。実スコアは
                # preprocess.py の evaluate_cot_quality() が上書きする。
            }
        })
    return data


# QA知識ベース（54トピック、2025-09-15拡充）。
# generate_qa_data() の学習データ生成と、infer.py の既知トピック判定
# （knowledge_base.is_known_topic）の両方から単一の情報源として参照される。
# モジュールレベルに置くことで、学習データと推論時の「既知/未知」判定が
# 常に一致することを保証する（片方だけ更新してズレる事故を防ぐ）。
# 注意: 「Pythonでリストから重複を除く方法は？」「SQLのJOINとは？」は技術カテゴリ(TECH_SAMPLES)にも
# あり、以前はQAにも別の答えで入っていたため、同じ入力に矛盾した答えを教えていた。QA側からは削除した。
QA_KNOWLEDGE_PAIRS: list[tuple[str, str, str]] = [
        # --- 地理チェーン: 富士山 → 静岡/山梨 → 日本の地理 → 世界の地理 ---
        ("富士山の標高は？", "まず、富士山が日本最高峰の山であることを確認する。次に、標高を調べる。", "3776メートル"),
        ("富士山はどこにありますか？", "まず、富士山の所在地を確認する。静岡県と山梨県にまたがる活火山であり、両県から異なる姿が見えることで知られる。", "静岡県と山梨県"),
        ("静岡県の名産品は？", "まず、静岡県の気候が温暖で茶の栽培に適していることを確認する。そのため古くから茶の生産が盛んである。", "お茶（茶葉）"),
        ("山梨県の名産品は？", "まず、山梨県が盆地で寒暖差が大きい気候であることを確認する。この気候がぶどう栽培に適しているため、ワイン産業も盛んである。", "ぶどう・ワイン"),
        ("日本の首都はどこですか？", "まず、日本の行政区分を確認する。次に、首都の定義を確認する。日本の首都は東京都であり、関東地方に位置する。", "東京"),
        ("東京はどの地方にありますか？", "まず、日本の地方区分を確認する。東京都は関東地方に属し、政治・経済の中心である。", "関東地方"),
        ("日本で一番大きい湖は？", "まず、日本の湖の面積を比較する。滋賀県に位置する湖が最大である。", "琵琶湖"),
        ("琵琶湖はどこにありますか？", "まず、琵琶湖の所在地を確認する。近畿地方の滋賀県に位置し、京阪神の水源となっている。", "滋賀県"),
        ("日本で一番長い川は？", "まず、日本の河川の長さを比較する。新潟県などを流れる川が最長である。", "信濃川"),
        ("日本の人口はおよそ何人？", "まず、日本の総人口の統計を確認する。近年は減少傾向にある。", "約1億2000万人"),
        ("北海道の気候の特徴は？", "まず、北海道が日本最北に位置することを確認する。緯度が高いため冷涼な気候となる。", "冷涼な気候（亜寒帯）"),
        ("沖縄県の気候の特徴は？", "まず、沖縄県が日本最南に位置することを確認する。緯度が低く海に囲まれているため、一年を通して温暖である。", "亜熱帯性の温暖な気候"),
        ("世界で一番高い山は？", "まず、世界の山の標高を比較する。ヒマラヤ山脈に位置する山が最高峰である。", "エベレスト"),
        ("エベレストはどこにありますか？", "まず、エベレストの所在地を確認する。ネパールと中国（チベット）の国境に位置する。", "ネパールと中国の国境"),
        ("世界で一番長い川は？", "まず、世界の河川の長さを比較する。アフリカ大陸を流れる川が最長とされる。", "ナイル川"),
        ("世界で一番広い海は？", "まず、世界の海洋の面積を比較する。太平洋が最大の面積を持つ。", "太平洋"),

        # --- 科学チェーン: 水 → 元素・化学 → 物理 → 生物 ---
        ("水の化学式は？", "まず、水を構成する元素を確認する。水素原子2つと酸素原子1つが結合してできている。", "H2O"),
        ("水は何度で氷になりますか？", "まず、水の状態変化を確認する。標準気圧下では0℃で液体から固体（氷）に変わる。", "0℃"),
        ("水は何度で沸騰しますか？", "まず、水の沸点を確認する。標準気圧下では100℃で気体（水蒸気）に変わる。", "100℃"),
        ("空気中で一番多い気体は？", "まず、大気の組成を確認する。約78%を占める気体が最も多い。", "窒素"),
        ("人間が呼吸に使う気体は？", "まず、大気中の気体の役割を確認する。酸素は細胞のエネルギー産生に必要である。", "酸素"),
        ("光の速度はいくら？", "まず、光の速度は真空中で一定であることを確認する。正確には秒速299,792,458メートルであり、これは物理学の基本定数である。", "毎秒約30万キロメートル"),
        ("音の速さは光よりどうですか？", "まず、音は空気の振動が伝わる現象であることを確認する。空気中の音速は光速よりはるかに遅いため、雷は光ってから遅れて音が届く。", "音は光よりはるかに遅い"),
        ("地球の平均気温は？", "まず、地球全体の平均気温の統計値を確認する。産業革命以降、温室効果ガスの増加により上昇傾向にある。", "約15℃"),
        ("地球から太陽までの距離は？", "まず、地球と太陽の位置関係を確認する。この距離は天文学で1天文単位と呼ばれる基準になっている。", "約1億5000万キロメートル"),
        ("太陽系で一番大きい惑星は？", "まず、太陽系の惑星の大きさを比較する。木星は太陽系最大のガス惑星である。", "木星"),
        ("太陽系で一番小さい惑星は？", "まず、太陽系の惑星の大きさを比較する。水星は太陽に最も近く、最も小さい惑星である。", "水星"),
        ("人間の骨の数は？", "まず、成人の骨格を確認する。子供の頃より骨が融合するため、成人では数が少なくなる。", "約206個"),
        ("人間の心臓は何室ありますか？", "まず、心臓の構造を確認する。左右それぞれに心房と心室があり、血液を効率よく循環させている。", "4室（4つの部屋）"),
        ("DNAの二重螺旋構造を発見したのは？", "まず、DNA構造研究の歴史を確認する。1953年に発表され、遺伝情報の仕組み解明の基礎となった。", "ワトソン、クリック、ウィルキンス"),
        ("光合成をする生物は？", "まず、光合成の仕組みを確認する。植物は葉緑体を持ち、光エネルギーを使って二酸化炭素と水から酸素と栄養を作る。", "植物（緑色植物）"),
        ("ピタゴラスの定理とは？", "まず、直角三角形の辺の関係を確認する。斜辺の二乗が他の2辺の二乗の和に等しいという定理であり、式で表すとa² + b² = c²である。", "a² + b² = c²"),
        ("鉄が錆びる原因は？", "まず、錆びの化学反応を確認する。鉄が空気中の酸素や水分と反応して酸化することで錆が生じる。", "酸素と水分による酸化"),

        # --- IT/プログラミングチェーン: Python → データ構造 → SQL → ネットワーク ---
        ("Pythonのリストと辞書の違いは？", "まず、それぞれのデータ構造の特徴を確認する。リストは順序付きの値の並び、辞書はキーと値のペアで管理される。", "リストは値の並び、辞書はキーと値のペア"),
        ("Pythonで例外処理に使う構文は？", "まず、エラー処理の仕組みを確認する。try節でエラーが起きうる処理を囲み、except節で対処する。", "try-except"),
        ("SQLでデータを絞り込む句は？", "まず、条件に合うデータだけを取得したい場面を確認する。WHERE句を使うと指定した条件に合う行だけを抽出できる。", "WHERE句"),
        ("IPアドレスとは？", "まず、ネットワーク上の機器の識別方法を確認する。各機器に割り当てられる識別番号であり、住所のような役割を持つ。", "ネットワーク上の機器を識別する番号"),
        ("HTTPとHTTPSの違いは？", "まず、Web通信の仕組みを確認する。HTTPSは通信が暗号化されている点がHTTPと異なり、安全性が高い。", "HTTPSは通信が暗号化されている"),
        ("CPUの役割は？", "まず、コンピュータの構成要素を確認する。CPUはプログラムの命令を実行する、いわばコンピュータの頭脳である。", "命令を実行する演算装置（コンピュータの頭脳）"),
        ("RAMとストレージの違いは？", "まず、それぞれの役割を確認する。RAMは作業中のデータを一時的に保持し、電源を切ると消える。ストレージはデータを長期的に保存する。", "RAMは一時記憶、ストレージは長期保存"),
        ("AIとは何の略ですか？", "まず、この言葉の由来を確認する。Artificial Intelligenceの略で、人間の知的な作業を機械が行う技術を指す。", "Artificial Intelligence（人工知能）"),

        # --- 歴史チェーン: 日本史 → 世界史 ---
        ("江戸幕府を開いたのは誰ですか？", "まず、江戸時代の始まりを確認する。1603年に征夷大将軍に任命され、幕府を開いた。", "徳川家康"),
        ("明治維新が起きたのはいつ頃ですか？", "まず、江戸幕府が終わった経緯を確認する。1868年前後に政治体制が大きく変わり、近代化が進んだ。", "1868年前後"),
        ("日本国憲法が施行されたのはいつですか？", "まず、戦後の日本の歩みを確認する。1947年に施行され、国民主権・平和主義・基本的人権の尊重を柱としている。", "1947年"),
        ("フランス革命が起きたのはいつですか？", "まず、フランスの王政の歴史を確認する。1789年に始まり、身分制社会が崩れる大きな転換点となった。", "1789年"),
        ("第二次世界大戦が終わったのはいつですか？", "まず、20世紀の大きな戦争の歴史を確認する。1945年に終結し、国際連合の設立につながった。", "1945年"),
        ("万里の長城はどこの国にありますか？", "まず、万里の長城が築かれた目的を確認する。北方民族の侵入を防ぐために建設された、中国の建造物である。", "中国"),

        # --- 単語1語だけの入力にも対応できるよう、短い入力パターンも用意 ---
        ("富士山", "まず、富士山が何かを確認する。日本最高峰の山で、標高3776メートル、静岡県と山梨県にまたがる活火山である。", "日本最高峰の山（標高3776メートル）"),
        ("東京", "まず、東京が何かを確認する。日本の首都であり、関東地方に位置する政治・経済の中心地である。", "日本の首都"),
        ("Python", "まず、Pythonが何かを確認する。読みやすい文法が特徴のプログラミング言語であり、AI開発などにも広く使われる。", "プログラミング言語の一種"),
        ("水", "まず、水が何かを確認する。化学式H2O、水素と酸素からなる、生命に不可欠な物質である。", "H2Oで表される化合物"),
        ("エベレスト", "まず、エベレストが何かを確認する。ネパールと中国の国境にある、世界最高峰の山である。", "世界最高峰の山"),
]


def generate_qa_data(count: int) -> list[dict]:
    """高品質QAペア（20%）。まず/よって等の接続詞を入れてステップ性を明示する。
    CoTの末尾は必ず「よって、答えは○○である。」で終える
    （EM評価で解答部分を機械的に抽出できるようにするため。算数カテゴリと同じ設計）。

    2025-09-15 拡充: 8件→60件超に増量。単発の一問一答ではなく、
    地理・科学・IT・歴史それぞれを「1つのトピックから関連トピックへ連想的に
    広げる」形で構成し、CoTには単なる事実の暗唱でなく「なぜそうなるか」の
    理由・背景も一言添えて意味理解を促す（ユーザーの提案に基づく方針）。
    生Webスクレイピングではなく、Gate1のCoT品質基準を満たす統一フォーマットの
    手作りデータとして追加している。

    2025-09-XX: qa_pairsをモジュールレベルのQA_KNOWLEDGE_PAIRSに切り出し、
    infer.py側の既知トピック判定（knowledge_base.py）と共有するようにした。
    """
    qa_pairs = QA_KNOWLEDGE_PAIRS

    def _vary_question(q: str) -> str:
        """質問文の聞き方をランダムに変える（言い換えバンクを優先して使う）。

        以前は「{stem}を教えて」「{topic}といえば？」などの機械的な派生形を使っていたが、
          - 「富士山の標高はを教えて」のような不自然な日本語になる
          - 「エベレストといえば？」のように、同じ入力に複数の事実（別々の答え）が対応し、
            矛盾した教師信号になる
        という問題があったため廃止した。代わりに src/paraphrase_bank.py の自然な言い換えを使う。
        """
        return vary_question(q)

    data = []
    for i in range(count):
        q, explanation, ans = random.choice(qa_pairs)
        q_var = _vary_question(q)
        cot_text = f"{explanation}よって、答えは{ans}である。"
        data.append({
            "id": f"qa_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": q_var,
            "cot": cot_text,
            "answer": ans,
            "meta": {
                "source": "synthetic",
                "lang": "ja",
                "category": "qa",
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


TECH_SAMPLES: list[tuple[str, str, str]] = [
    ("Pythonでリストから重複を除く方法は？",
     "まず、集合（set）を使う方法を検討する。list(set(original_list))で重複なしリストが得られる。"
     "次に、順序保持が必要か確認する。順序を保ちたい場合は辞書を使う。"
     "よって、list(dict.fromkeys(original_list))を使うのがよい。",
     "set()を使う方法またはdict.fromkeys()を使う方法"),
    ("JSONファイルの読み込み方法は？",
     "まず、Pythonの標準ライブラリを確認する。jsonモジュールが利用できる。"
     "次に、読み込みコードを組み立てる。よって、"
     "import json; data = json.load(open('file.json'))で読み込める。",
     "json.load()を使用"),
    ("SQLのJOINとは？",
     "まず、複数テーブルの結合が必要な場面を確認する。次に、JOINの種類を整理する。"
     "INNER JOIN、LEFT JOIN、RIGHT JOIN、FULL OUTER JOINがある。"
     "よって、目的に応じて適切なJOINを選択する。",
     "複数テーブルを結合する操作"),
]


def generate_technical_data(count: int) -> list[dict]:
    """技術文書・仕様書（25%）。まず/次に/よって等の接続詞でステップ性を明示する。"""
    tech_samples = TECH_SAMPLES

    data = []
    for i in range(count):
        q, explanation, ans = random.choice(tech_samples)
        data.append({
            "id": f"technical_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": vary_question(q),
            "cot": explanation,
            "answer": ans,
            "meta": {
                "source": "synthetic",
                "lang": "ja",
                "category": "technical",
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


CODE_SAMPLES: list[tuple[str, str, str]] = [
    ("Pythonでリストをソートするコードは？",
     "まず、対象のリストを確認する。numbers = [3, 1, 4, 1, 5]。"
     "次に、sorted()関数を使う。sorted_numbers = sorted(numbers)  # 昇順。"
     "よって、降順にしたい場合は reversed_numbers = sorted(numbers, reverse=True) とする。",
     "sorted(numbers) または sorted(numbers, reverse=True)"),
    ("for ループで0から9まで出力するコードは？",
     "まず、0から9までの範囲を確認する。次に、range(10)を使う。"
     "よって、for i in range(10):\n    print(i) と書けば0から9までを出力できる。",
     "for i in range(10):\n    print(i)"),
    ("Pythonでリスト内包表記を使って偶数だけ抽出するコードは？",
     "まず、元のリストを確認する。numbers = [1, 2, 3, 4, 5, 6]。"
     "次に、条件付きリスト内包表記を組み立てる。"
     "よって、evens = [n for n in numbers if n % 2 == 0] と書ける。",
     "[n for n in numbers if n % 2 == 0]"),
    ("Pythonで辞書の値でループ処理するコードは？",
     "まず、対象の辞書を確認する。d = {'a': 1, 'b': 2}。"
     "次に、items()メソッドを使う。"
     "よって、for key, value in d.items():\n    print(key, value) と書ける。",
     "for key, value in d.items(): print(key, value)"),
    ("Pythonで文字列を分割するコードは？",
     "まず、対象の文字列を確認する。s = 'apple,banana,orange'。"
     "次に、split()メソッドを使う。"
     "よって、parts = s.split(',') と書けば ['apple', 'banana', 'orange'] が得られる。",
     "s.split(',')"),
    ("Pythonでファイルを1行ずつ読み込むコードは？",
     "まず、対象ファイルを開く必要があることを確認する。次に、withブロックを使う。"
     "よって、with open('file.txt') as f:\n    for line in f:\n        print(line) と書ける。",
     "with open('file.txt') as f:\n    for line in f: print(line)"),
    ("Pythonで例外処理を書くコードは？",
     "まず、エラーが起きうる処理を確認する。次に、try/exceptで囲む。"
     "よって、try:\n    x = 1 / 0\nexcept ZeroDivisionError:\n    print('ゼロ除算エラー') と書ける。",
     "try/exceptでZeroDivisionErrorを捕捉する"),
    ("Pythonで関数を定義するコードは？",
     "まず、defキーワードで関数を定義する。次に、引数と戻り値を決める。"
     "よって、def add(a, b):\n    return a + b と書ける。",
     "def add(a, b): return a + b"),
    ("Pythonでクラスを定義するコードは？",
     "まず、classキーワードでクラスを定義する。次に、__init__メソッドで初期化する。"
     "よって、class Dog:\n    def __init__(self, name):\n        self.name = name と書ける。",
     "class Dog: def __init__(self, name): self.name = name"),
    ("Pythonでリストの最大値を求めるコードは？",
     "まず、対象のリストを確認する。numbers = [3, 7, 2, 9, 4]。"
     "次に、max()関数を使う。"
     "よって、biggest = max(numbers) と書けば9が得られる。",
     "max(numbers)"),
    ("Pythonで文字列を数値に変換するコードは？",
     "まず、対象の文字列を確認する。s = '123'。"
     "次に、int()関数を使う。よって、n = int(s) と書けば整数123に変換できる。",
     "int(s)"),
    ("Pythonでリストの要素数を数えるコードは？",
     "まず、対象のリストを確認する。numbers = [1, 2, 3, 4]。"
     "次に、len()関数を使う。よって、count = len(numbers) と書けば4が得られる。",
     "len(numbers)"),
    ("Pythonで辞書にキーが存在するか確認するコードは？",
     "まず、対象の辞書を確認する。d = {'a': 1}。"
     "次に、in演算子を使う。よって、if 'a' in d:\n    print('存在する') と書ける。",
     "if 'a' in d: ..."),
    ("Pythonでリストに要素を追加するコードは？",
     "まず、対象のリストを確認する。numbers = [1, 2, 3]。"
     "次に、append()メソッドを使う。よって、numbers.append(4) と書けば末尾に4が追加される。",
     "numbers.append(4)"),
    ("Pythonで2つのリストをまとめて処理するコードは？",
     "まず、対象の2つのリストを確認する。names = ['a', 'b']、ages = [10, 20]。"
     "次に、zip()関数を使う。"
     "よって、for name, age in zip(names, ages):\n    print(name, age) と書ける。",
     "for name, age in zip(names, ages): print(name, age)"),
    ("Pythonでリストの中身を逆順にするコードは？",
     "まず、対象のリストを確認する。numbers = [1, 2, 3]。"
     "次に、reversed()関数かスライスを使う。"
     "よって、reversed_list = numbers[::-1] と書けば逆順になる。",
     "numbers[::-1]"),
    ("Pythonで文字列を連結するコードは？",
     "まず、対象の文字列を確認する。a = 'Hello'、b = 'World'。"
     "次に、+演算子かf文字列を使う。よって、result = f'{a}, {b}!' と書ける。",
     "f'{a}, {b}!'"),
    ("PythonでCSVファイルを読み込むコードは？",
     "まず、標準ライブラリを確認する。csvモジュールが使える。"
     "次に、読み込みコードを組み立てる。"
     "よって、import csv\nwith open('file.csv') as f:\n    reader = csv.reader(f) と書ける。",
     "csvモジュールのcsv.reader()を使用"),
    ("Pythonでリストから条件に合う最初の要素を探すコードは？",
     "まず、対象のリストを確認する。numbers = [1, 3, 5, 8, 9]。"
     "次に、next()とジェネレータ式を使う。"
     "よって、result = next(n for n in numbers if n % 2 == 0) と書けば8が得られる。",
     "next(n for n in numbers if n % 2 == 0)"),
    ("Pythonで辞書を値でソートするコードは？",
     "まず、対象の辞書を確認する。d = {'a': 3, 'b': 1, 'c': 2}。"
     "次に、sorted()とitems()、key引数を使う。"
     "よって、sorted(d.items(), key=lambda x: x[1]) と書けば値の昇順になる。",
     "sorted(d.items(), key=lambda x: x[1])"),
    ("Pythonで無限ループを書くコードは？",
     "まず、終了条件を必ず用意することを確認する。次に、while Trueとbreakを使う。"
     "よって、while True:\n    if condition:\n        break と書ける。",
     "while True: ... break"),
]



def generate_code_data(count: int) -> list[dict]:
    """コード＋解説（15%）。まず/次に/よって等の接続詞でステップ性を明示する。"""
    code_samples = CODE_SAMPLES

    data = []
    for i in range(count):
        q, explanation, ans = random.choice(code_samples)
        data.append({
            "id": f"code_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": vary_question(q),
            "cot": explanation,
            "answer": ans,
            "meta": {
                "source": "synthetic",
                "lang": "ja",
                "category": "code",
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


CONVERSATION_SAMPLES: list[tuple[str, str, str]] = [
    ("こんにちは", "まず、挨拶を受け取る。次に、丁寧に返答する。こんにちは。お疲れ様です。",
     "こんにちは。お疲れ様です。"),
    ("今日の天気はどう？", "まず、天気の状況を確認する。次に、要点を伝える。雲が多いですが雨は降らないでしょう。",
     "雲が多いですが雨は降らないでしょう。"),
    ("昼食は何を食べた？", "まず、質問の内容を確認する。次に、実際の行動を答える。ラーメンを食べました。",
     "ラーメンを食べました。"),
    ("ありがとう", "まず、感謝の言葉を受け取る。次に、丁寧に返答する。どういたしまして。お役に立てて嬉しいです。",
     "どういたしまして。お役に立てて嬉しいです。"),
    ("さようなら", "まず、別れの挨拶を受け取る。次に、丁寧に返答する。さようなら。また話しましょう。",
     "さようなら。また話しましょう。"),
    ("お元気ですか？", "まず、体調を尋ねられていることを確認する。次に、丁寧に返答する。おかげさまで元気です。ありがとうございます。",
     "おかげさまで元気です。ありがとうございます。"),
    ("疲れた", "まず、相手の状態を受け止める。次に、労いの言葉をかける。お疲れ様です。無理をしないでくださいね。",
     "お疲れ様です。無理をしないでくださいね。"),
    ("よろしくお願いします", "まず、挨拶を受け取る。次に、丁寧に返答する。こちらこそ、よろしくお願いします。",
     "こちらこそ、よろしくお願いします。"),
]


def generate_conversation_data(count: int) -> list[dict]:
    """短文会話・対話例（10%）。短い会話でも接続詞を使い最低限のステップ性を持たせる。"""
    conversations = CONVERSATION_SAMPLES

    data = []
    for i in range(count):
        input_text, response, ans = random.choice(conversations)
        data.append({
            "id": f"conversation_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": vary_question(input_text),
            "cot": response,
            "answer": ans,
            "meta": {
                "source": "synthetic",
                "lang": "ja",
                "category": "conversation",
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


def main():
    parser = argparse.ArgumentParser(description="合成学習データ生成")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--num_samples", type=int, default=50000, help="総サンプル数")
    parser.add_argument("--seed", type=int, default=42, help="乱数seed")
    parser.add_argument("--cot_ratio", type=float, default=1.0,
                        help="CoT（思考過程）を付与するサンプルの割合（0.0〜1.0）。"
                             "plan §Gate3 のCoT比率A/B実験（cot20/cot40/cot60）用。"
                             "1.0未満にすると、その割合の残りのサンプルは cot フィールドを"
                             "空にし、[BOS] input answer [EOS] という直接回答形式にする"
                             "（generate()側の挙動は変えず、データ側だけで制御する）。")
    parser.add_argument("--version", default="v001",
                        help="出力ファイル名に使うバージョンタグ（例: v001, v_cot40）。"
                             "異なる実験のデータを別ファイルとして共存させるために使う。")
    args = parser.parse_args()

    random.seed(args.seed)

    # 比率に従ってサンプル数を配分（plan §2.1）
    ratios = {
        "arithmetic": 0.30,    # 数学・論理問題
        "technical": 0.25,     # 技術文書
        "code": 0.15,          # コード
        "qa": 0.20,            # QA
        "conversation": 0.10,  # 会話
    }

    all_data = []

    print(f"合成データ生成開始（全{args.num_samples}件、cot_ratio={args.cot_ratio}）...")

    # 各カテゴリごとに生成
    for category, ratio in ratios.items():
        count = int(args.num_samples * ratio)
        print(f"  {category}: {count}件を生成中...", end="", flush=True)

        if category == "arithmetic":
            data = generate_arithmetic_data(count)
        elif category == "technical":
            data = generate_technical_data(count)
        elif category == "code":
            data = generate_code_data(count)
        elif category == "qa":
            data = generate_qa_data(count)
        elif category == "conversation":
            data = generate_conversation_data(count)

        all_data.extend(data)
        print(f" ✓")

    # CoT比率の適用: cot_ratio未満の割合のサンプルは cot を空にする
    # （カテゴリを問わず全体に対して一様に適用。plan Gate3のA/B実験用）
    if args.cot_ratio < 1.0:
        cot_kept = 0
        for item in all_data:
            if random.random() < args.cot_ratio:
                item["meta"]["has_cot"] = True
                cot_kept += 1
            else:
                item["cot"] = ""
                item["meta"]["has_cot"] = False
        print(f"  CoT付与: {cot_kept}/{len(all_data)}件（目標比率 {args.cot_ratio:.0%}）")
    else:
        for item in all_data:
            item["meta"]["has_cot"] = True

    # シャッフル
    random.shuffle(all_data)

    # 9:1 で train/val 分割
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # train.jsonl を保存
    train_path = f"{args.output_dir}/{args.version}.train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ {train_path} 保存完了（{len(train_data)}件）")

    # val.jsonl を保存
    val_path = f"{args.output_dir}/{args.version}.val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ {val_path} 保存完了（{len(val_data)}件）")

    # corpus.txt（SentencePiece 学習用）を出力
    corpus_path = f"{args.output_dir}/{args.version}_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(item["input"] + " ")
            if item.get("cot"):
                f.write(item["cot"] + " ")
            f.write(item.get("answer", "") + "\n")
    print(f"✓ {corpus_path} 保存完了（SentencePiece学習用）")

    print("\n✅ データ生成完了")
    print(f"   train: {len(train_data)}件")
    print(f"   val: {len(val_data)}件")


if __name__ == "__main__":
    main()
