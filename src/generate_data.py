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
            input_text = template_input.format(people, amount)
            cot_text = template_cot.format(amount, people, amount, people, result, result)
            answer = template_ans.format(result)
        elif "パン" in template_input:
            price = random.randint(100, 500)
            qty = random.randint(2, 20)
            result = price * qty
            input_text = template_input.format(price, qty)
            cot_text = template_cot.format(price, qty, price, qty, result, result)
            answer = template_ans.format(result)
        elif "%" in template_input:
            base = random.randint(100, 10000)
            pct = random.randint(5, 50)
            result = base * pct // 100
            input_text = template_input.format(base, pct)
            cot_text = template_cot.format(base, pct, base, pct, result, result)
            answer = template_ans.format(result)
        elif "時速" in template_input:
            speed = random.randint(40, 120)
            hours = random.randint(1, 8)
            result = speed * hours
            input_text = template_input.format(speed, hours)
            cot_text = template_cot.format(speed, hours, speed, hours, result, result)
            answer = template_ans.format(result)
        else:  # 面積
            height = random.randint(2, 50)
            width = random.randint(2, 50)
            result = height * width
            input_text = template_input.format(height, width)
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
    """
    qa_pairs = [
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
        ("Pythonでリストから重複を除く方法は？", "まず、集合（set）を使う方法を検討する。集合は同じ値を1つしか持てない性質があるため重複除去に使える。", "set()を使う方法"),
        ("Pythonのリストと辞書の違いは？", "まず、それぞれのデータ構造の特徴を確認する。リストは順序付きの値の並び、辞書はキーと値のペアで管理される。", "リストは値の並び、辞書はキーと値のペア"),
        ("Pythonで例外処理に使う構文は？", "まず、エラー処理の仕組みを確認する。try節でエラーが起きうる処理を囲み、except節で対処する。", "try-except"),
        ("SQLのJOINとは？", "まず、複数テーブルの結合が必要な場面を確認する。共通のキーを使って別々のテーブルのデータを1つにまとめる操作である。", "複数テーブルを結合する操作"),
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

    def _vary_question(q: str) -> str:
        """質問文の意味を変えずに、聞き方（言い回し）だけをランダムに変える。

        以前は「？」の有無だけの単純なバリエーションしかなく、
        「日本の首都はどこですか？」という固定文型にしか対応できなかった。
        「〜を教えて」「〜について教えて」等、聞き方自体の多様なパターンを
        混ぜることで、未知の言い回しへの汎化性能向上を狙う
        （sanity_checkで、固定文型からわずかに外れた質問に弱いことが判明したため）。
        """
        base = q.rstrip("？")
        variants = [q]  # 元の文はそのまま候補に残す

        if base.endswith("とは"):
            # 「ピタゴラスの定理とは」→「〜とはについて教えて」のような
            # 「とは」の二重表現になるのを防ぐため、「とは」を除去してから繋げる
            stem = base[:-2]
            variants.append(f"{stem}とは？")
            variants.append(f"{stem}について教えて")
            variants.append(f"{stem}とは何ですか？")
        elif base.endswith("ですか"):
            stem = base[:-3]  # "ですか" を除去
            variants.append(f"{stem}？")
            variants.append(f"{stem}を教えて")
            variants.append(f"{stem}について教えて")
        else:
            variants.append(f"{base}？")
            variants.append(f"{base}を教えて")
            variants.append(f"{base}について教えて")

        return random.choice(variants)

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


def generate_technical_data(count: int) -> list[dict]:
    """技術文書・仕様書（25%）。まず/次に/よって等の接続詞でステップ性を明示する。"""
    tech_samples = [
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

    data = []
    for i in range(count):
        q, explanation, ans = random.choice(tech_samples)
        data.append({
            "id": f"technical_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": q,
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


def generate_code_data(count: int) -> list[dict]:
    """コード＋解説（15%）。まず/次に/よって等の接続詞でステップ性を明示する。"""
    code_samples = [
        ("Pythonでリストをソートするコードは？",
         "まず、対象のリストを確認する。numbers = [3, 1, 4, 1, 5]。"
         "次に、sorted()関数を使う。sorted_numbers = sorted(numbers)  # 昇順。"
         "よって、降順にしたい場合は reversed_numbers = sorted(numbers, reverse=True) とする。",
         "sorted()関数を使用"),
        ("for ループで0から9まで出力するコードは？",
         "まず、0から9までの範囲を確認する。次に、range(10)を使う。"
         "よって、for i in range(10):\n    print(i) と書けば0から9までを出力できる。",
         "for i in range(10): print(i)"),
    ]

    data = []
    for i in range(count):
        q, explanation, ans = random.choice(code_samples)
        data.append({
            "id": f"code_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": q,
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


def generate_conversation_data(count: int) -> list[dict]:
    """短文会話・対話例（10%）。短い会話でも接続詞を使い最低限のステップ性を持たせる。"""
    conversations = [
        ("こんにちは", "まず、挨拶を受け取る。次に、丁寧に返答する。こんにちは。お疲れ様です。", "挨拶への応答"),
        ("今日の天気はどう？", "まず、天気の状況を確認する。次に、要点を伝える。雲が多いですが雨は降らないでしょう。", "天気についての応答"),
        ("昼食は何を食べた？", "まず、質問の内容を確認する。次に、実際の行動を答える。ラーメンを食べました。", "行動についての応答"),
    ]

    data = []
    for i in range(count):
        input_text, response, ans = random.choice(conversations)
        data.append({
            "id": f"conversation_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": input_text,
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

    print(f"合成データ生成開始（全{args.num_samples}件）...")

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

    # シャッフル
    random.shuffle(all_data)

    # 9:1 で train/val 分割
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # train.jsonl を保存
    train_path = f"{args.output_dir}/v001.train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ {train_path} 保存完了（{len(train_data)}件）")

    # val.jsonl を保存
    val_path = f"{args.output_dir}/v001.val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ {val_path} 保存完了（{len(val_data)}件）")

    # corpus.txt（SentencePiece 学習用）を出力
    corpus_path = f"{args.output_dir}/corpus.txt"
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
