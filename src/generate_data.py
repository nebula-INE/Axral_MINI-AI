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
    """数学・論理問題（CoT付き、30%）"""
    templates = [
        # 割り算: (input_template, cot_template, answer_template)
        ("{}人で{}円を割り勘すると1人いくら？",
         "合計額を確認する。{}円。人数を確認する。{}人。割り勘額を計算する。{} ÷ {} = {}。",
         "{}円"),
        # 掛け算
        ("1個{}円のパンを{}個買うといくら？",
         "単価を確認する。{}円。数量を確認する。{}個。合計額を計算する。{} × {} = {}。",
         "{}円"),
        # 割合
        ("{}円の{}%はいくら？",
         "基準額を確認する。{}円。割合を確認する。{}%。割合金額を計算する。{} × {} ÷ 100 = {}。",
         "{}円"),
        # 時間・距離
        ("時速{}kmで{}時間走ると何km進む？",
         "速度を確認する。時速{}km。時間を確認する。{}時間。距離を計算する。{} × {} = {}。",
         "{}km"),
        # 面積
        ("縦{}m、横{}mの長方形の面積は？",
         "縦の長さを確認する。{}m。横の長さを確認する。{}m。面積を計算する。{} × {} = {}。",
         "{}m²"),
    ]

    data = []
    for i in range(count):
        template_input, template_cot, template_ans = random.choice(templates)
        # テンプレートに合わせた具体値を生成
        if "割り勘" in template_input:
            amount = random.randint(1200, 10000)
            people = random.randint(2, 10)
            result = amount // people
            input_text = template_input.format(people, amount)
            cot_text = template_cot.format(amount, people, amount, people, result)
            answer = template_ans.format(result)
        elif "パン" in template_input:
            price = random.randint(100, 500)
            qty = random.randint(2, 20)
            result = price * qty
            input_text = template_input.format(price, qty)
            cot_text = template_cot.format(price, qty, price, qty, result)
            answer = template_ans.format(result)
        elif "%" in template_input:
            base = random.randint(100, 10000)
            pct = random.randint(5, 50)
            result = base * pct // 100
            input_text = template_input.format(base, pct)
            cot_text = template_cot.format(base, pct, base, pct, result)
            answer = template_ans.format(result)
        elif "時速" in template_input:
            speed = random.randint(40, 120)
            hours = random.randint(1, 8)
            result = speed * hours
            input_text = template_input.format(speed, hours)
            cot_text = template_cot.format(speed, hours, speed, hours, result)
            answer = template_ans.format(result)
        else:  # 面積
            height = random.randint(2, 50)
            width = random.randint(2, 50)
            result = height * width
            input_text = template_input.format(height, width)
            cot_text = template_cot.format(height, width, height, width, result)
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
                "cot_quality_score": 0.85,  # 高品質に設定（テンプレートベース）
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


def generate_qa_data(count: int) -> list[dict]:
    """高品質QAペア（20%）"""
    qa_pairs = [
        ("日本の首都はどこですか？", "日本の首都は東京都である。東京は関東地方に位置し、日本の政治・経済の中心。", "東京"),
        ("富士山の標高は？", "富士山は日本最高峰の山であり、標高は3776メートル。", "3776メートル"),
        ("水の化学式は？", "水は水素と酸素から構成される化合物。化学式はH2O。", "H2O"),
        ("光の速度はいくら？", "光の速度は真空中で毎秒約30万キロメートル。正確には299,792,458 m/s。", "毎秒約30万キロメートル"),
        ("地球の平均気温は？", "地球の平均気温は約15℃。ただし産業革命以降は上昇傾向。", "約15℃"),
        ("人間の骨の数は？", "成人の人間の体には約206個の骨がある。", "約206個"),
        ("DNAの二重螺旋構造を発見したのは？", "DNAの二重螺旋構造はワトソン、クリック、ウィルキンスにより発見された。", "ワトソン、クリック、ウィルキンス"),
        ("ピタゴラスの定理とは？", "ピタゴラスの定理は、直角三角形の斜辺の二乗が他の2辺の二乗の和に等しいというもの。式: a² + b² = c²。", "a² + b² = c²"),
    ]

    data = []
    for i in range(count):
        q, explanation, ans = random.choice(qa_pairs)
        # 質問に多少バリエーションをつける
        q_var = q if i % 3 != 0 else q.replace("？", "")
        data.append({
            "id": f"qa_{datetime.now().strftime('%Y%m%d')}_{i:05d}",
            "input": q_var,
            "cot": explanation,
            "answer": ans,
            "meta": {
                "source": "synthetic",
                "lang": "ja",
                "category": "qa",
                "cot_quality_score": 0.80,
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


def generate_technical_data(count: int) -> list[dict]:
    """技術文書・仕様書（25%）"""
    tech_samples = [
        ("Pythonでリストから重複を除く方法は？",
         "集合（set）を使う方法が最も簡単。list(set(original_list))で重複なしリストが得られる。ただし順序は保証されない。順序を保ちたい場合は辞書を使う: list(dict.fromkeys(original_list))。",
         "set()を使う方法またはdict.fromkeys()を使う方法"),
        ("JSONファイルの読み込み方法は？",
         "Pythonのjsonモジュールを使う。import json; data = json.load(open('file.json'))で読み込める。",
         "json.load()を使用"),
        ("SQLのJOINとは？",
         "SQLのJOINは複数のテーブルを結合する操作。INNER JOIN、LEFT JOIN、RIGHT JOIN、FULL OUTER JOINがある。",
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
                "cot_quality_score": 0.78,
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


def generate_code_data(count: int) -> list[dict]:
    """コード＋解説（15%）"""
    code_samples = [
        ("Pythonでリストをソートするコードは？",
         "numbers = [3, 1, 4, 1, 5]\nsorted_numbers = sorted(numbers)  # 昇順\nreversed_numbers = sorted(numbers, reverse=True)  # 降順\nこれでソート済みリストが得られる。",
         "sorted()関数を使用"),
        ("for ループで0から9まで出力するコードは？",
         "for i in range(10):\n    print(i)\nrange(10)は0から9までの数列を生成する。",
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
                "cot_quality_score": 0.75,
                "date_created": datetime.now().isoformat(),
            }
        })
    return data


def generate_conversation_data(count: int) -> list[dict]:
    """短文会話・対話例（10%）"""
    conversations = [
        ("こんにちは", "こんにちは。お疲れ様です。", "挨拶への応答"),
        ("今日の天気はどう？", "雲が多いですが雨は降らないでしょう。", "天気についての応答"),
        ("昼食は何を食べた？", "ラーメンを食べました。", "行動についての応答"),
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
                "cot_quality_score": 0.70,
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
