"""
arith_cot.py
算数の「手順つきCoT」（1ステップ=ごく小さな計算）の生成と、1ステップあたりの計算量の測定。

背景:
  現行の算数CoTは「6574 × 35 ÷ 100 = 2300」のように、4桁×2桁の掛け算と割り算を1つの式で要求する。
  小型モデルには、この1ステップが大きすぎる。そこで人間の筆算と同じく、位ごとに分解して
  「1桁×1桁」「2数の足し算」「位取り（×10、÷100）」だけの式に刻む。

式の書式（eval_cot.verify_arithmetic の正規表現 `[\\d.\\s+\\-*/×÷]+=\\s*[-\\d.]+` に合わせる）:
  - 1つの式は「数 演算 数 = 数」。式の直前は必ず日本語で区切る（数字が左辺に混ざらないように）。
  - 割り算の余りは「= 3余り1」のように式の外（余り）に書く。
  - 最後の式の右辺は必ず最終答え（verify_arithmetic が最終回答との対応を確認できる）。
  - 改行は使わず「。」区切り（最終回答の抽出が「最後の。区切り区間」のため）。

使い方:
  python -m src.arith_cot --demo           # 各種の例を表示
  python -m src.arith_cot --audit          # 現行(oneshot)と新(stepwise)の計算量を比較
"""
from __future__ import annotations

import argparse
import random
import re
from fractions import Fraction
from math import gcd

# --------------------------------------------------------------------------- 数値の範囲
# 既定は現行 generate_arithmetic_data と同じ範囲（比較を公平にするため）。
# wide は「桁数が増えても手順が通用するか」を見るための広い範囲（学習範囲外の数字への汎化用）。
RANGES = {
    "default": {"split": ((2, 10), (120, 1000)), "price": ((100, 500), (2, 20)),
                "percent": ((100, 10000), (5, 50)), "speed": ((40, 120), (1, 8)),
                "area": ((2, 50), (2, 50))},
    "wide": {"split": ((2, 20), (120, 1000)), "price": ((100, 999), (2, 30)),
             "percent": ((100, 20000), (5, 95)), "speed": ((40, 250), (1, 9)),
             "area": ((2, 99), (2, 99))},
}


# --------------------------------------------------------------------------- 手順の部品
def _fmt(n: int) -> str:
    return str(n)


def _digits_places(n: int) -> list[tuple[int, int]]:
    """n を位ごとに分解し、(その位の数字, 位の重み) を大きい位から返す。0の位は飛ばす。
    例: 275 -> [(2,100),(7,10),(5,1)]"""
    s = str(n)
    out = []
    for i, ch in enumerate(s):
        d = int(ch)
        if d:
            out.append((d, 10 ** (len(s) - 1 - i)))
    return out


def _add_chain(terms: list[int]) -> tuple[list[str], int]:
    """項を左から順に2つずつ足す。例: [600,210,15] -> ['600 + 210 = 810', '810 + 15 = 825']"""
    steps = []
    acc = terms[0]
    for t in terms[1:]:
        nxt = acc + t
        steps.append(f"{acc} + {t} = {nxt}")
        acc = nxt
    return steps, acc


def _mul_single_digit(a: int, d: int) -> tuple[list[str], int]:
    """a × d（dは1桁）。aが2桁以上なら位ごとに分けて「(桁×位) × d」を計算し、足し合わせる。"""
    if a < 10:
        r = a * d
        return [f"{a} × {d} = {r}"], r
    parts = _digits_places(a)
    steps, terms = [], []
    for digit, weight in parts:
        v = digit * weight
        r = v * d
        steps.append(f"{v} × {d} = {r}")
        terms.append(r)
    if len(terms) == 1:
        return steps, terms[0]
    add_steps, total = _add_chain(terms)
    return steps + add_steps, total


def mul_steps(a: int, b: int) -> tuple[list[str], int]:
    """a × b を、1桁×1桁・位取り・2数の足し算だけに分解した式の列と結果を返す。
    bを位ごとに分け、各位について a × (その位の数字) を求めてから位取り(×10^k)を行い、最後に足す。"""
    if b < 10:
        return _mul_single_digit(a, b)
    steps: list[str] = []
    partials: list[int] = []
    for digit, weight in _digits_places(b):
        if digit == 1:
            r = a
        else:
            sub, r = _mul_single_digit(a, digit)
            steps += sub
        if weight > 1:
            shifted = r * weight
            steps.append(f"{r} × {weight} = {shifted}")
            r = shifted
        partials.append(r)
    if len(partials) == 1:
        return steps, partials[0]
    add_steps, total = _add_chain(partials)
    return steps + add_steps, total


def div_steps(total: int, d: int) -> tuple[list[str], int]:
    """total ÷ d（割り切れる前提）を、筆算と同じく上の位から1桁ずつ割る式の列にする。
    各ステップは「(前の余り×10＋次の桁) ÷ d = 商の1桁 余り r」。
    最後に商の各桁を位取りして足し、最終式の右辺を商そのものにする。"""
    assert total % d == 0, "割り切れる組み合わせだけを扱う"
    digits = [int(c) for c in str(total)]
    steps: list[str] = []
    rem = 0
    q_digits: list[int] = []
    started = False
    for i, dg in enumerate(digits):
        cur = rem * 10 + dg
        if not started and cur < d and i < len(digits) - 1:
            rem = cur  # まだ割れない上位の桁は次の桁とまとめる
            continue
        started = True
        q, rem = divmod(cur, d)
        q_digits.append(q)
        steps.append(f"{cur} ÷ {d} = {q}余り{rem}")
    n = len(q_digits)
    result = int("".join(str(q) for q in q_digits))
    terms = [q * 10 ** (n - 1 - i) for i, q in enumerate(q_digits) if q]
    if len(terms) > 1:
        steps.append(" + ".join(str(t) for t in terms) + f" = {result}")
    elif result >= 10:
        # 商が「1桁×位取り」1項だけ（例: 300）。最後の割り算の右辺は0になるので、
        # 位取りの式を足して最終式の右辺を商に揃える。
        lead = next(q for q in q_digits if q)
        steps.append(f"{lead} × {result // lead} = {result}")
    return steps, result


def shift_div_100_steps(p: int) -> tuple[list[str], int]:
    """p ÷ 100（pは100の倍数）。位取り（下2桁を落とす）の1ステップ。"""
    assert p % 100 == 0
    return [f"{p} ÷ 100 = {p // 100}"], p // 100


# --------------------------------------------------------------------------- 問題種別ごとのCoT
def _join(sentences: list[str]) -> str:
    return "".join(s if s.endswith("。") else s + "。" for s in sentences)


def cot_multiply(label_a: str, a: int, unit_a: str, label_b: str, b: int, unit_b: str,
                 what: str, result_unit_phrase: str) -> tuple[str, int]:
    """掛け算問題の手順つきCoT。what=「合計額」等、result_unit_phrase=「円」「km」「m²」等。"""
    eqs, result = mul_steps(a, b)
    head = [f"まず、{label_a}を確認する。{a}{unit_a}である。",
            f"次に、{label_b}を確認する。{b}{unit_b}である。"]
    if len(eqs) == 1:
        body = [f"よって、{what}を計算する。{eqs[0]}"]
    else:
        body = [f"よって、{what}を位ごとに分けて計算する。{eqs[0]}"] + eqs[1:]
    tail = [f"したがって、答えは{result}{result_unit_phrase}である。"]
    return _join(head + body + tail), result


def cot_split(people: int, amount: int) -> tuple[str, int]:
    eqs, result = div_steps(amount, people)
    head = [f"まず、合計額を確認する。{amount}円である。", f"次に、人数を確認する。{people}人である。"]
    body = [f"よって、{people}で割る計算を上の位から順に行う。{eqs[0]}"] + eqs[1:]
    tail = [f"したがって、答えは{result}円である。"]
    return _join(head + body + tail), result


def cot_percent(base: int, pct: int) -> tuple[str, int]:
    eqs, prod = mul_steps(base, pct)
    div_eq, result = shift_div_100_steps(prod)
    head = [f"まず、基準額を確認する。{base}円である。", f"次に、割合を確認する。{pct}%である。"]
    body = [f"よって、{base}に{pct}をかける。{eqs[0]}"]
    body += eqs[1:]
    body += [f"最後に、100で割る。{div_eq[0]}"]
    tail = [f"したがって、答えは{result}円である。"]
    return _join(head + body + tail), result


# --------------------------------------------------------------------------- 5つの文型
def _pct_base(pct: int, lo: int, hi: int, rnd: random.Random) -> int:
    """base × pct が100の倍数になる基準額（割合の計算を割り切れる形に限定する）。"""
    m = 100 // gcd(pct, 100)
    lo_k, hi_k = (lo + m - 1) // m, hi // m
    return rnd.randint(lo_k, hi_k) * m


def make_problem(kind: str, rnd: random.Random, ranges: str = "default") -> dict:
    """kind: split / price / percent / speed / area。
    戻り値: {a, b, cot, answer_value, answer_text}（質問文は呼び出し側で言い換えを付ける）。"""
    r = RANGES[ranges][kind]
    if kind == "split":
        people = rnd.randint(*r[0])
        per = rnd.randint(*r[1])
        amount = people * per
        cot, res = cot_split(people, amount)
        a, b, unit = people, amount, "円"
    elif kind == "price":
        a, b = rnd.randint(*r[0]), rnd.randint(*r[1])
        cot, res = cot_multiply("単価", a, "円", "数量", b, "個", "合計額", "円")
        unit = "円"
    elif kind == "percent":
        pct = rnd.randint(*r[1])
        base = _pct_base(pct, r[0][0], r[0][1], rnd)
        cot, res = cot_percent(base, pct)
        a, b, unit = base, pct, "円"
    elif kind == "speed":
        a, b = rnd.randint(*r[0]), rnd.randint(*r[1])
        cot, res = cot_multiply("速度", a, "km", "時間", b, "時間", "距離", "km")
        cot = cot.replace(f"速度を確認する。{a}kmである", f"速度を確認する。時速{a}kmである")
        unit = "km"
    else:  # area
        a, b = rnd.randint(*r[0]), rnd.randint(*r[1])
        cot, res = cot_multiply("縦の長さ", a, "m", "横の長さ", b, "m", "面積", "m²")
        unit = "m²"
    return {"a": a, "b": b, "cot": cot, "value": res, "answer": f"{res}{unit}"}


# --------------------------------------------------------------------------- 計算量の測定
_EXPR_RE = re.compile(r"[\d.\s+\-*/×÷]+=\s*[-\d.]+")
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[+\-×÷*/]")


def _len0(n: int) -> int:
    """末尾のゼロを除いた桁数（200×3 の 200 は位取りで、実質1桁として扱う）。"""
    s = str(abs(int(n))).rstrip("0")
    return max(len(s), 1)


def _is_pow10(n: int) -> bool:
    return n >= 10 and set(str(n)[1:]) <= {"0"} and str(n)[0] == "1"


def op_load(a: Fraction, op: str, b: Fraction) -> int:
    """二項演算1回の計算量（桁レベルの基本操作の数の目安）。
       ×: 桁数の積（末尾ゼロは位取りとして除く）  1桁×1桁=1, 3桁×1桁=3, 4桁×2桁=8
       ÷: 同様（÷10^k は位取りで1）
       ＋/－: 小さい方の数の「0でない桁」の数（桁が重ならない足し算は1）。桁上がりの難しさは含めていない
    """
    ai, bi = int(a), int(b)
    if op in "×*":
        return _len0(ai) * _len0(bi)
    if op in "÷/":
        return 1 if _is_pow10(bi) else _len0(ai) * _len0(bi)
    # 足し算: 小さい方の数に含まれる「0でない桁」の数（=実際に桁ごとの足し算をする回数）。
    # 800 + 90 + 6 のように桁が重ならない足し算は、位を並べるだけなので1。
    nz = lambda n: max(sum(c != "0" for c in str(abs(n))), 1)
    return min(nz(ai), nz(bi))


def expression_load(lhs: str) -> int:
    """式（左辺）の計算量。複数の演算がある場合は、左から順に評価して各演算の量を足す
    （1ステップでまとめて暗算させる量、という意味で合算する）。"""
    toks = _TOKEN_RE.findall(lhs.replace(" ", ""))
    if not toks:
        return 0
    acc = Fraction(toks[0])
    load = 0
    i = 1
    while i + 1 < len(toks):
        op, nxt = toks[i], Fraction(toks[i + 1])
        load += op_load(acc, op, nxt)
        if op in "×*":
            acc *= nxt
        elif op in "÷/":
            acc = acc / nxt if nxt else acc
        elif op == "+":
            acc += nxt
        else:
            acc -= nxt
        i += 2
    return load


def cot_equation_loads(cot: str) -> list[int]:
    """CoT中の各式の計算量。"""
    loads = []
    for m in _EXPR_RE.findall(cot):
        lhs = m.split("=")[0]
        loads.append(expression_load(lhs))
    return loads


def verify_cot_math(cot: str) -> bool:
    """CoT中の全式が（余りを含めて）数学的に正しいか。割り算は 商×除数＋余り の形で厳密に検査する。"""
    for m in re.finditer(r"([\d\s+\-×÷]+)=\s*(\d+)(?:余り(\d+))?", cot):
        lhs, rhs, rem = m.group(1), int(m.group(2)), m.group(3)
        toks = _TOKEN_RE.findall(lhs.replace(" ", ""))
        if len(toks) == 3 and toks[1] == "÷":
            a, d = int(toks[0]), int(toks[2])
            if a != rhs * d + int(rem or 0):
                return False
            continue
        acc = Fraction(toks[0])
        for j in range(1, len(toks) - 1, 2):
            op, nxt = toks[j], Fraction(toks[j + 1])
            acc = acc * nxt if op == "×" else acc / nxt if op == "÷" else acc + nxt if op == "+" else acc - nxt
        if acc != rhs:
            return False
    return True


# --------------------------------------------------------------------------- 現行(oneshot)の再現（測定用）
def legacy_cot(kind: str, rnd: random.Random) -> tuple[str, str]:
    """generate_data.generate_arithmetic_data と同じ範囲・同じCoT文型（測定のため独立に再現）。"""
    if kind == "split":
        people, per = rnd.randint(2, 10), rnd.randint(120, 1000)
        amount = people * per
        return (f"まず、合計額を確認する。{amount}円である。次に、人数を確認する。{people}人である。"
                f"よって、割り勘額を計算する。{amount} ÷ {people} = {per}。したがって、答えは{per}円である。"), f"{per}円"
    if kind == "price":
        p, q = rnd.randint(100, 500), rnd.randint(2, 20)
        return (f"まず、単価を確認する。{p}円である。次に、数量を確認する。{q}個である。"
                f"よって、合計額を計算する。{p} × {q} = {p*q}。したがって、答えは{p*q}円である。"), f"{p*q}円"
    if kind == "percent":
        base, pct = rnd.randint(100, 10000), rnd.randint(5, 50)
        res = base * pct // 100
        return (f"まず、基準額を確認する。{base}円である。次に、割合を確認する。{pct}%である。"
                f"よって、割合金額を計算する。{base} × {pct} ÷ 100 = {res}。したがって、答えは{res}円である。"), f"{res}円"
    if kind == "speed":
        s, h = rnd.randint(40, 120), rnd.randint(1, 8)
        return (f"まず、速度を確認する。時速{s}kmである。次に、時間を確認する。{h}時間である。"
                f"よって、距離を計算する。{s} × {h} = {s*h}。したがって、答えは{s*h}kmである。"), f"{s*h}km"
    h, w = rnd.randint(2, 50), rnd.randint(2, 50)
    return (f"まず、縦の長さを確認する。{h}mである。次に、横の長さを確認する。{w}mである。"
            f"よって、面積を計算する。{h} × {w} = {h*w}。したがって、答えは{h*w}m²である。"), f"{h*w}m²"


KINDS = ["split", "price", "percent", "speed", "area"]


def audit(n_per_kind: int = 2000, seed: int = 0, hard_threshold: int = 6, ranges: str = "default") -> dict:
    """現行と手順つきの、1ステップ（1式）あたりの計算量を比べる。"""
    import statistics
    out = {}
    for style in ("oneshot", "stepwise"):
        rnd = random.Random(seed)
        per_kind = {}
        all_max, all_loads, n_steps, lengths = [], [], [], []
        for kind in KINDS:
            kmax, kloads, ksteps, klen = [], [], [], []
            for _ in range(n_per_kind):
                cot = legacy_cot(kind, rnd)[0] if style == "oneshot" else make_problem(kind, rnd, ranges)["cot"]
                loads = cot_equation_loads(cot)
                kmax.append(max(loads) if loads else 0)
                kloads += loads
                ksteps.append(len(loads))
                klen.append(len(cot))
            per_kind[kind] = {
                "式の数(平均)": round(statistics.mean(ksteps), 2),
                "最大の1式の計算量(平均)": round(statistics.mean(kmax), 2),
                "最大の1式の計算量(最大)": max(kmax),
                f"重い式(計算量>={hard_threshold})を含む問題の割合": round(sum(m >= hard_threshold for m in kmax) / len(kmax), 3),
                "CoTの文字数(平均/最大)": f"{statistics.mean(klen):.0f}/{max(klen)}",
            }
            all_max += kmax
            all_loads += kloads
            n_steps += ksteps
            lengths += klen
        out[style] = {
            "per_kind": per_kind,
            "overall": {
                "最大の1式の計算量(平均)": round(statistics.mean(all_max), 2),
                f"重い式(>= {hard_threshold})を含む問題の割合": round(sum(m >= hard_threshold for m in all_max) / len(all_max), 3),
                "式の数(平均)": round(statistics.mean(n_steps), 2),
                "CoTの文字数(平均/最大)": f"{statistics.mean(lengths):.0f}/{max(lengths)}",
            },
        }
    return out


def _print_audit(res: dict) -> None:
    names = {"oneshot": "現行(1式でまとめて計算)", "stepwise": "手順つき(位ごとに分解)"}
    for style, body in res.items():
        print(f"\n=== {names[style]} ===")
        for kind, d in body["per_kind"].items():
            print(f"  [{kind}] " + " / ".join(f"{k}={v}" for k, v in d.items()))
        print("  [全体] " + " / ".join(f"{k}={v}" for k, v in body["overall"].items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="算数の手順つきCoT: 例の表示と計算量の測定")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--n", type=int, default=2000, help="種別ごとのサンプル数")
    ap.add_argument("--hard_threshold", type=int, default=6)
    ap.add_argument("--ranges", choices=list(RANGES), default="default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.demo:
        rnd = random.Random(args.seed)
        for kind in KINDS:
            p = make_problem(kind, rnd, args.ranges)
            print(f"[{kind}] {p['a']}, {p['b']}\n  {p['cot']}\n  答え: {p['answer']}\n")
    if args.audit or not args.demo:
        _print_audit(audit(args.n, args.seed, args.hard_threshold, args.ranges))


if __name__ == "__main__":
    main()
