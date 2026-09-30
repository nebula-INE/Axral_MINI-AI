"""
search_agent.py
未知トピックの質問をWeb検索（Wikipedia）で補うモジュール。

既知トピックガード（knowledge_base.is_known_topic）が「学習していない話題」と判定した
質問に対して、「分かりません」で終わらせず、Wikipediaの要約を取得して答える。

設計方針:
  - 標準ライブラリ(urllib)のみ使用。APIキー不要・追加インストール不要。
  - Kaggleはデフォルトでインターネットが無効。Notebook右側の Settings → Internet を
    ONにしない限り接続に失敗する。失敗時は例外を投げず None を返し、
    呼び出し側が従来の「分かりません」にフォールバックできるようにする。
  - モデル自身の生成ではなく、検索で取得した文章をそのまま（出典URL付きで）返す。
    300万〜数百万パラメータのモデルに「検索結果を読んで要約させる」のは、
    検索するかどうかの判断と同様に新しい学習課題になってしまうため。
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

_WIKI_HOST = "https://ja.wikipedia.org"
_USER_AGENT = "Axral-MINI-AI/0.1 (research project; contact: local)"

# 検索クエリから取り除く、質問の言い回し部分。
_QUERY_NOISE = ("について教えて", "を教えて", "とは何ですか", "とは何", "でしょうか", "ですか", "とは", "は")


def _clean_query(text: str) -> str:
    q = text.strip().rstrip("？?").strip()  # 先に疑問符を除去（「とは？」が「は？」に誤マッチするのを防ぐ）
    for noise in _QUERY_NOISE:
        if q.endswith(noise):
            q = q[: -len(noise)]
            break
    return q.strip() or text.strip()


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _truncate_sentences(text: str, max_chars: int = 200) -> str:
    """句点区切りで max_chars を超えない範囲までの文章を返す（最低1文は残す）。"""
    sentences = [s for s in re.split(r"(?<=。)", text) if s.strip()]
    if not sentences:
        return text[:max_chars]
    out = sentences[0]
    for s in sentences[1:]:
        if len(out) + len(s) > max_chars:
            break
        out += s
    return out.strip()


def search_wikipedia(query: str, timeout: float = 8.0) -> dict | None:
    """日本語Wikipediaで検索し、最上位記事の {title, extract, url} を返す。

    見つからない・ネットワーク不通・API形式が想定外の場合は None を返す（例外は投げない）。
    """
    q = _clean_query(query)
    try:
        search_url = (
            f"{_WIKI_HOST}/w/api.php?action=query&list=search&format=json"
            f"&srlimit=1&srsearch={urllib.parse.quote(q)}"
        )
        hits = _get_json(search_url, timeout).get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]

        summary_url = f"{_WIKI_HOST}/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        data = _get_json(summary_url, timeout)
        extract = data.get("extract", "").strip()
        if not extract:
            return None
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page") \
            or f"{_WIKI_HOST}/wiki/{urllib.parse.quote(title)}"
        return {"title": title, "extract": extract, "url": page_url}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def search_and_answer(query: str, timeout: float = 8.0, max_chars: int = 200) -> str | None:
    """検索して回答文字列を作る。取得できなければ None（呼び出し側でフォールバック）。"""
    result = search_wikipedia(query, timeout=timeout)
    if result is None:
        return None
    body = _truncate_sentences(result["extract"], max_chars=max_chars)
    return f"【Web検索: {result['title']}】{body}（出典: {result['url']}）"
