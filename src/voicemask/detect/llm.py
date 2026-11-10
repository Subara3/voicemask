"""Claude による検出（比較用）。

答えは data/llm/answer/<key>.json に残す。答えがなければ Claude API に聞く（上限は llm_client の予算）。
`voicemask llm-export` で問いだけ書き出し、Claude Code に答えてもらうこともできる。

伏せる前の文字を Claude に見せるので、「LLMに渡す前に伏せる」という目的とは逆向きになる。本線には入れない。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..models import PII_TYPES, Detection
from .tokenview import TokenView

PROMPT = """次は通話の音声認識結果です。各単語の前に [番号] を付けています。
氏名・電話番号・住所・郵便番号・生年月日・カード番号・メールアドレスに当たる単語の範囲を、すべて答えてください。
認識の誤りで別の言葉に化けていても、文脈から個人情報だと分かるものは含めてください。
注文番号・金額・予約日などは個人情報ではありません。
答えは JSON の配列だけ。各要素は {"type": 種類, "start": 最初の番号, "end": 最後の番号}。
種類は NAME, PHONE, ADDRESS, POSTAL, BIRTHDATE, CARD, EMAIL のどれか。
"""


MASKED_NOTE = "［伏］は手元ですでに伏せた単語です。［伏］以外で、まだ伏せていない個人情報だけを答えてください。\n"


def numbered_text(view: TokenView, masked: list[bool] | None = None) -> str:
    """masked を渡すと、伏せた単語を「［伏］」に置き換える。Claude には手元で取りこぼした分しか届かない。"""
    def w(i: int) -> str:
        return "［伏］" if masked and masked[i] else view.tokens[i].written
    return "\n".join(" ".join(f"[{i}]{w(i)}" for i in range(a, b)) for a, b in view.seg_bounds)


def ask_key(view: TokenView, masked: list[bool] | None = None) -> str:
    return hashlib.sha256(numbered_text(view, masked).encode()).hexdigest()[:24]


def export_question(view: TokenView, root: str | Path = "data/llm") -> Path:
    p = Path(root) / "ask" / f"{ask_key(view)}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(PROMPT + "\n" + numbered_text(view), encoding="utf-8")
    return p


class LLMDetector:
    def __init__(self, root: str | Path = "data/llm", use_api: bool = True):
        self.root = Path(root)
        self.use_api = use_api
        self.missing = 0

    def detect(self, view: TokenView, masked: list[bool] | None = None) -> list[Detection]:
        p = self.root / "answer" / f"{ask_key(view, masked)}.json"
        if not p.exists() and self.use_api:
            from ..llm_client import ask

            text = ask(PROMPT + (MASKED_NOTE if masked else "") + "\n" + numbered_text(view, masked), max_tokens=3000)
            try:
                body = text[text.index("["): text.rindex("]") + 1]
            except ValueError:
                body = "[]"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        if not p.exists():
            self.missing += 1
            return []
        try:
            items = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            return []
        out, n = [], len(view.tokens)
        for it in items:
            try:
                s, e, t = int(it["start"]), int(it["end"]) + 1, str(it["type"])
            except (KeyError, TypeError, ValueError):
                continue
            if t in PII_TYPES and 0 <= s < e <= n:
                out.append(Detection(t, s, e, "llm", "Claude"))
        return out
