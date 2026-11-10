"""検出器が共通で使う、認識結果の見え方。

単語ごとに照合用の文字（norm）を作り、発話区間ごとの文字列と「文字の位置→単語の添字」の対応を持つ。
"""

from __future__ import annotations

from ..models import Recognition
from ..textnorm import norm_token


class TokenView:
    def __init__(self, rec: Recognition):
        self.rec = rec
        self.tokens = rec.tokens
        self.norm = [norm_token(t.written, t.spoken) for t in self.tokens]
        self.seg_bounds: list[tuple[int, int]] = []
        self.seg_of: list[int] = []
        i = 0
        for si, s in enumerate(rec.segments):
            self.seg_bounds.append((i, i + len(s.tokens)))
            self.seg_of.extend([si] * len(s.tokens))
            i += len(s.tokens)
        # 発話区間ごとの文字列と、文字→単語の対応
        self._text: list[str] = []
        self._char_tok: list[list[int]] = []
        for a, b in self.seg_bounds:
            chars, owner = [], []
            for k in range(a, b):
                chars.append(self.norm[k])
                owner.extend([k] * len(self.norm[k]))
            self._text.append("".join(chars))
            self._char_tok.append(owner)

    def segment_text(self, si: int) -> str:
        return self._text[si] if 0 <= si < len(self._text) else ""

    def char_to_token(self, si: int, pos: int) -> int:
        """発話区間 si の文字位置 pos を含む単語の添字。末尾を越えたら区間の終わり。"""
        owner = self._char_tok[si]
        return owner[pos] if pos < len(owner) else self.seg_bounds[si][1]

    def span_to_tokens(self, si: int, start: int, end: int) -> tuple[int, int]:
        owner = self._char_tok[si]
        if not owner or start >= len(owner):
            b = self.seg_bounds[si][1]
            return b, b
        return owner[start], owner[max(start, min(end, len(owner)) - 1)] + 1

    def find(self, si: int, s: str) -> int:
        """発話区間 si で s が最後に出る位置の直後（なければ -1）。"""
        t = self._text[si]
        i = t.rfind(s)
        return -1 if i < 0 else i + len(s)
