"""文字の正規化と読みの生成。

認識結果の数字は「090」「〇九〇」「ゼロキュウゼロ」のように揺れるので、照合の前に算用数字へそろえる。
"""

from __future__ import annotations

import re
import unicodedata

DIGIT_READING = {
    "0": "ぜろ", "1": "いち", "2": "に", "3": "さん", "4": "よん",
    "5": "ご", "6": "ろく", "7": "なな", "8": "はち", "9": "きゅう",
}

KANJI_DIGITS = str.maketrans("〇一二三四五六七八九零", "01234567890")

# 長い語から順に照合する。「まる」は単独では数字とみなさない（kana_digits_to_arabic を参照）
_KANA_DIGITS = [
    ("きゅう", "9"), ("ぜろ", "0"), ("れい", "0"), ("まる", "0"), ("いち", "1"), ("さん", "3"),
    ("よん", "4"), ("ろく", "6"), ("なな", "7"), ("しち", "7"), ("はち", "8"), ("にい", "2"),
    ("ごお", "5"), ("に", "2"), ("ご", "5"), ("し", "4"), ("く", "9"),
]

_SEP = re.compile(r"[\s、。，．,.・\-‐－―—~〜()（）「」『』\[\]【】]+")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def digits_to_reading(s: str, sep: str = "の、") -> str:
    """'090-1234-5678' → 'ぜろきゅうぜろの、いちにさんよんの、ごろくななはち'"""
    groups = [g for g in re.split(r"[-‐－\s]+", nfkc(s)) if g]
    return sep.join("".join(DIGIT_READING.get(c, c) for c in g) for g in groups)


def kana_digits_to_arabic(s: str) -> str | None:
    """文字全体が数字の読みだけなら算用数字にする。違えば None。"""
    rest = kata_to_hira(nfkc(s)).replace("の", "").replace("ー", "").replace(" ", "")
    if rest in ("まる", ""):
        return None
    out = []
    while rest:
        for word, d in _KANA_DIGITS:
            if rest.startswith(word):
                out.append(d)
                rest = rest[len(word):]
                break
        else:
            return None
    return "".join(out)


def norm_token(written: str, spoken: str = "") -> str:
    """単語1つを照合用にそろえる。数字の読みなら算用数字にする。"""
    t = nfkc(written).translate(KANJI_DIGITS)
    if not t:
        return ""
    k = kana_digits_to_arabic(t)
    if k is None and spoken and not re.search(r"[一-龥]", t):
        k = kana_digits_to_arabic(spoken)
    return k if k is not None else t


def normalize_for_match(s: str) -> str:
    """値の照合用。全角半角・漢数字・区切り記号の違いを吸収する。"""
    t = nfkc(s).translate(KANJI_DIGITS)
    return _SEP.sub("", t).lower()


def digits_only(s: str) -> str:
    return "".join(c for c in nfkc(s).translate(KANJI_DIGITS) if c.isdigit())
