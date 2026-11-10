"""オフライン確認用の疑似認識。正解の部品から、時刻とconfidenceの付いた単語列を作る。

本物の認識の代わりにはならない（数値は記事に使えない）。処理を最後まで通す確認と、テストに使う。
"""

from __future__ import annotations

import random
import re

from .models import CallTruth, Recognition, Segment, Token
from .asr import split_by_gaps

_RUN = re.compile(r"[一-龥々]+|[ぁ-ん]+|[ァ-ヶー]+|[0-9０-９]+|[A-Za-z@._\-]+|[、。？！]|.")
_JUNK = ["さとう", "かとう", "そう", "いい", "とう", "こう", "加藤", "高尾", "西", "三重"]


def _words(text: str) -> list[str]:
    out = []
    for m in _RUN.finditer(text):
        w = m.group(0)
        if re.fullmatch(r"[ぁ-ん]+", w) and len(w) > 3:
            out.extend(w[i:i + 3] for i in range(0, len(w), 3))
        elif w.strip():
            out.append(w)
    return out


def fake_recognize(truth: CallTruth, seed: int = 0, err_pii: float = 0.15, err_other: float = 0.03,
                   jitter_ms: int = 60) -> Recognition:
    rng = random.Random(f"{truth.key}-{seed}")
    toks: list[Token] = []
    for p in truth.parts:
        ws = _words(p["text"])
        if not ws:
            continue
        dur = max(1, p["end_ms"] - p["start_ms"])
        total = sum(len(w) for w in ws)
        t = p["start_ms"]
        err = err_pii if p["pii_type"] else err_other
        for w in ws:
            d = dur * len(w) / total
            s = int(t + rng.uniform(-jitter_ms, jitter_ms))
            e = int(t + d + rng.uniform(-jitter_ms, jitter_ms))
            t += d
            if w in "、。？！":
                toks.append(Token(w, max(0, e), max(0, e), 1.0))
                continue
            if rng.random() < err:
                toks.append(Token(rng.choice(_JUNK), max(0, s), max(s + 1, e), rng.uniform(0.15, 0.55)))
            else:
                toks.append(Token(w, max(0, s), max(s + 1, e), min(1.0, rng.uniform(0.7, 1.0))))
    segs: list[Segment] = split_by_gaps(toks, 300)
    return Recognition(truth.audio_path, "fake", segs)
