"""文脈による検出。誤認識で別の言葉に化けた個人情報を、confidence の低さで拾う。

個人情報は、名前や番号を尋ねた直後に来る。そこで誤認識が起きると文字は別の言葉に化け、
規則でも固有表現抽出でも見つからない。ただし confidence は下がる。

- detect_lowconf：質問の手がかりの後 window_ms 以内で、confidence が閾値未満の内容語を伏せる（条件C）
- detect_answers：質問への答えの文を、confidence に関係なく丸ごと伏せる（条件C2）
- expand_neighbors：検出した範囲の隣で confidence が閾値未満の単語へ範囲を広げる
"""

from __future__ import annotations

import re

from ..models import Detection
from .tokenview import TokenView

CUES: dict[str, tuple[str, ...]] = {
    "NAME": ("お名前", "名前", "氏名", "フルネーム", "名義"),
    "PHONE": ("お電話番号", "電話番号", "連絡先", "携帯"),
    "ADDRESS": ("ご住所", "住所", "お届け先"),
    "POSTAL": ("郵便番号",),
    "BIRTHDATE": ("生年月日", "お誕生日", "誕生日"),
    "CARD": ("カード番号", "下4桁", "下四桁"),
    "EMAIL": ("メールアドレス",),
}

FUNCTION_WORDS = {
    "です", "でした", "ます", "ました", "は", "が", "を", "に", "の", "で", "と", "も", "はい", "ええ", "えっと",
    "あの", "その", "まあ", "ね", "よ", "か", "から", "まで", "お願い", "します", "いたし", "ございます",
    "ありがとう", "すみません", "えー", "あー", "申し", "ます", "と申します", "なります",
}

# 手がかりの後が「お願いします」「伺えますか」なら、答えは次の発話にある
_REQUEST = re.compile(r"^[はをもの]?.{0,14}?(お願い|教えて|ください|伺|でしょうか|ますか|[？?])")


def is_content(w: str) -> bool:
    return bool(w) and w not in FUNCTION_WORDS and not all(c in "、。？！?! " for c in w)


def _cue_hits(view: TokenView):
    """(種類, 発話区間, 手がかりの直後の単語) を返す。"""
    for si in range(len(view.seg_bounds)):
        for typ, cues in CUES.items():
            pos = max((view.find(si, c) for c in cues), default=-1)
            if pos >= 0:
                yield typ, si, pos


NUMERIC = {"PHONE", "POSTAL", "CARD", "BIRTHDATE"}
_NUM_PIECE = re.compile(r"^[\d\-‐－ーの、,年月日]+$")


def _numeric_run(view: TokenView, i: int, threshold: float) -> tuple[int, int]:
    """番号の並びの一部が自信なしなら、並び全体（数字・区切り・自信のない語）を伏せる。

    番号は一部でも残ると意味を持つうえ、どこまでが番号かは前後の数字を見ないと決まらないため。
    """
    a, b = view.seg_bounds[view.seg_of[i]]
    ok = lambda k: bool(_NUM_PIECE.match(view.norm[k] or "")) or (
        view.tokens[k].confidence < threshold and is_content(view.norm[k]))
    s, e = i, i + 1
    while s > a and ok(s - 1):
        s -= 1
    while e < b and ok(e):
        e += 1
    return s, e


def detect_lowconf(view: TokenView, threshold: float, window_ms: int = 8000) -> list[Detection]:
    toks, out = view.tokens, []
    for typ, si, pos in _cue_hits(view):
        i = view.char_to_token(si, pos)
        a = view.seg_bounds[si][0]
        anchor = toks[i - 1].end_ms if i > a else toks[a].start_ms
        while i < len(toks) and toks[i].start_ms <= anchor + window_ms:
            if toks[i].confidence < threshold and is_content(view.norm[i]):
                s, e = _numeric_run(view, i, threshold) if typ in NUMERIC else (i, i + 1)
                out.append(Detection(typ, s, e, "context", f"{CUES[typ][0]}の後で confidence {toks[i].confidence:.2f}"))
            i += 1
    return out


def detect_answers(view: TokenView, window_ms: int = 8000) -> list[Detection]:
    toks, out = view.tokens, []
    for typ, si, pos in _cue_hits(view):
        rest = view.segment_text(si)[pos:]
        if not rest or _REQUEST.match(rest):
            if si + 1 >= len(view.seg_bounds):
                continue
            i = view.seg_bounds[si + 1][0]       # 尋ねている → 次の発話が答え
        elif rest[0] in "はが":
            i = view.char_to_token(si, pos)       # 「電話番号は〜です」→ 同じ文の続きが答え
        else:
            continue
        limit = toks[i].start_ms + min(window_ms, _WINDOW.get(typ, 5000)) if i < len(toks) else 0
        got = False
        while i < len(toks) and toks[i].start_ms <= limit:
            w = view.norm[i]
            cue = _cue_at(view, i)
            if cue:
                if cue[0] != typ and got:
                    break
                i = cue[1]  # 「下」「4」「桁」のように割れた手がかりの言葉は伏せない
                continue
            if w in ("。", "？", "?"):
                # 番号は「。」をはさんで続く（E2E）。名前も「田中。ナオトです」と割れるので、すぐ「です」が来るなら続ける
                if got and typ not in _NUMERIC and not _end_soon(view, i + 1):
                    break
            elif _answer_ends(w, typ):
                break
            elif w in _SKIP or any(c in w for c in CUES[typ]):
                pass  # 「下4桁は」と言い直す、「佐藤様」の「様」
            elif is_content(w):
                out.append(Detection(typ, i, i + 1, "answer", f"{CUES[typ][0]}への答え"))
                got = True
            i += 1
    return out


# 答えの終わり。ここで止めないと、続けて担当者が尋ねた次の質問（「お客様のお名前を」）まで伏せてしまう（実測で消しすぎの大半）
_END = ("です", "と申します", "でございます", "になります", "ですね")
_STOP_IN = ("お願い", "伺", "でしょうか", "ください", "確認", "ありがとう", "申し", "ございます", "かしこまり", "承知",
            "お客様", "契約者", "新しい", "支払い", "ご本人", "ご登録", "ご予約", "ご注文")
_STOP_EQ = ("お", "ご")
# 答えを探す長さ。メールアドレスは1文字ずつ読むので長い
_WINDOW = {"EMAIL": 15000, "ADDRESS": 8000}


def _cue_at(view: TokenView, i: int) -> tuple[str, int] | None:
    """単語 i から手がかりの言葉が始まるなら (種類, 手がかりの次の単語)。"""
    s = ""
    for j in range(i, min(i + 4, len(view.norm))):
        s += view.norm[j] or ""
        for typ, cues in CUES.items():
            if s in cues:
                return typ, j + 1
        if not any(c.startswith(s) for cs in CUES.values() for c in cs):
            return None
    return None
_SKIP = ("様", "さま", "さん", "はい", "ええ", "えっと", "えー", "あの")


_NUMERIC = ("PHONE", "POSTAL", "CARD", "BIRTHDATE")


def _end_soon(view: TokenView, j: int, within: int = 3) -> bool:
    for k in range(j, min(j + within + 1, len(view.norm))):
        if view.norm[k] in _END:
            return True
        if view.norm[k] in ("。", "？", "?"):
            return False
    return False


def _answer_ends(w: str, typ: str) -> bool:
    if w in _END or w in _STOP_EQ or any(s in w for s in _STOP_IN):
        return True
    return any(c in w for k, cs in CUES.items() if k != typ for c in cs)


def expand_neighbors(view: TokenView, dets: list[Detection], threshold: float) -> list[Detection]:
    toks, out = view.tokens, []
    for d in dets:
        if d.token_end <= d.token_start or d.token_start >= len(view.seg_of):
            continue
        a, b = view.seg_bounds[view.seg_of[d.token_start]]
        s, e = d.token_start, d.token_end
        while s > a and toks[s - 1].confidence < threshold and is_content(view.norm[s - 1]):
            s -= 1
        while e < b and toks[e].confidence < threshold and is_content(view.norm[e]):
            e += 1
        if (s, e) != (d.token_start, d.token_end):
            out.append(Detection(d.pii_type, s, e, "neighbor", f"{d.source}の隣で confidence が低い"))
    return out
