"""規則による検出：番号・生年月日・メールアドレス・住所の形と、敬称の前の名前。"""

from __future__ import annotations

import re

from ..models import Detection
from .tokenview import TokenView

# 数字の並び。間に区切り（ハイフン・「の」・読点・空白）を挟んでもよい
# E2E は数字の組のあいだに「。」を入れる（「5885。5765。」）ので、「。」も区切りに数える
_DIGITS = re.compile(r"\d(?:[\-‐－ーの、。,\s]{0,2}\d)*")
_BIRTH = re.compile(r"(?:(?:19|20)\d{2}|(?:昭和|平成|令和)(?:\d{1,2}|元))年\d{1,2}月\d{1,2}日")
# 年の読み「せんきゅうひゃくななじゅうろく」は「1907、16年」「190716年」と割れて認識されることがある（実測）。
# 年の桁数を問わず「数字…年 月 日」の形なら生年月日とみなす。日が化けたときは月までで拾い、発話の終わりまで伸ばす
_BIRTH_LOOSE = re.compile(r"\d(?:[\d、,\s]*\d)?年\d{1,2}月(?:\d{1,2}日)?")
BIRTH_CUES = ("生年月日", "お誕生日", "誕生日")
_END_WORDS = ("です", "でございます", "になります", "。", "？", "?")
_EMAIL = re.compile(r"[A-Za-z0-9._\-]+(?:@|アットマーク)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_EMAIL_HINT = re.compile(r"@|アットマーク|ドット|どっと")
_PREF = r"(?:北海道|東京都|京都府|大阪府|[一-龥]{2,3}県)"
_ADDR_PREF = re.compile(_PREF + r"[^。、？?\s]*?(?=です|でございます|になります|[。、？?]|$)")
_ADDR_CITY = re.compile(r"[一-龥ヶ]{1,6}[市区町村][^。、？?\s]{0,16}?\d+(?:丁目|番地?|号|-)(?:\d+(?:番地?|号)?|-)*")

POSTAL_CUES = ("郵便番号", "〒")
LAST4_CUES = ("下4桁", "下四桁", "下よんけた", "下よん桁")
HONORIFICS = ("様", "さま", "さん")
_NOT_NAME = {"お客", "皆", "奥", "お子", "ご本人", "本人", "担当", "ご契約者", "契約者", "会員", "お", "ご",
             "こちら", "そちら", "お客様", "皆様"}


PHONE_CUES = ("電話番号", "お電話", "携帯", "連絡先")
CARD_CUES = ("カード番号",)


def _num_type(digits: str, before: str, formatted: str) -> str:
    """数字の並びの種類。桁数で決まらないときは、直前の質問の手がかりで決める。

    誤認識で桁が欠けると（「029-6386」→「029-63は、知力」）、桁数の規則では拾えない。
    しかも誤認識した語の confidence が高いこともある（実測で0.98）。そこで質問の直後の数字は、桁が足りなくても伏せる。
    """
    n = len(digits)
    if n in (10, 11) and digits.startswith("0"):
        return "PHONE"
    if 14 <= n <= 16:
        return "CARD"
    if n == 7 and (any(c in before for c in POSTAL_CUES) or re.fullmatch(r"\d{3}\D{1,2}\d{4}", formatted)):
        return "POSTAL"
    if n == 4 and any(c in before for c in LAST4_CUES):
        return "CARD"
    if 3 <= n <= 8 and any(c in before for c in POSTAL_CUES):
        return "POSTAL"
    if 4 <= n <= 12 and any(c in before for c in PHONE_CUES):
        return "PHONE"
    if 4 <= n <= 16 and any(c in before for c in CARD_CUES):
        return "CARD"
    return ""


_FILLER = ("はい", "えっと", "えー", "あの", "、", "。")


def _email_bounds(view: TokenView, first_at: int, last_at: int) -> tuple[int, int]:
    """「@」を含む読み上げの範囲。

    1文字ずつ読むと間が空き、認識結果の区切り（発話）が途中で割れる（「YOECHI.」と「YAMADA46@example.ネット」が別の発話）。
    そこで発話の区切りは見ずに、単語の並びをさかのぼる。質問の終わり（「〜ます」「。」）か、2秒を超える無音で止める。
    後ろは「です」や句点まで（「.ネット」「.と公務」のようにドメインが化けても伏せる）。最大で「@」から4秒。
    """
    toks, norm = view.tokens, view.norm
    s = first_at
    while s > 0:
        p = s - 1
        # E2E は1文字ずつの読み上げの途中にも「。」を入れるので、句点では止めず、質問の終わり（「〜ます」「？」）で止める
        if norm[p] in ("？", "?") or re.search(r"(ます|ますか|でしょうか)$", norm[p] or "") or toks[s].start_ms - toks[p].end_ms > 2000:
            break
        if toks[first_at].start_ms - toks[p].start_ms > 15000:
            break
        s = p
    while s < first_at and norm[s] in _FILLER:
        s += 1
    e, limit = last_at + 1, toks[last_at].end_ms + 4000
    while e < len(toks) and norm[e] not in _END_WORDS and toks[e].start_ms < limit             and toks[e].start_ms - toks[e - 1].end_ms <= 2000:
        e += 1
    return s, e


def detect_rules(view: TokenView) -> list[Detection]:
    out: list[Detection] = []
    for si in range(len(view.seg_bounds)):
        text = view.segment_text(si)
        prev = view.segment_text(si - 1)

        def add(typ: str, m: re.Match, reason: str) -> None:
            s, e = view.span_to_tokens(si, m.start(), m.end())
            if e > s:
                out.append(Detection(typ, s, e, "rule", reason))

        for m in _BIRTH.finditer(text):
            add("BIRTHDATE", m, "生年月日の形")
        for m in _BIRTH_LOOSE.finditer(text):
            if _BIRTH.fullmatch(m.group(0)):
                continue
            s, e = view.span_to_tokens(si, m.start(), m.end())
            if e <= s:
                continue
            if not m.group(0).endswith("日"):
                if not any(c in prev[-20:] + text[:m.start()] for c in BIRTH_CUES):
                    continue
                b_ = view.seg_bounds[si][1]
                while e < b_ and view.norm[e] not in _END_WORDS:
                    e += 1
            out.append(Detection("BIRTHDATE", s, e, "rule", "生年月日の形（年が割れた）"))
        for m in _DIGITS.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            typ = _num_type(digits, prev[-20:] + text[max(0, m.start() - 12):m.start()], m.group(0))
            if typ:
                add(typ, m, f"{len(digits)}桁の数字")
        for rx in (_ADDR_PREF, _ADDR_CITY):
            for m in rx.finditer(text):
                add("ADDRESS", m, "住所の形")
        for m in _EMAIL.finditer(text):
            add("EMAIL", m, "メールアドレスの形")

        a, b = view.seg_bounds[si]
        # 1文字ずつ読み上げたメールアドレスは、単語が細かく割れ、一部がかなに化ける（「SOTARO_はいえMaDa@example.com」）。
        # 「@」のある発話では、発話の頭（「はい」「えっと」を除く）から「@」の後のドメインまでを伏せる
        hit = [i for i in range(a, b) if re.search(r"@|アットマーク", view.norm[i])]
        if hit:
            s, e = _email_bounds(view, hit[0], hit[-1])
            out.append(Detection("EMAIL", s, e, "rule", "アットマークのある発話"))

        # 「中島様」：敬称の直前1〜2語を名前とみなす
        for i in range(a + 1, b):
            if view.norm[i] in HONORIFICS:
                j, name = i - 1, []
                while j >= a and len(name) < 2:
                    w = view.norm[j]
                    if not w or w in _NOT_NAME or not re.fullmatch(r"[一-龥々ァ-ヶーぁ-ん]+", w):
                        break
                    name.append(j)
                    j -= 1
                if name:
                    out.append(Detection("NAME", min(name), i, "rule", "敬称の前"))
    return out


_ADDR_TAIL = re.compile(r"^(?:\d+|\d+(?:丁目|番地?|号)|丁目|番地?|号|[-‐－の]|[一-龥]{1,4}(?:町|丁目|村|字))$")


def extend_address_tail(view: TokenView, dets: list[Detection]) -> list[Detection]:
    """住所の後ろに続く「6丁目22番20号」を住所に含める。固有表現抽出は番地を取りこぼしやすいため。"""
    out = []
    for d in dets:
        if d.pii_type != "ADDRESS" or d.token_start >= len(view.seg_of):
            continue
        _, b = view.seg_bounds[view.seg_of[d.token_start]]
        e = d.token_end
        while e < len(view.norm):
            # 「15番。14号」：句点や読点をはさんでも番地が続くなら含める（E2E で実測）
            k = e
            while k < len(view.norm) and view.norm[k] in ("。", "、"):
                k += 1
            if k < len(view.norm) and view.norm[k] and _ADDR_TAIL.match(view.norm[k]) and (k == e or k < b + 3):
                e = k + 1
            else:
                break
        if e > d.token_end:
            out.append(Detection("ADDRESS", d.token_start, e, "rule", "住所の番地"))
    return out
