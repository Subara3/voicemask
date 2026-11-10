"""リアルタイムで伏せるときの評価。音声を何ミリ秒ためれば、伏せる判断が間に合うか。

音声を D ミリ秒遅らせて流すとき、時刻 t の音声が外に出るのは t+D。
伏せる判断は、その単語を含む確定結果（A イベント）が届いた時刻 a に決まる。
したがって区間のうち t+D >= a の部分だけを伏せられ、それより前の部分は伏せる前に流れてしまう。
"""

from __future__ import annotations

import random

from .asr import StreamLog, results_to_recognition
from .evaluate import COVER_OK, overlap
from .mask import MaskConfig, Detectors, compute_mask, default_shift
from .models import CallTruth, Recognition


def segment_arrivals(log: StreamLog) -> list[tuple[int, int, float]]:
    """確定結果ごとに (発話の開始ms, 終了ms, 届いた時刻ms) を返す。"""
    out = []
    for arrived, data in log.finals():
        r = results_to_recognition(log.audio_path, log.engine, [(0, data)], gap_ms=None)
        out.extend((seg.start_ms, seg.end_ms, arrived) for seg in r.segments)
    return out


def partial_texts(log: StreamLog) -> list[tuple[float, list[tuple[float, str]]]]:
    """確定結果ごとに (届いた時刻, その発話の途中結果 [(届いた時刻, 文字)]) を返す。

    途中結果（U）の単語には時刻がないので、文字だけを残す。「...」はまだ決まっていない部分の印なので消す。
    """
    out, buf = [], []
    for e in log.events:
        if e["type"] == "partial_raw":
            buf.append((e["arrived_ms"], (e["data"].get("text") or "").replace(".", "")))
        elif e["type"] == "final" and e.get("data"):
            out.append((e["arrived_ms"], buf))
            buf = []
    return out


def partial_arrivals(log: StreamLog, rec: Recognition) -> list[tuple[int, int, float, list]]:
    """segment_arrivals に、その発話の途中結果を添える。"""
    finals = partial_texts(log)
    segs = segment_arrivals(log)
    out, k = [], 0
    for arrived, parts in finals:
        while k < len(segs) and segs[k][2] == arrived:
            s, e, a = segs[k]
            out.append((s, e, a, parts))
            k += 1
    return out


def simulate_arrivals(rec: Recognition, seed: int = 0, base_ms: int = 400, jitter_ms: int = 300) -> list[tuple[int, int, float]]:
    """オフライン確認用。発話の終わりから base±jitter ms 後に確定結果が届いたことにする。"""
    rng = random.Random(seed)
    return [(s.start_ms, s.end_ms, s.end_ms + base_ms + rng.uniform(-jitter_ms, jitter_ms)) for s in rec.segments]


def effective_intervals(res, arrivals: list[tuple[int, int, float]], delay_ms: int, shift_ms: int = 0) -> list[tuple[int, int]]:
    """D ミリ秒ためたときに、実際に伏せられる区間（重なりはまとめる）。

    区間が複数の発話にまたがるときは、いちばん遅く届いた確定結果を待たないと伏せられない。
    """
    from .audio import merge_intervals

    out = []
    for iv in res.intervals:
        # 区間は shift_ms だけ前にずらしてあるので、発話の時刻と比べるときは戻す
        s0, e0 = iv.start_ms + shift_ms, iv.end_ms + shift_ms
        hits = [a for s, e, a in arrivals if min(e, e0) > max(s, s0)]
        decide = max(hits) if hits else float("inf")
        start = max(iv.start_ms, int(decide - delay_ms)) if decide != float("inf") else iv.end_ms
        if start < iv.end_ms:
            out.append((start, iv.end_ms))
    return merge_intervals(out, 0)


def reveal_time(text: str, parts: list[tuple[float, str]], final_at: float) -> float:
    """区間の文字が途中結果に初めて出た時刻。出なければ確定結果の時刻。"""
    from .textnorm import normalize_for_match

    want = normalize_for_match(text)  # 途中結果は「Y O K O」のように空白が入ったり、区切り記号が違ったりする
    for t, p in parts:
        if want and want in normalize_for_match(p):
            return min(t, final_at)
    return final_at


def partial_intervals(res, rec: Recognition, arrivals: list, delay_ms: int, shift_ms: int = 0) -> list[tuple[int, int]]:
    """途中結果で判断する場合の、実際に伏せられる区間。

    伏せる範囲は確定結果の時刻で決める。途中結果に時刻はないので、判断が付いた時点で
    「その語はもう話された」ことしか分からない。ここでは、判断が付いた時刻から、ためた分だけ遡って伏せられるとみなす。
    """
    from .audio import merge_intervals

    toks = rec.tokens
    out = []
    for iv in res.intervals:
        s0, e0 = iv.start_ms + shift_ms, iv.end_ms + shift_ms
        text = "".join(t.written for t in toks if not t.is_punct and t.start_ms >= s0 - 1 and t.end_ms <= e0 + 1)
        hits = [reveal_time(text, parts, a) for s, e, a, parts in arrivals if min(e, e0) > max(s, s0)]
        if not hits:
            continue
        decide = max(hits)
        start = max(iv.start_ms, int(decide - delay_ms))
        if start < iv.end_ms:
            out.append((start, iv.end_ms))
    return merge_intervals(out, 0)


def stream_leaks(items: list[tuple[CallTruth, Recognition, list]], cfg: MaskConfig, detectors: Detectors,
                 delays_ms: list[int], use_partial: bool = False) -> list[dict]:
    rows = []
    results = [(t, r, compute_mask(r, cfg, detectors=detectors), a,
                default_shift(r.engine) if cfg.shift_ms is None else cfg.shift_ms) for t, r, a in items]
    for d in delays_ms:
        pii = masked = spans = leaks = 0
        for truth, rec, res, arr, shift in results:
            eff = partial_intervals(res, rec, arr, d, shift) if use_partial else effective_intervals(res, arr, d, shift)
            for sp in truth.spans:
                m = overlap(eff, sp.start_ms, sp.end_ms)
                pii += sp.dur
                masked += m
                spans += 1
                leaks += 1 if sp.dur and m / sp.dur < COVER_OK else 0
        rows.append({"ためる時間ms": d, "漏れ率（時間）": 1 - masked / pii if pii else 0.0,
                     "区間漏れ率": leaks / spans if spans else 0.0, "件数": spans})
    return rows


def decision_lags(items) -> dict:
    """発話の終わりから確定結果が届くまでの時間（ms）の分布。"""
    lags = sorted(x[2] - x[1] for _, _, arr in items for x in arr)
    if not lags:
        return {"n": 0}
    return {"n": len(lags), "p50": lags[len(lags) // 2], "p90": lags[int(len(lags) * 0.9) - 1 if len(lags) > 1 else 0], "max": lags[-1]}

