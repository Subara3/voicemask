"""リアルタイムで伏せるときの評価。音声を何ミリ秒ためれば、伏せる判断が間に合うか。

音声を D ミリ秒遅らせて流すとき、時刻 t の音声が外に出るのは t+D。
伏せる判断は、その単語を含む確定結果（A イベント）が届いた時刻 a に決まる。
したがって区間のうち t+D >= a の部分だけを伏せられ、それより前の部分は伏せる前に流れてしまう。
"""

from __future__ import annotations

import random

from .asr import StreamLog, results_to_recognition
from .evaluate import COVER_OK, overlap
from .mask import MaskConfig, Detectors, compute_mask
from .models import CallTruth, Recognition


def segment_arrivals(log: StreamLog) -> list[tuple[int, int, float]]:
    """確定結果ごとに (発話の開始ms, 終了ms, 届いた時刻ms) を返す。"""
    out = []
    for arrived, data in log.finals():
        r = results_to_recognition(log.audio_path, log.engine, [(0, data)], gap_ms=None)
        out.extend((seg.start_ms, seg.end_ms, arrived) for seg in r.segments)
    return out


def simulate_arrivals(rec: Recognition, seed: int = 0, base_ms: int = 400, jitter_ms: int = 300) -> list[tuple[int, int, float]]:
    """オフライン確認用。発話の終わりから base±jitter ms 後に確定結果が届いたことにする。"""
    rng = random.Random(seed)
    return [(s.start_ms, s.end_ms, s.end_ms + base_ms + rng.uniform(-jitter_ms, jitter_ms)) for s in rec.segments]


def effective_intervals(res, arrivals: list[tuple[int, int, float]], delay_ms: int) -> list[tuple[int, int]]:
    out = []
    for iv in res.intervals:
        best, best_ov = None, -1
        for s, e, a in arrivals:
            ov = min(e, iv.end_ms) - max(s, iv.start_ms)
            if ov > best_ov:
                best, best_ov = a, ov
        decide = best if best is not None else float("inf")
        start = max(iv.start_ms, int(decide - delay_ms))
        if start < iv.end_ms:
            out.append((start, iv.end_ms))
    return out


def stream_leaks(items: list[tuple[CallTruth, Recognition, list]], cfg: MaskConfig, detectors: Detectors,
                 delays_ms: list[int]) -> list[dict]:
    rows = []
    results = [(t, compute_mask(r, cfg, detectors=detectors), a) for t, r, a in items]
    for d in delays_ms:
        pii = masked = spans = leaks = 0
        for truth, res, arr in results:
            eff = effective_intervals(res, arr, d)
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
    lags = sorted(a - e for _, _, arr in items for _, e, a in arr)
    if not lags:
        return {"n": 0}
    return {"n": len(lags), "p50": lags[len(lags) // 2], "p90": lags[int(len(lags) * 0.9) - 1 if len(lags) > 1 else 0], "max": lags[-1]}

