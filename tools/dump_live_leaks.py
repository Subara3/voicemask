"""リアルタイム版の漏れを、調整用の通話だけで並べる（評価用には使わない）。

    python tools/dump_live_leaks.py --rec runs/v1_stream_dev --delay-ms 1500
"""
import argparse
from pathlib import Path

from voicemask.asr import StreamLog
from voicemask.evaluate import COVER_OK, overlap, trim_to_voice
from voicemask.live import LiveMasker, emitted_intervals, feed
from voicemask.mask import CONDITIONS, Detectors
from voicemask.synth import load_truths

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="data/v1/truth.jsonl")
ap.add_argument("--rec", default="runs/v1_stream_dev")
ap.add_argument("--delay-ms", type=int, default=1500)
a = ap.parse_args()

cfg = CONDITIONS["C2"]
det = Detectors(cfg.base)
tr = {t.key: t for t in load_truths(a.truth, "clean")}
for p in sorted(Path(a.rec).glob("call*.json")):
    if int(p.name[4:8]) >= 100:
        raise SystemExit("調整用（call0000〜0099）以外は見ない")
    t = trim_to_voice(tr[p.stem], Path(a.truth).parent)
    log = StreamLog.load(p)
    m = feed(LiveMasker(cfg, det, log.engine, log.audio_path), log.events)
    eff = emitted_intervals(m.history, a.delay_ms, t.duration_ms)
    final = m.history[-1].intervals if m.history else []
    rec = log.to_recognition()
    for sp in t.spans:
        cov = overlap(eff, sp.start_ms, sp.end_ms) / max(1, sp.dur)
        if cov < COVER_OK:
            fin = overlap(final, sp.start_ms, sp.end_ms) / max(1, sp.dur)
            # いつ伏せる区間に入ったか
            first = next((h.at_ms for h in m.history if overlap(h.intervals, sp.start_ms, sp.end_ms) / max(1, sp.dur) >= COVER_OK), None)
            ctx = "".join(x.written for x in rec.tokens if x.end_ms > sp.start_ms - 2000 and x.start_ms < sp.end_ms + 800)
            parts = [e for e in log.events if e["type"] == "partial_raw" and sp.start_ms < e["arrived_ms"] < sp.end_ms + 3000]
            ptxt = (parts[len(parts) // 2]["data"].get("text", "") if parts else "")[-40:]
            print(f"{p.stem[:8]} {sp.pii_type:9} {sp.text[:16]:16} 出た{cov:.2f} 最後{fin:.2f} "
                  f"終わり{sp.end_ms} 判断{int(first) if first else '-'} | {ctx[:50]} | 途中:{ptxt}")
