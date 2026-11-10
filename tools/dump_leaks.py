"""調整用の通話（既定 0〜99番）で漏れた区間を、前後の認識結果と一緒に並べる。評価用の通話は見ない。"""
import argparse
import sys
from pathlib import Path

from voicemask.asr import load_recognition
from voicemask.evaluate import COVER_OK, overlap
from voicemask.mask import CONDITIONS, Detectors, compute_mask
from voicemask.synth import load_truths
from voicemask.evaluate import trim_to_voice

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="data/v1/truth.jsonl")
ap.add_argument("--rec", default="runs/v1_hybrid")
ap.add_argument("--variant", default="clean")
ap.add_argument("--condition", default="C")
ap.add_argument("--types", default="EMAIL,BIRTHDATE")
ap.add_argument("--dev-max", type=int, default=99)
ap.add_argument("--pad-before", type=int, default=None)
a = ap.parse_args()

types = set(a.types.split(","))
from dataclasses import replace
cfg = replace(CONDITIONS[a.condition], pad_before_ms=a.pad_before)
det = Detectors(cfg.base)
for t in [trim_to_voice(x, Path(a.truth).parent) for x in load_truths(a.truth, a.variant)]:
    if int(t.call_id.replace("call", "")) > a.dev_max:
        continue
    p = Path(a.rec) / f"{t.key}.json"
    if not p.exists():
        continue
    rec = load_recognition(p)
    res = compute_mask(rec, cfg, detectors=det)
    for sp in t.spans:
        if sp.pii_type not in types:
            continue
        cov = overlap(res.merged, sp.start_ms, sp.end_ms) / max(1, sp.dur)
        if cov >= COVER_OK:
            continue
        near = [(tk, m) for tk, m in zip(rec.tokens, res.masked) if tk.end_ms > sp.start_ms - 2500 and tk.start_ms < sp.end_ms + 500]
        line = "".join((f"[{tk.written}]" if m else tk.written) + ("" if tk.confidence >= .6 else "?") for tk, m in near)
        print(f"{t.call_id} {sp.pii_type} 覆い{cov:.0%} 正解「{sp.text}」\n    {line}")
