"""手元で伏せたあとの文字を Claude に送り、モデルごとの漏れ率を比べる。

    VOICEMASK_LLM_MODEL=claude-opus-5-5 python tools/llm_compare.py --limit 200 --out results/x.md

答えは data/llm/answer_<モデル>/ に残すので、2回目からは API を呼ばない。上限は llm_client の予算（既定10ドル）。
"""

import argparse
import os
import sys
from pathlib import Path

from voicemask.cli import _pairs
from voicemask.evaluate import Counts, score_call
from voicemask.llm_client import DEFAULT_MODEL, spent_usd
from voicemask.mask import CONDITIONS, Detectors, mask_then_llm

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="data/v1/truth.jsonl")
ap.add_argument("--rec", default="runs/v1_hybrid")
ap.add_argument("--start", default="call0100")
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--out", default=None)
a = ap.parse_args()

model = os.environ.get("VOICEMASK_LLM_MODEL", DEFAULT_MODEL)
pairs = [(t, r) for t, r in _pairs(a.truth, a.rec, "clean") if t.call_id >= a.start][: a.limit]
cfg = CONDITIONS["C2"]
det = Detectors(cfg.base)
before = spent_usd()
first_c, after_c, sent = Counts(), Counts(), 0
for i, (t, rec) in enumerate(pairs):
    first, after = mask_then_llm(rec, cfg, det)
    c1 = score_call(t, rec, first)
    first_c.add(c1)
    after_c.add(score_call(t, rec, after))
    sent += c1.text_leaks  # 文字が伏せられずに残った個人情報＝Claude に届いたもの
    print(f"\r{i + 1}/{len(pairs)} {spent_usd() - before:.3f}ドル", end="", file=sys.stderr)
print(file=sys.stderr)


def row(name, m):
    return f"| {name} | {m['漏れ率（時間）']:.1%} | {m['区間漏れ率']:.1%} | {m['文字漏れ率']:.1%} | {m['消しすぎ率']:.1%} | {m['適合率']:.1%} | {m['件数']} |"


md = [f"# 手元で伏せてから {model} に送る（{len(pairs)}本、{a.start}から、ハイブリッド・雑音なし）", "",
      "| 条件 | 漏れ率（時間） | 区間漏れ率 | 文字漏れ率 | 消しすぎ率 | 適合率 | 件数 |", "|---|---|---|---|---|---|---|",
      row("C2（手元だけ）", first_c.metrics()), row(f"C2のあとに伏せた文字を{model}へ", after_c.metrics()), "",
      f"{model} に届いた個人情報：{sent}件／{first_c.spans}件。今回の費用 {spent_usd() - before:.3f}ドル（累計 {spent_usd():.3f}ドル）"]
print("\n".join(md))
if a.out:
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8")
