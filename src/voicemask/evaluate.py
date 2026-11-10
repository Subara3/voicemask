"""評価：正解の区間と伏せた区間を比べて、漏れ率・消しすぎ率などを出す。

指標
- 漏れ率（時間）    正解の個人情報の音声のうち、伏せ損ねた長さの割合
- 区間漏れ率        個人情報1件ごとに、90%未満しか伏せられなかった件数の割合
- 文字漏れ率        正解の区間に半分以上かかる単語のうち、伏せていない内容語が残った件数の割合
- 消しすぎ率        個人情報でない発話のうち、伏せた長さの割合
- 適合率            伏せた長さのうち、本当に個人情報だった割合
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .detect.context import is_content
from .detect.tokenview import TokenView
from .mask import CONDITIONS, Detectors, MaskConfig, compute_mask, with_threshold
from .models import CallTruth, PII_LABELS_JA, Recognition

COVER_OK = 0.9


def overlap(a: list[tuple[int, int]], s: int, e: int) -> int:
    return sum(max(0, min(e, y) - max(s, x)) for x, y in a)


@dataclass
class Counts:
    pii_ms: int = 0
    pii_masked_ms: int = 0
    other_ms: int = 0
    other_masked_ms: int = 0
    masked_ms: int = 0
    masked_true_ms: int = 0
    spans: int = 0
    span_leaks: int = 0
    text_leaks: int = 0

    def add(self, o: "Counts") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))

    def metrics(self) -> dict:
        r = lambda a, b: a / b if b else 0.0
        return {
            "漏れ率（時間）": 1 - r(self.pii_masked_ms, self.pii_ms) if self.pii_ms else 0.0,
            "区間漏れ率": r(self.span_leaks, self.spans),
            "文字漏れ率": r(self.text_leaks, self.spans),
            "消しすぎ率": r(self.other_masked_ms, self.other_ms),
            "適合率": r(self.masked_true_ms, self.masked_ms),
            "件数": self.spans,
        }


def trim_to_voice(truth: CallTruth, base_dir, cache: dict | None = None) -> CallTruth:
    """正解の区間を、声の出ている範囲に詰める。

    台本は部品ごとに合成してつないでいるので、部品の前後に合成音声の無音（実測0.4〜0.8秒）が付く。
    そのまま採点すると、無音を伏せていないことまで「漏れ」に数えてしまう。
    どの版も同じ時間軸なので、雑音のない版の音声で声のある範囲を求める。
    """
    import numpy as np
    from pathlib import Path
    from dataclasses import replace
    from . import audio

    key = truth.call_id
    if cache is not None and key in cache:
        env, sr_ms, thr = cache[key]
    else:
        x, sr = audio.read_wav(Path(base_dir) / "audio" / f"{truth.call_id}__clean.wav")
        hop = sr // 100  # 10ms
        n = len(x) // hop
        env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
        thr = float(np.max(env)) * 10 ** (-30 / 20)  # 合成音声の頭に付く息の音（-40〜-50dB）は声に数えない
        sr_ms = 10
        if cache is not None:
            cache[key] = (env, sr_ms, thr)

    def trim(s: int, e: int) -> tuple[int, int]:
        a, b = s // sr_ms, min(len(env), -(-e // sr_ms))
        on = np.nonzero(env[a:b] > thr)[0]
        if not len(on):
            return s, e
        return max(s, (a + on[0]) * sr_ms - 50), min(e, (a + on[-1] + 1) * sr_ms + 50)  # 弱い子音のぶん50ms残す

    parts = []
    for p in truth.parts:
        s, e = trim(p["start_ms"], p["end_ms"])
        parts.append({**p, "start_ms": s, "end_ms": e})
    spans = [replace(sp, start_ms=a, end_ms=b) for sp in truth.spans for a, b in [trim(sp.start_ms, sp.end_ms)]]
    return replace(truth, parts=parts, spans=spans)


def speech_parts(truth: CallTruth) -> list[tuple[int, int, bool]]:
    """(開始, 終了, 個人情報か) の部品。無音は含まない。"""
    return [(p["start_ms"], p["end_ms"], bool(p["pii_type"])) for p in truth.parts]


def score_call(truth: CallTruth, rec: Recognition, res, by_type: dict[str, Counts] | None = None) -> Counts:
    c = Counts()
    merged = res.merged
    for s, e, pii in speech_parts(truth):
        if not pii:
            c.other_ms += e - s
            c.other_masked_ms += overlap(merged, s, e)
    pii_iv = [(sp.start_ms, sp.end_ms) for sp in truth.spans]
    c.masked_ms = sum(y - x for x, y in merged)
    c.masked_true_ms = sum(overlap(pii_iv, x, y) for x, y in merged)  # 伏せた長さのうち個人情報の部分
    toks = rec.tokens
    view = TokenView(rec)
    for sp in truth.spans:
        cs = Counts(spans=1, pii_ms=sp.dur)
        cs.pii_masked_ms = overlap(merged, sp.start_ms, sp.end_ms)
        if sp.dur and cs.pii_masked_ms / sp.dur < COVER_OK:
            cs.span_leaks = 1
        # 文字漏れ：区間に半分以上かかる内容語が、伏せずに残っているか
        for i, t in enumerate(toks):
            d = max(1, t.end_ms - t.start_ms)
            if t.is_punct or not is_content(view.norm[i]):
                continue
            if max(0, min(t.end_ms, sp.end_ms) - max(t.start_ms, sp.start_ms)) / d >= 0.5 and not res.masked[i]:
                cs.text_leaks = 1
                break
        c.add(cs)
        if by_type is not None:
            by_type[sp.pii_type].add(cs)
    return c


def evaluate(pairs: list[tuple[CallTruth, Recognition]], cfg: MaskConfig, base_cache: dict | None = None,
             detectors: Detectors | None = None) -> tuple[dict, dict[str, dict]]:
    total, by_type = Counts(), defaultdict(Counts)
    for truth, rec in pairs:
        base = None
        if base_cache is not None:
            if truth.key not in base_cache:
                base_cache[truth.key] = (detectors or Detectors(cfg.base)).base(TokenView(rec), cfg.base)
            base = base_cache[truth.key]
        res = compute_mask(rec, cfg, base_dets=base, detectors=detectors)
        c = score_call(truth, rec, res, by_type)
        total.add(c)
    return total.metrics(), {PII_LABELS_JA.get(k, k): v.metrics() for k, v in sorted(by_type.items())}


def compare_conditions(pairs, detectors: Detectors, threshold: float = 0.6, pad_ms: int = 120,
                       pad_before_ms: int | None = None) -> dict[str, dict]:
    out, cache = {}, {}
    for name, cfg in CONDITIONS.items():
        cfg = MaskConfig(**{**cfg.__dict__, "threshold": threshold, "pad_ms": pad_ms, "pad_before_ms": pad_before_ms})
        m, _ = evaluate(pairs, cfg, cache, detectors)
        if not cfg.mask_audio:
            # 条件A は音声を伏せないので、音声の漏れは100%。文字だけの漏れ率を見る
            m["漏れ率（時間）"] = 1.0
            m["区間漏れ率"] = 1.0
        out[name] = m
    return out


def sweep(pairs, detectors: Detectors, context: str = "lowconf", steps: int = 21, pad_ms: int = 120,
          pad_before_ms: int | None = None) -> list[dict]:
    cache, rows = {}, []
    base_cfg = MaskConfig(context=context, pad_ms=pad_ms, pad_before_ms=pad_before_ms)
    for k in range(steps):
        thr = k / (steps - 1)
        m, _ = evaluate(pairs, with_threshold(base_cfg, thr), cache, detectors)
        rows.append({"閾値": thr, **m})
    return rows


def timing_errors(pairs) -> dict:
    """正解の部品の境界と、認識した単語の時刻のずれ（ms）。個人情報の部品だけを見る。"""
    starts, ends = [], []
    for truth, rec in pairs:
        toks = [t for t in rec.tokens if not t.is_punct]
        for sp in truth.spans:
            inside = [t for t in toks if min(t.end_ms, sp.end_ms) - max(t.start_ms, sp.start_ms) > 0]
            if inside:
                starts.append(inside[0].start_ms - sp.start_ms)
                ends.append(inside[-1].end_ms - sp.end_ms)

    def stats(v: list[int]) -> dict:
        if not v:
            return {"n": 0}
        a = sorted(abs(x) for x in v)
        return {"n": len(v), "平均": sum(v) / len(v), "abs_p50": a[len(a) // 2], "abs_p90": a[int(len(a) * 0.9) - 1 if len(a) > 1 else 0], "abs_max": a[-1]}

    return {"開始": stats(starts), "終了": stats(ends)}


# ---------------------------------------------------------------- 出力


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def table_md(rows: dict[str, dict], first_col: str = "条件", leaks_only: bool = False) -> str:
    cols = ["漏れ率（時間）", "区間漏れ率", "文字漏れ率"] + ([] if leaks_only else ["消しすぎ率", "適合率"])
    lines = [f"| {first_col} | " + " | ".join(cols) + " | 件数 |", "|---" * (len(cols) + 2) + "|"]
    for name, m in rows.items():
        lines.append(f"| {name} | " + " | ".join(fmt_pct(m[c]) for c in cols) + f" | {m['件数']} |")
    return "\n".join(lines)


def plot_sweep(rows: list[dict], path: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    x = [r["閾値"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(x, [r["漏れ率（時間）"] * 100 for r in rows], marker="o", ms=3, color="#c0392b", label="漏れ率（時間）")
    ax.plot(x, [r["区間漏れ率"] * 100 for r in rows], marker="s", ms=3, color="#e67e22", label="区間漏れ率")
    ax.plot(x, [r["消しすぎ率"] * 100 for r in rows], marker="^", ms=3, color="#2c3e50", label="消しすぎ率")
    ax.set_xlabel("confidence の閾値（これ未満を文脈の判定で伏せる）")
    ax.set_ylabel("%")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
