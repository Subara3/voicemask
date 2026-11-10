"""伏せる範囲を決め、文字と音声の両方を伏せる。

検出は単語の添字で返るので、単語の時刻を区間に変えれば、文字と音声を同じ範囲で伏せられる。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import audio
from .detect.context import detect_answers, detect_lowconf, expand_neighbors
from .detect.rules import detect_rules, extend_address_tail
from .detect.tokenview import TokenView
from .models import PII_LABELS_JA, Detection, MaskInterval, Recognition
from .textnorm import normalize_for_match


@dataclass(frozen=True)
class MaskConfig:
    base: tuple[str, ...] = ("rules", "ner")  # 閾値に左右されない検出器（"llm" も足せる）
    context: str = "lowconf"     # "none" / "lowconf"（条件C）/ "answer"（条件C2）
    threshold: float = 0.6       # これ未満の confidence を「自信がない」とみなす
    window_ms: int = 8000        # 質問の手がかりから何ミリ秒先までを答えとみなすか
    pad_ms: int = 120            # 単語の時刻の後ろに足す余白
    pad_before_ms: int | None = None  # 前に足す余白（None なら pad_ms）。E2E は時刻が遅れて出るので広げる
    shift_ms: int | None = None  # 単語の時刻を前にずらす量。None はエンジンで決める（E2E 450ms、ハイブリッド 0）
    merge_gap_ms: int = 200      # これ以下の隙間は1つの区間にまとめる
    mask_audio: bool = True      # False は文字だけ伏せる（条件A）
    audio_mode: str = "beep"     # "beep" / "silence"

    def label(self) -> str:
        return (f"base={'+'.join(self.base)} context={self.context} thr={self.threshold:g} shift={'auto' if self.shift_ms is None else self.shift_ms} "
                f"pad={self.pad_before_ms if self.pad_before_ms is not None else self.pad_ms}/{self.pad_ms} audio={'on' if self.mask_audio else 'off'}")


# 記事で比べる条件
CONDITIONS = {
    "A": MaskConfig(context="none", mask_audio=False),
    "B": MaskConfig(context="none"),
    "C": MaskConfig(context="lowconf"),
    "C2": MaskConfig(context="answer"),
}


@dataclass
class MaskResult:
    detections: list[Detection]
    intervals: list[MaskInterval]
    merged: list[tuple[int, int]]   # 音声に当てる区間（重なりなし）
    masked: list[bool]              # 単語ごとに伏せたか
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class Detectors:
    """重い検出器（GiNZA・Claude）を1回だけ読み込む。"""

    def __init__(self, names: tuple[str, ...]):
        self.ner = self.llm = None
        if "ner" in names:
            try:
                from .detect.ner import GinzaDetector
                self.ner = GinzaDetector()
            except (ImportError, OSError) as e:
                import sys
                print(f"GiNZA を読み込めないので ner を使わずに続けます（{e}）", file=sys.stderr)
        if "llm" in names:
            from .detect.llm import LLMDetector
            self.llm = LLMDetector()

    def base(self, view: TokenView, names: tuple[str, ...]) -> list[Detection]:
        d: list[Detection] = []
        if "rules" in names:
            d += detect_rules(view)
        if "ner" in names and self.ner:
            d += self.ner.detect(view)
        if "llm" in names and self.llm:
            d += self.llm.detect(view)
        return d + extend_address_tail(view, d)


E2E_SHIFT_MS = 450  # E2E は単語の時刻が約0.45秒遅れて出る（1週目の実測、調整用24本で 300〜500ms を試して450msが最良）


def default_shift(engine: str) -> int:
    return E2E_SHIFT_MS if "-a2-" in engine or engine.startswith("e2e") else 0


def compute_mask(rec: Recognition, cfg: MaskConfig, base_dets: list[Detection] | None = None,
                 detectors: Detectors | None = None) -> MaskResult:
    """base_dets を渡すと、重い検出をやり直さずに文脈の判定だけを行う（閾値を動かす評価用）。"""
    view = TokenView(rec)
    if base_dets is None:
        base_dets = (detectors or Detectors(cfg.base)).base(view, cfg.base)
    dets = list(base_dets)
    if cfg.context == "lowconf":
        dets += expand_neighbors(view, dets, cfg.threshold)
        dets += detect_lowconf(view, cfg.threshold, cfg.window_ms)
    elif cfg.context == "answer":
        dets += expand_neighbors(view, dets, cfg.threshold)
        dets += detect_answers(view, cfg.window_ms)

    toks = view.tokens
    cfg = replace(cfg, shift_ms=default_shift(rec.engine) if cfg.shift_ms is None else cfg.shift_ms)
    flag = [False] * len(toks)
    typ = [""] * len(toks)
    intervals: list[MaskInterval] = []
    order = {"rule": 0, "ner": 1, "llm": 1, "neighbor": 2, "answer": 3, "context": 3}
    for d in sorted(dets, key=lambda d: order.get(d.source, 9)):
        span = [i for i in range(d.token_start, d.token_end) if not toks[i].is_punct]
        if not span:
            continue
        for i in span:
            flag[i] = True
            typ[i] = typ[i] or ("SUSPECT" if d.source in ("context", "answer") else d.pii_type)
        if cfg.mask_audio:
            intervals.append(MaskInterval(
                max(0, toks[span[0]].start_ms - cfg.shift_ms - (cfg.pad_ms if cfg.pad_before_ms is None else cfg.pad_before_ms)),
                max(0, toks[span[-1]].end_ms - cfg.shift_ms + cfg.pad_ms),
                d.pii_type, d.source, d.reason, min(toks[i].confidence for i in span)))
    merged = audio.merge_intervals([(iv.start_ms, iv.end_ms) for iv in intervals], cfg.merge_gap_ms)
    merged = _bridge_silence(merged, toks, flag, cfg.shift_ms)
    return MaskResult(dets, intervals, merged, flag, _lines(rec, flag, typ))


def mask_then_llm(rec: Recognition, cfg: MaskConfig, detectors: Detectors) -> tuple[MaskResult, MaskResult]:
    """手元の検出で伏せてから、伏せたあとの文字だけを Claude に送り、取りこぼしを足す。

    Claude に届く個人情報は、手元で取りこぼした分だけになる。返すのは (手元だけ, 手元＋Claude)。
    """
    from .detect.llm import LLMDetector

    view = TokenView(rec)
    base = detectors.base(view, cfg.base)
    first = compute_mask(rec, cfg, base_dets=base)
    llm = detectors.llm or LLMDetector()
    extra = llm.detect(view, first.masked)
    return first, compute_mask(rec, cfg, base_dets=base + extra)


def _bridge_silence(merged: list[tuple[int, int]], toks, flag: list[bool], shift_ms: int = 0,
                    max_gap_ms: int = 1500) -> list[tuple[int, int]]:
    """隣り合う区間のあいだに、伏せていない声の単語がなければつなぐ（「田中」と「翔太」のあいだの間）。

    つなぐのは無音だけなので、消しすぎは増えない。区間の切れ目から名前の区切りが分かることも防げる。
    """
    out: list[tuple[int, int]] = []
    for s, e in merged:
        if out and s - out[-1][1] <= max_gap_ms:
            ps, pe = out[-1]
            spoken = any(not f and not t.is_punct and t.end_ms - shift_ms > pe and t.start_ms - shift_ms < s
                         for t, f in zip(toks, flag))
            if not spoken:
                out[-1] = (ps, e)
                continue
        out.append((s, e))
    return out


def _lines(rec: Recognition, flag: list[bool], typ: list[str]) -> list[str]:
    out, k = [], 0
    for seg in rec.segments:
        buf, prev = [], None
        for t in seg.tokens:
            if flag[k]:
                lab = f"［{PII_LABELS_JA.get(typ[k], '個人情報')}］"
                if lab != prev:
                    buf.append(lab)
                prev = lab
            else:
                buf.append(t.written)
                prev = None
            k += 1
        m, s = divmod(seg.start_ms // 1000, 60)
        out.append(f"[{m:02d}:{s:02d}] " + "".join(buf))
    return out


# ---------------------------------------------------------------- 監査ログ


def audit_key() -> bytes:
    k = os.environ.get("VOICEMASK_AUDIT_KEY")
    return k.encode() if k else secrets.token_bytes(32)


def audit_records(rec: Recognition, res: MaskResult, key: bytes) -> list[dict]:
    """伏せた区間の記録。値そのものは残さず、鍵つきハッシュ（HMAC）だけ残す。

    電話番号のように候補の少ない値は、鍵なしのハッシュだと総当たりで戻せるため。
    """
    toks = rec.tokens
    out = []
    for iv in res.intervals:
        inside = "".join(t.written for t in toks if t.start_ms >= iv.start_ms and t.end_ms <= iv.end_ms)
        h = hmac.new(key, normalize_for_match(inside).encode(), hashlib.sha256).hexdigest()[:16]
        d = asdict(iv)
        d.pop("reason")
        d["reason"] = iv.reason.split(" confidence")[0]
        out.append({**d, "hmac": h})
    return out


# ---------------------------------------------------------------- ファイル単位の処理


def process(rec: Recognition, src_audio: str | Path, out_dir: str | Path, cfg: MaskConfig,
            detectors: Detectors | None = None, delete_source: bool = False, llm_after: bool = False) -> MaskResult:
    """伏せた音声・伏せた文字・監査ログを書き出す。delete_source なら元の音声を消す。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src_audio).stem
    if llm_after:  # 手元で伏せたあとの文字だけを Claude に見せ、取りこぼしを足す
        res = mask_then_llm(rec, cfg, detectors or Detectors(cfg.base))[1]
    else:
        res = compute_mask(rec, cfg, detectors=detectors)
    if cfg.mask_audio:
        x, sr = audio.read_wav(src_audio)
        audio.write_wav(out / f"{stem}.masked.wav", audio.apply_mask(x, sr, res.merged, cfg.audio_mode), sr,
                        mulaw=audio.wav_is_mulaw(src_audio))
    (out / f"{stem}.masked.txt").write_text(res.text + "\n", encoding="utf-8")
    log = {"config": cfg.label(), "intervals": audit_records(rec, res, audit_key()), "source_deleted": delete_source}
    (out / f"{stem}.audit.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    if delete_source:
        Path(src_audio).unlink(missing_ok=True)
    return res


def with_threshold(cfg: MaskConfig, thr: float) -> MaskConfig:
    return replace(cfg, threshold=thr)
