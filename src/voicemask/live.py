"""通話中にリアルタイムで伏せる。

音声を D ミリ秒ためてから外に出し、そのあいだに届いた認識結果で伏せる区間を決める。

- 途中結果（U）：単語に時刻がないので、発話の始まり（S）から届いた時点までに、文字数で割り振った仮の時刻を付けて
  規則で探す。仮の時刻はずれるので、見つけた区間は前後を広めにとる（仮の区間）
- 確定結果（A）：単語の時刻で、規則・GiNZA・質問の答えを探し直す（確かな区間）。仮の区間はここで置き換わる
- 外に出す音声は、出す時点で分かっている区間（確かな区間＋仮の区間）だけでピー音にする。出してしまった音声は戻せない

判断は D によらないので、届いたイベントを時刻どおりに流し直せば、どの D でも採点できる（`replay`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .asr import results_to_recognition, split_by_gaps
from .audio import merge_intervals
from .detect.context import _END, _REQUEST, _WINDOW, _cue_hits
from .detect.rules import detect_rules, extend_address_tail
from .detect.tokenview import TokenView
from .mask import MaskConfig, Detectors, compute_mask, default_shift
from .models import Recognition, Token

PARTIAL_LAG_MS = 200   # 途中結果は、届いた時点より少し前までの音声を表している
PARTIAL_STEP_MS = 100  # 途中結果の届く間隔（既定）
PROVISIONAL_PAD_MS = 400  # 仮の時刻で見つけた区間に、前後に足す余白
ANSWER_HEAD_MS = 200      # 質問のあとの発話は、始まり（S）のこれだけ前から伏せておく


@dataclass
class Snapshot:
    at_ms: float                       # この判断が付いた時刻（送り始めからの経過ms＝音声の時刻）
    intervals: list[tuple[int, int]]   # この時点で伏せると決めている区間
    lines: list[str] = field(default_factory=list)  # この時点の伏せた文字（画面表示用）


class LiveMasker:
    def __init__(self, cfg: MaskConfig, detectors: Detectors, engine: str = "-a-general",
                 audio_path: str = "", provisional_pad_ms: int = PROVISIONAL_PAD_MS, ahead: bool = False):
        """ahead=True は、質問のあとの発話や、途中で切れた答えの続きを、文字が出る前から伏せておく。

        調整用40本では、1.5秒ためたときの漏れが 7.8%→3.6% に減るかわりに、消しすぎが 4.3%→9.2% に増えた
        （質問の次の発話が担当者の言い直しだったり、「はい、カード番号は」まで伏せたりする）。既定は使わない。
        """
        self.cfg = cfg
        self.det = detectors
        self.engine = engine
        self.audio_path = audio_path
        self.pad = provisional_pad_ms
        self.ahead = ahead
        self.finals: list[dict] = []
        self.final_rec: Recognition | None = None
        self.final_base: list = []      # 確定分の検出（GiNZA を含む）。確定結果が届いたときだけ計算する
        self.final_res = None
        self.utt_start = 0
        self.current: list[tuple[int, int]] = []
        self.base_iv: list[tuple[int, int]] = []   # 確定・途中結果から決めた区間
        self.waiting: int | None = None            # 質問のあと、答えを待っている（値は答えの長さの上限ms）
        self.waiting_after = 0                     # 質問の終わりの時刻
        self.open_iv: tuple[int, int] | None = None  # 答えの発話を、文字が出る前から伏せておく区間
        self.lines: list[str] = []
        self.history: list[Snapshot] = []

    # ------------------------------------------------------------ イベント

    def on_start(self, audio_ms: int, at_ms: float) -> None:
        self.utt_start = audio_ms
        if self.waiting is not None and audio_ms >= self.waiting_after - 300:
            # 「お名前を伺えますか」の次の発話は、文字が出るのを待たずに頭から伏せておく。確定結果で正しい範囲に直る
            self.open_iv = (max(0, audio_ms - ANSWER_HEAD_MS), audio_ms + self.waiting)
            self.waiting = None
            self._update(at_ms)

    def _question_at_end(self, view: TokenView) -> int | None:
        """いちばん新しい発話が、個人情報を尋ねて終わっていれば、答えの長さの上限ms。"""
        last = len(view.seg_bounds) - 1
        for typ, si, pos in _cue_hits(view):
            if si == last:
                rest = view.segment_text(si)[pos:]
                if not rest or _REQUEST.match(rest):
                    return _WINDOW.get(typ, 5000)
        return None

    @staticmethod
    def _answer_continues(rec: Recognition, masked: list[bool]) -> bool:
        """いちばん新しい発話が、伏せた語のまま文の終わりなしで切れているか。"""
        toks = rec.tokens
        i = len(toks) - 1
        while i >= 0 and toks[i].is_punct and toks[i].written not in ("。", "？", "?"):
            i -= 1
        if i < 0 or toks[i].written in ("。", "？", "?"):
            return False
        # 最後の語か、その1つ前（「956の」の「の」）を伏せていれば、答えの途中で切れている
        return masked[i] or (i > 0 and masked[i - 1] and len(toks[i].written) <= 2)

    def _update(self, at_ms: float) -> None:
        self.current = merge_intervals(self.base_iv + ([self.open_iv] if self.open_iv else []), 0)
        self._snap(at_ms)

    def on_final(self, data: dict, at_ms: float) -> None:
        self.finals.append(data)
        rec = results_to_recognition(self.audio_path, self.engine, [(0, d) for d in self.finals])
        view = TokenView(rec)
        self.final_rec = rec
        self.final_base = self.det.base(view, self.cfg.base)
        self.final_res = compute_mask(rec, self.cfg, base_dets=self.final_base)
        self.base_iv = list(self.final_res.merged)
        self.lines = self.final_res.lines
        self.open_iv = None
        win = self._question_at_end(view) if self.ahead else None
        if self.ahead and win is None and self._answer_continues(rec, self.final_res.masked):
            win = 5000  # 「956の」で発話が切れ、「0422の…」が次の発話で来る。続きも頭から伏せておく
        if win is not None:
            q_end = rec.segments[-1].end_ms
            if self.utt_start > q_end:  # 答えがもう始まっている（質問の確定結果より先に話し始めた）
                self.open_iv = (max(0, self.utt_start - ANSWER_HEAD_MS), self.utt_start + win)
            else:
                self.waiting, self.waiting_after = win, q_end
        self._update(at_ms)

    def on_partial(self, data: dict, at_ms: float) -> None:
        words = [t.get("written", "") for r in data.get("results", []) for t in r.get("tokens", [])]
        words = [w for w in words if w and w.strip(".")]
        if not words:
            return
        # 末尾の「.」は、まだ文字になっていない音声の印（100msごとに1つ増える）。そのぶん文字の終わりは前にある
        text = "".join(t.get("written", "") for r in data.get("results", []) for t in r.get("tokens", []))
        pending = (len(text) - len(text.rstrip("."))) * PARTIAL_STEP_MS
        end = max(self.utt_start + 1, int(at_ms) - PARTIAL_LAG_MS - pending)
        total = sum(len(w) for w in words)
        toks, pos = [], 0
        for w in words:  # 発話の始まりから今までに、文字数で割り振る
            s = self.utt_start + (end - self.utt_start) * pos // total
            pos += len(w)
            e = self.utt_start + (end - self.utt_start) * pos // total
            toks.append(Token(w, s, e, 1.0, ""))
        if self.open_iv:
            # 先回りの区間は、途中結果に答えの終わり（です・と申します）が出たらそこで打ち切る
            for t in toks:
                if any(t.written.startswith(w) for w in _END):
                    self.open_iv = (self.open_iv[0], max(self.open_iv[0], min(self.open_iv[1], t.end_ms + 100)))
                    break
        base_toks = self.final_rec.tokens if self.final_rec else []
        n_final = len(base_toks)
        # 途中結果の仮の時刻には無音がないので、「。」「？」の後ろで発話を区切る（質問と答えが1つの発話で届くため）
        rec = Recognition(self.audio_path, self.engine, split_by_gaps(base_toks) + _split_at_sentence_end(toks))
        view = TokenView(rec)
        # 確定分の検出は使い回す。途中結果を足した全体には規則をかけ、GiNZA は途中結果の発話だけにかける（全体は重い）
        rules = [d for d in detect_rules(view) if d.token_end > n_final]
        ner = []
        if self.det.ner and "ner" in self.cfg.base:
            from dataclasses import replace as _r
            pv = TokenView(Recognition(self.audio_path, self.engine, split_by_gaps(toks)))
            ner = [_r(d, token_start=d.token_start + n_final, token_end=d.token_end + n_final) for d in self.det.ner.detect(pv)]
        dets = self.final_base + rules + ner
        dets += extend_address_tail(view, dets)
        res = compute_mask(rec, self.cfg, base_dets=dets)
        lo = toks[0].start_ms
        out = list(self.final_res.merged) if self.final_res else []
        for s, e in res.merged:
            if e > lo:  # 途中結果にかかる区間は仮の時刻なので広げる
                out.append((max(0, s - self.pad), e + self.pad))
        self.base_iv = merge_intervals(out, 0)
        self.lines = res.lines
        self._update(at_ms)

    def _snap(self, at_ms: float) -> None:
        self.history.append(Snapshot(at_ms, list(self.current), list(self.lines)))

    # ------------------------------------------------------------ 出す音声

    def masked_at(self, audio_ms: int) -> bool:
        return any(s <= audio_ms < e for s, e in self.current)


def _split_at_sentence_end(toks: list[Token]):
    from .models import Segment

    segs, cur = [], []
    for t in toks:
        cur.append(t)
        if t.written in ("。", "？", "?"):
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return [Segment(g, g[0].start_ms, g[-1].end_ms, 1.0) for g in segs if g]


def feed(masker: LiveMasker, events: list[dict]) -> LiveMasker:
    """StreamLog のイベントを、届いた順に流し直す。"""
    for e in sorted(events, key=lambda e: e["arrived_ms"]):
        if e["type"] == "utterance_started":
            masker.on_start(e.get("time", 0), e["arrived_ms"])
        elif e["type"] == "partial_raw":
            masker.on_partial(e["data"], e["arrived_ms"])
        elif e["type"] == "final" and e.get("data"):
            masker.on_final(e["data"], e["arrived_ms"])
    return masker


def emitted_intervals(history: list[Snapshot], delay_ms: int, end_ms: int, step_ms: int = 10) -> list[tuple[int, int]]:
    """D ミリ秒ためて流したとき、実際にピー音になった区間。

    音声の時刻 x は、時刻 x+D に外へ出る。そのときに分かっている区間に x が入っていれば伏せられる。
    """
    out, k, cur = [], -1, []
    for x in range(0, end_ms, step_ms):
        t = x + delay_ms
        while k + 1 < len(history) and history[k + 1].at_ms <= t:
            k += 1
            cur = history[k].intervals
        if any(s <= x < e for s, e in cur):
            out.append((x, x + step_ms))
    return merge_intervals(out, 0)


def score_live(items, cfg: MaskConfig, detectors: Detectors, delays_ms: list[int], ahead: bool = False) -> list[dict]:
    """items は (正解, StreamLog)。ためる時間ごとに、区間漏れ率と消しすぎ率を出す。"""
    from .evaluate import COVER_OK, overlap, speech_parts

    hist = []
    for truth, log in items:
        m = feed(LiveMasker(cfg, detectors, log.engine, log.audio_path, ahead=ahead), log.events)
        hist.append((truth, m.history))
    rows = []
    for d in delays_ms:
        spans = leaks = 0
        pii = pii_m = other = other_m = 0
        for truth, h in hist:
            eff = emitted_intervals(h, d, truth.duration_ms)
            for sp in truth.spans:
                cov = overlap(eff, sp.start_ms, sp.end_ms)
                spans += 1
                pii += sp.dur
                pii_m += cov
                leaks += 1 if sp.dur and cov / sp.dur < COVER_OK else 0
            for s, e, is_pii in speech_parts(truth):
                if not is_pii:
                    other += e - s
                    other_m += overlap(eff, s, e)
        rows.append({"ためる時間ms": d, "区間漏れ率": leaks / spans if spans else 0.0,
                     "漏れ率（時間）": 1 - pii_m / pii if pii else 0.0,
                     "消しすぎ率": other_m / other if other else 0.0, "件数": spans})
    return rows


def run_live(path, delay_ms: int, cfg: MaskConfig, detectors: Detectors, engine: str = "hybrid",
             with_log: bool = False, beep_hz: int = 1000, on_line=None, ahead: bool = False):
    """録音ファイルを通話に見立てて実時間で流し、delay_ms 遅れで伏せた音声を作る。

    返すのは (伏せた音声, サンプリング周波数, StreamLog, LiveMasker, 実際にピー音にした区間)。
    """
    import threading
    import time

    import numpy as np

    from . import audio
    from .asr import engine_name, stream_file

    x, sr = audio.read_wav(path)
    dur = int(len(x) * 1000 / sr)
    masker = LiveMasker(cfg, detectors, engine_name(engine), str(path), ahead=ahead)
    out = np.zeros_like(x)
    emitted: list[tuple[int, int]] = []
    clock = {"now": None}
    done = threading.Event()
    lock = threading.Lock()
    shown = [0]

    def on_event(item):
        with lock:
            feed(masker, [item])
            if on_line and item["type"] == "final":
                for ln in masker.lines[shown[0]:]:
                    on_line(ln)
                shown[0] = len(masker.lines)

    def emitter():
        pos, step = 0, 10  # 10ms ずつ外に出す
        while pos < dur:
            if clock["now"] is None:
                time.sleep(0.01)
                continue
            target = int(clock["now"]()) - delay_ms
            while pos < min(target, dur):
                a, b = pos * sr // 1000, min(len(x), (pos + step) * sr // 1000)
                with lock:
                    hit = masker.masked_at(pos)
                if hit:
                    t = np.arange(a, b) / sr  # 通しの位置から作る（10msごとに位相が切れるとプツプツ鳴る）
                    out[a:b] = 0.3 * np.sin(2 * np.pi * beep_hz * t)
                    emitted.append((pos, pos + step))
                else:
                    out[a:b] = x[a:b]
                pos += step
            time.sleep(0.02)
        done.set()

    th = threading.Thread(target=emitter, daemon=True)
    th.start()
    log = stream_file(path, engine, on_event=on_event, on_begin=lambda now: clock.__setitem__("now", now),
                      logging_opt_out=not with_log)
    done.wait(timeout=delay_ms / 1000 + 120)
    return out, sr, log, masker, merge_intervals(emitted, 0)
