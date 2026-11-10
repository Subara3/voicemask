"""AmiVoice API での認識（公式ライブラリ amivoice-api-client を使う）。

- 同期HTTP：録音ファイル1本を送る。結果は1区間にまとまって返るので、単語間の無音で発話に区切り直す
- WebSocket：実時間で音声を流し、イベントが届いた時刻を記録する（リアルタイムで伏せるときの遅れの評価用）

決めていること
- confidenceLevel=0 を明示する。既定の0.1では、信頼度の低い発話が結果ごと捨てられ、
  文字に出ないまま音声に個人情報が残る
- ログ保存なしの接続先を使う（logging_opt_out=True）
- E2Eエンジンは20秒を超える音声で結果が繰り返されることがあるので、長い無音で区切って送れるようにする
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio
from .models import Recognition, Segment, Token

ENGINES = {"hybrid": "-a-general", "e2e": "-a2-ja-general", "e2e-batch": "-a2b-ja-general"}


def engine_name(e: str) -> str:
    return ENGINES.get(e, e)


def make_settings(engine: str, profile_words: str | None = None, profile_id: str | None = None,
                  codec: str | None = None, keep_filler: bool = False, confidence_level: float = 0.0):
    from amivoice import RecognitionSettings

    kw = dict(grammar_file_names=engine_name(engine), confidence_level=confidence_level, logging_opt_out=True)
    if profile_words:
        kw["profile_words"] = profile_words
    if profile_id:
        kw["profile_id"] = profile_id
    if codec:
        kw["codec"] = codec
    if keep_filler:
        kw["keep_filler_token"] = True
    return RecognitionSettings(**kw)


# ---------------------------------------------------------------- 応答の変換


def _tokens_from_utterance(u, offset_ms: int = 0) -> list[Token]:
    toks = []
    for t in u.tokens:
        toks.append(Token(
            written=t.written or "",
            start_ms=(t.starttime if t.starttime is not None else -1) + (offset_ms if t.starttime is not None else 0),
            end_ms=(t.endtime if t.endtime is not None else -1) + (offset_ms if t.endtime is not None else 0),
            confidence=float(t.confidence) if t.confidence is not None else 1.0,
            spoken=t.spoken or "",
        ))
    _fill_missing_times(toks, (u.starttime or 0) + offset_ms, (u.endtime or 0) + offset_ms)
    return toks


def _fill_missing_times(toks: list[Token], seg_start: int, seg_end: int) -> None:
    """時刻のない単語（句読点など）に、前後の単語の時刻を入れる。"""
    for i, t in enumerate(toks):
        if t.start_ms < 0 or t.end_ms < 0:
            prev_end = next((toks[j].end_ms for j in range(i - 1, -1, -1) if toks[j].end_ms >= 0), seg_start)
            t.start_ms = t.end_ms = prev_end


def split_by_gaps(tokens: list[Token], gap_ms: int = 600) -> list[Segment]:
    """単語間の無音か、文の終わり（。？）の後の無音で発話に区切る（同期HTTPの結果は1区間で返るため）。

    話者が替わる間（0.3〜1秒）は無音だけでは切れないことがあるので、文末の後は短い無音でも切る。
    """
    segs: list[list[Token]] = []
    for t in tokens:
        after_end = bool(segs) and segs[-1][-1].written in ("。", "？", "?")
        gap = t.start_ms - _last_end(segs[-1]) if segs else 0
        if segs and not t.is_punct and (gap >= gap_ms or (after_end and gap >= gap_ms // 3)):
            segs.append([t])
        elif segs:
            segs[-1].append(t)
        else:
            segs.append([t])
    out = []
    for g in segs:
        timed = [t for t in g if not t.is_punct]
        conf = [t.confidence for t in timed]
        out.append(Segment(g, (timed or g)[0].start_ms, (timed or g)[-1].end_ms, sum(conf) / len(conf) if conf else 0.0))
    return out


def _last_end(g: list[Token]) -> int:
    return max((t.end_ms for t in g if not t.is_punct), default=g[-1].end_ms)


def results_to_recognition(audio_path: str, engine: str, payloads: list[tuple[int, dict]], gap_ms: int | None = 600) -> Recognition:
    """(オフセットms, 応答JSON) の並びから Recognition を作る。保存した JSON からも同じように作れる。"""
    from amivoice.models import parse_result

    tokens: list[Token] = []
    segs: list[Segment] = []
    for offset, data in payloads:
        r = parse_result(json.dumps(data, ensure_ascii=False), raise_on_error=False)
        for u in r.results:
            tk = _tokens_from_utterance(u, offset)
            if not tk:
                continue
            if gap_ms is None:
                timed = [t for t in tk if not t.is_punct] or tk
                conf = [t.confidence for t in timed]
                segs.append(Segment(tk, timed[0].start_ms, timed[-1].end_ms, sum(conf) / len(conf)))
            else:
                tokens.extend(tk)
    if gap_ms is not None:
        segs = split_by_gaps(tokens, gap_ms)
    segs.sort(key=lambda s: s.start_ms)
    return Recognition(audio_path, engine_name(engine), segs, [{"offset_ms": o, "data": d} for o, d in payloads])


# ---------------------------------------------------------------- 同期HTTP


def split_points(x: np.ndarray, sr: int, max_s: float, min_gap_ms: int = 300) -> list[tuple[int, int]]:
    """max_s 秒以下の塊に分ける。切る位置は、できるだけ長い無音の真ん中にする。"""
    if x.size <= max_s * sr:
        return [(0, x.size)]
    win = sr // 100
    n = x.size // win
    level = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    quiet = level < 10 ** (-40 / 20)
    out, start = [], 0
    while x.size - start > max_s * sr:
        lo, hi = start // win + int(2 * 100), int((start + max_s * sr) // win)
        best, best_len, run = None, 0, 0
        for i in range(lo, min(hi, n)):
            run = run + 1 if quiet[i] else 0
            if run * 10 >= min_gap_ms and run >= best_len:
                best, best_len = i - run // 2, run
        cut = (best if best is not None else hi) * win
        out.append((start, cut))
        start = cut
    out.append((start, x.size))
    return out


def recognize_file(path: str | Path, engine: str, max_chunk_s: float | None = None, **settings_kw) -> Recognition:
    from amivoice import AmiVoiceClient
    from amivoice.exceptions import RecognitionError

    mulaw = audio.wav_is_mulaw(path)
    # 応答が返らないまま80分待ち続けたことがある（E2E、2026-11-10）。待つ上限を付け、切れたら2回まで送り直す
    client = AmiVoiceClient(settings=make_settings(engine, **settings_kw), connect_timeout=15000, receive_timeout=180000)

    def send(p: str):
        for k in range(3):
            try:
                return client.transcribe(p)
            except RecognitionError:
                raise
            except Exception as e:  # 時間切れ・接続の切断
                if k == 2:
                    raise
                print(f"\n送り直します（{type(e).__name__}）", flush=True)

    payloads: list[tuple[int, dict]] = []
    if not max_chunk_s:
        r = send(str(path))
        payloads.append((0, r.data))
    else:
        x, sr = audio.read_wav(path)
        tmp = Path(path).with_suffix(".chunk.wav")
        for a, b in split_points(x, sr, max_chunk_s):
            audio.write_wav(tmp, x[a:b], sr, mulaw=mulaw)
            try:
                r = send(str(tmp))
                payloads.append((audio.samples_to_ms(a, sr), r.data))
            except RecognitionError as e:
                if "below the threshold" in str(e):
                    # confidenceLevel=0 でも、塊全体の信頼度が低いと結果ごと返らない（E2E で実測）。
                    # 認識できなかった塊として空で残し、何本あったかを後で数えられるようにする
                    payloads.append((audio.samples_to_ms(a, sr), {"results": [], "rejected": str(e)}))
                elif "no text" not in str(e):  # 無音の塊は結果が空になる
                    raise
            finally:
                tmp.unlink(missing_ok=True)
    return results_to_recognition(str(path), engine, payloads)


def save_recognition(rec: Recognition, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8")


def load_recognition(path: str | Path) -> Recognition:
    return Recognition.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- WebSocket


@dataclass
class StreamLog:
    audio_path: str
    engine: str
    chunk_ms: int
    events: list[dict] = field(default_factory=list)  # {"type", "arrived_ms", "data"(final のみ)}
    sent_done_ms: float = 0.0

    def finals(self) -> list[tuple[float, dict]]:
        return [(e["arrived_ms"], e["data"]) for e in self.events if e["type"] == "final" and e.get("data")]

    def to_recognition(self) -> Recognition:
        # WebSocket は発話区間ごとに結果が届くので区切り直さない
        return results_to_recognition(self.audio_path, self.engine, [(0, d) for _, d in self.finals()], gap_ms=None)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StreamLog":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def stream_file(path: str | Path, engine: str, chunk_ms: int = 100, realtime: bool = True, **settings_kw) -> StreamLog:
    """音声を実時間で流し、各イベントが届いた時刻（送り始めからの経過ms）を記録する。"""
    from amivoice import AmiVoiceClient

    x, sr = audio.read_wav(path)
    mulaw = audio.wav_is_mulaw(path)
    if mulaw:
        data, codec, bpm = audio.mulaw_encode(x).tobytes(), "MULAW", sr // 1000
    else:
        data, codec, bpm = audio.to_pcm16_bytes(x), {16000: "LSB16K", 8000: "LSB8K"}[sr], sr * 2 // 1000
    log = StreamLog(str(path), engine_name(engine), chunk_ms)
    client = AmiVoiceClient(settings=make_settings(engine, codec=codec, **settings_kw))
    t0 = [0.0]

    def now() -> float:
        return (time.perf_counter() - t0[0]) * 1000.0

    with client.stream() as session:
        def receive():
            for ev in session.events():
                item = {"type": str(ev.type), "arrived_ms": now()}
                if ev.is_final and ev.result is not None:
                    item["data"] = ev.result.data
                elif ev.utterance is not None:
                    item["time"] = ev.utterance.time
                log.events.append(item)

        t0[0] = time.perf_counter()
        th = threading.Thread(target=receive, daemon=True)
        th.start()
        step = chunk_ms * bpm
        for i in range(0, len(data), step):
            session.write(data[i:i + step])
            if realtime:
                wait = (i + step) / bpm - now()
                if wait > 0:
                    time.sleep(wait / 1000.0)
        log.sent_done_ms = now()
        session.end(timeout=60)
        th.join(timeout=5)
    return log
