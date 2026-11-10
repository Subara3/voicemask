"""台本から通話音声を合成し、個人情報の正解区間を記録する。

発話を「個人情報の前・個人情報・後」の部品に分けて別々に合成し、前後の無音を切ってつなぐ。
つないだ位置がそのまま正解の時刻になるので、合成エンジンの時刻情報に頼らない。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio
from .models import CallScript, CallTruth, TruthSpan
from .tts import TTS, Voice

SR = 16000

# 版：雑音の SN比（None なら雑音なし）と電話音質（8kHz・mu-law の WAV で保存）
VARIANTS = {
    "clean": (None, False),
    "noise20": (20.0, False),
    "noise10": (10.0, False),
    "tel": (None, True),
}


@dataclass
class SynthConfig:
    part_gap_ms: tuple[int, int] = (30, 80)
    turn_gap_ms: tuple[int, int] = (400, 900)
    edge_ms: int = 500
    rates: tuple[float, ...] = (0.9, 1.0, 1.15)
    trim_db: float = -45.0
    keep_ms: int = 15


def trim_silence(x: np.ndarray, sr: int, db: float = -45.0, keep_ms: int = 15) -> np.ndarray:
    """前後の無音を切る。10ms ごとの RMS が閾値を超える範囲を残す。"""
    win = max(1, sr // 100)
    n = x.size // win
    if n == 0:
        return x
    level = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    idx = np.nonzero(level > 10 ** (db / 20))[0]
    if idx.size == 0:
        return x[:0]
    keep = audio.ms_to_samples(keep_ms, sr)
    return x[max(0, idx[0] * win - keep): min(x.size, (idx[-1] + 1) * win + keep)]


def pick_voices(engines: list[TTS], rng: random.Random) -> dict[str, tuple[TTS, Voice]]:
    """オペレーターと顧客の声を選ぶ。氏名を読むのは顧客なので、Azure があれば顧客を Azure に寄せる。"""
    def choose(pool: list[TTS], gender: str, avoid: Voice | None) -> tuple[TTS, Voice]:
        eng = rng.choice(pool)
        cands = [v for v in eng.voices if v.gender == gender and v != avoid] or eng.voices
        return eng, rng.choice(cands)

    op = choose(engines, rng.choice("FM"), None)
    azure = [e for e in engines if e.name == "azure"]
    cu_pool = azure if azure and rng.random() < 0.7 else engines
    cu = choose(cu_pool, rng.choice("FM"), op[1])
    return {"operator": op, "customer": cu}


def synth_call(script: CallScript, engines: list[TTS], rng: random.Random, cfg: SynthConfig | None = None):
    cfg = cfg or SynthConfig()
    who = pick_voices(engines, rng)
    rate = {k: rng.choice(cfg.rates) for k in who}
    chunks = [audio.silence(cfg.edge_ms, SR)]
    cur = chunks[0].size
    spans: list[TruthSpan] = []
    parts_log: list[dict] = []
    for ui, utt in enumerate(script.utterances):
        eng, voice = who[utt.speaker]
        for pi, part in enumerate(utt.parts):
            x, sr = eng.synthesize(part.reading, voice, rate[utt.speaker])
            x = trim_silence(audio.resample(x, sr, SR), SR, cfg.trim_db, cfg.keep_ms)
            s, e = audio.samples_to_ms(cur, SR), audio.samples_to_ms(cur + x.size, SR)
            parts_log.append({"utt": ui, "speaker": utt.speaker, "start_ms": s, "end_ms": e,
                              "text": part.text, "pii_type": part.pii_type, "slot": part.slot})
            if part.pii_type:
                spans.append(TruthSpan(s, e, part.pii_type, part.text, utt.speaker))
            chunks.append(x)
            cur += x.size
            if pi < len(utt.parts) - 1:
                g = audio.silence(rng.randint(*cfg.part_gap_ms), SR)
                chunks.append(g)
                cur += g.size
        g = audio.silence(rng.randint(*cfg.turn_gap_ms), SR)
        chunks.append(g)
        cur += g.size
    chunks.append(audio.silence(cfg.edge_ms, SR))
    voices = {k: f"{v[1].engine}:{v[1].name}" for k, v in who.items()}
    voices.update({f"rate_{k}": r for k, r in rate.items()})
    return audio.normalize_peak(np.concatenate(chunks)), spans, parts_log, voices


def synth_dataset(scripts: list[CallScript], engines: list[TTS], out_dir: str | Path,
                  variants: list[str], seed: int = 1) -> Path:
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    truth_path = out / "truth.jsonl"
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    with truth_path.open("w", encoding="utf-8") as f:
        for i, sc in enumerate(scripts):
            x, spans, parts, voices = synth_call(sc, engines, rng)
            for var in variants:
                snr, tel = VARIANTS[var]
                y, sr = x, SR
                if snr is not None:
                    y = audio.add_noise(y, snr, nrng)
                if tel:
                    y, sr = audio.to_telephone(y, sr)
                rel = f"audio/{sc.call_id}__{var}.wav"
                audio.write_wav(out / rel, y, sr, mulaw=tel)
                t = CallTruth(sc.call_id, rel, audio.samples_to_ms(y.size, sr), sr, var, sc.scenario, spans, parts, voices)
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            print(f"\r合成 {i + 1}/{len(scripts)}", end="", flush=True)
    print()
    return truth_path


def load_truths(path: str | Path, variant: str | None = None) -> list[CallTruth]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = CallTruth.from_dict(json.loads(line))
            if variant is None or t.variant == variant:
                out.append(t)
    return out


def save_scripts(scripts: list[CallScript], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in scripts), encoding="utf-8")


def load_scripts(path: str | Path) -> list[CallScript]:
    return [CallScript.from_dict(json.loads(l)) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
