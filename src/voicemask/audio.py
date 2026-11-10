"""音声の読み書きと加工（雑音、電話音質、伏せ字）。

WAV は 16bit PCM と 8bit mu-law（フォーマット番号7）を読み書きする。
内部では float32（-1〜1）で扱う。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

MULAW_MU = 255


def ms_to_samples(ms: float, sr: int) -> int:
    return int(round(ms * sr / 1000.0))


def samples_to_ms(n: int, sr: int) -> int:
    return int(round(n * 1000.0 / sr))


def silence(ms: float, sr: int) -> np.ndarray:
    return np.zeros(ms_to_samples(ms, sr), dtype=np.float32)


# ---------------------------------------------------------------- mu-law


def mulaw_encode(x: np.ndarray) -> np.ndarray:
    """G.711 mu-law（8bit）。"""
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    mag = np.minimum(np.abs(pcm), 32635) + 0x84
    exp = np.floor(np.log2(mag)).astype(np.int32) - 7
    exp = np.clip(exp, 0, 7)
    mant = (mag >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8)


def mulaw_decode(u: np.ndarray) -> np.ndarray:
    u = ~u.astype(np.int32) & 0xFF
    sign = u & 0x80
    exp = (u >> 4) & 0x07
    mant = u & 0x0F
    mag = (((mant << 3) + 0x84) << exp) - 0x84
    pcm = np.where(sign != 0, -mag, mag)
    return (pcm / 32768.0).astype(np.float32)


# ---------------------------------------------------------------- WAV


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"WAV ではありません: {path}")
    pos, fmt, sr, ch, bits, body = 12, None, 0, 1, 16, b""
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt, ch, sr, _, _, bits = struct.unpack("<HHIIHH", chunk[:16])
        elif cid == b"data":
            body = chunk
        pos += 8 + size + (size & 1)
    if fmt == 1 and bits == 16:
        x = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    elif fmt == 7 and bits == 8:
        x = mulaw_decode(np.frombuffer(body, dtype=np.uint8))
    else:
        raise ValueError(f"未対応の WAV です（format={fmt}, bits={bits}）: {path}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr


def write_wav(path: str | Path, x: np.ndarray, sr: int, mulaw: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    if mulaw:
        body, fmt, bits, ext = mulaw_encode(x).tobytes(), 7, 8, b"\x00\x00"
    else:
        body, fmt, bits, ext = (np.round(x * 32767).astype("<i2")).tobytes(), 1, 16, b""
    block = bits // 8
    fmt_chunk = struct.pack("<HHIIHH", fmt, 1, sr, sr * block, block, bits) + ext
    out = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt_chunk) + 8 + len(body)) + b"WAVE"
    out += b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
    out += b"data" + struct.pack("<I", len(body)) + body
    Path(path).write_bytes(out)


def wav_is_mulaw(path: str | Path) -> bool:
    data = Path(path).read_bytes()[:64]
    i = data.find(b"fmt ")
    return i >= 0 and struct.unpack("<H", data[i + 8:i + 10])[0] == 7


def to_pcm16_bytes(x: np.ndarray) -> bytes:
    return np.round(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


# ---------------------------------------------------------------- 加工


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = np.gcd(sr_from, sr_to)
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)


def normalize_peak(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 0 else x


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """ピンクノイズに近い雑音を、発話部分の平均パワーに対して snr_db で足す。"""
    white = rng.standard_normal(x.size).astype(np.float32)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float32)
    f[0] = 1.0
    noise = np.fft.irfft(spec / np.sqrt(f), n=x.size).astype(np.float32)
    active = x[np.abs(x) > 0.01]
    p_sig = float(np.mean(active ** 2)) if active.size else float(np.mean(x ** 2) + 1e-12)
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    noise *= np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10.0)))
    return np.clip(x + noise, -1, 1).astype(np.float32)


def to_telephone(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """300〜3400Hz に絞って 8kHz にし、mu-law で量子化して戻す（音質の劣化を再現する）。"""
    sos = butter(6, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = resample(sosfilt(sos, x).astype(np.float32), sr, 8000)
    y = mulaw_decode(mulaw_encode(normalize_peak(y)))
    return y, 8000


def merge_intervals(iv: list[tuple[int, int]], gap_ms: int = 0) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1] + gap_ms:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def apply_mask(x: np.ndarray, sr: int, intervals: list[tuple[int, int]], mode: str = "beep") -> np.ndarray:
    """区間をピー音（1kHz）か無音に置き換える。端は5msでつなぐ。"""
    y = x.copy()
    fade = max(1, ms_to_samples(5, sr))
    for s_ms, e_ms in intervals:
        a, b = max(0, ms_to_samples(s_ms, sr)), min(x.size, ms_to_samples(e_ms, sr))
        if b <= a:
            continue
        if mode == "beep":
            t = np.arange(b - a) / sr
            fill = (0.25 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        else:
            fill = np.zeros(b - a, dtype=np.float32)
        env = np.ones(b - a, dtype=np.float32)
        n = min(fade, (b - a) // 2)
        if n > 0:
            env[:n] = np.linspace(0, 1, n)
            env[-n:] = np.linspace(1, 0, n)
        y[a:b] = fill * env
    return y
