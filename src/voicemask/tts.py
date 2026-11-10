"""音声合成（Google Cloud Chirp 3 HD・Azure AI Speech・オフライン確認用の疑似合成）。

合成結果は cache_dir に保存し、同じ文・同じ声・同じ話速なら API を呼ばない（再課金しない）。
APIキーはヘッダで送り、URL には入れない。
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import requests

from . import audio


@dataclass(frozen=True)
class Voice:
    engine: str
    name: str
    gender: str  # "F" / "M"


# Chirp 3 HD の声（公式の一覧。ja-JP で全部使えるかは1週目に確かめる）
CHIRP3_F = ["Achernar", "Aoede", "Autonoe", "Callirrhoe", "Despina", "Erinome", "Gacrux", "Kore",
            "Laomedeia", "Leda", "Pulcherrima", "Sulafat", "Vindemiatrix", "Zephyr"]
CHIRP3_M = ["Achird", "Algenib", "Algieba", "Alnilam", "Charon", "Enceladus", "Fenrir", "Iapetus",
            "Orus", "Puck", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Umbriel", "Zubenelgenubi"]
AZURE_F = ["ja-JP-NanamiNeural", "ja-JP-AoiNeural", "ja-JP-MayuNeural", "ja-JP-ShioriNeural"]
AZURE_M = ["ja-JP-KeitaNeural", "ja-JP-DaichiNeural", "ja-JP-NaokiNeural"]


class TTS:
    name = "base"
    voices: list[Voice] = []

    def __init__(self, cache_dir: str | Path = "data/tts_cache"):
        self.cache = Path(cache_dir)

    def synthesize(self, text: str, voice: Voice, rate: float = 1.0) -> tuple[np.ndarray, int]:
        key = hashlib.sha256(f"{voice.engine}|{voice.name}|{rate:.3f}|{text}".encode()).hexdigest()[:24]
        path = self.cache / voice.engine / f"{key}.wav"
        if path.exists():
            return audio.read_wav(path)
        wav = self._call(text, voice, rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav)
        return audio.read_wav(path)

    def _call(self, text: str, voice: Voice, rate: float) -> bytes:
        raise NotImplementedError


def _post(url: str, retries: int = 4, **kw) -> requests.Response:
    for i in range(retries + 1):
        try:
            r = requests.post(url, timeout=60, **kw)
        except requests.RequestException:
            if i == retries:
                raise
            time.sleep(2 ** i)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and i < retries:
            time.sleep(2 ** i)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"合成に失敗しました（HTTP {r.status_code}）: {r.text[:300]}")
        return r
    raise RuntimeError("合成に失敗しました")


class GoogleTTS(TTS):
    name = "google"
    voices = [Voice("google", f"ja-JP-Chirp3-HD-{n}", "F") for n in CHIRP3_F] + \
             [Voice("google", f"ja-JP-Chirp3-HD-{n}", "M") for n in CHIRP3_M]

    def _call(self, text: str, voice: Voice, rate: float) -> bytes:
        key = os.environ.get("GOOGLE_TTS_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_TTS_API_KEY を .env に入れてください")
        body = {
            "input": {"text": text},
            "voice": {"languageCode": "ja-JP", "name": voice.name},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000, "speakingRate": rate},
        }
        r = _post("https://texttospeech.googleapis.com/v1/text:synthesize", json=body,
                  headers={"X-Goog-Api-Key": key})
        return base64.b64decode(r.json()["audioContent"])


class AzureTTS(TTS):
    name = "azure"
    voices = [Voice("azure", n, "F") for n in AZURE_F] + [Voice("azure", n, "M") for n in AZURE_M]

    def _call(self, text: str, voice: Voice, rate: float) -> bytes:
        key = os.environ.get("AZURE_SPEECH_KEY", "")
        region = os.environ.get("AZURE_SPEECH_REGION", "japaneast")
        if not key:
            raise RuntimeError("AZURE_SPEECH_KEY を .env に入れてください")
        pct = f"{(rate - 1) * 100:+.0f}%"
        ssml = (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ja-JP">'
            f'<voice name="{voice.name}"><prosody rate="{pct}">{escape(text)}</prosody></voice></speak>'
        )
        r = _post(
            f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
            data=ssml.encode("utf-8"),
            headers={
                "Ocp-Apim-Subscription-Key": key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                "User-Agent": "voicemask",
            },
        )
        return r.content


class FakeTTS(TTS):
    """オフライン確認用。1文字ごとに母音らしい音を並べる（数値は本物の評価に使えない）。"""

    name = "fake"
    voices = [Voice("fake", "f1", "F"), Voice("fake", "f2", "F"), Voice("fake", "m1", "M"), Voice("fake", "m2", "M")]

    def _call(self, text: str, voice: Voice, rate: float) -> bytes:
        sr = 16000
        f0 = {"f1": 220, "f2": 250, "m1": 120, "m2": 140}[voice.name]
        chars = [c for c in text if not c.isspace()]
        dur = 0.11 / rate
        rng = np.random.default_rng(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
        out = [audio.silence(40, sr)]
        for c in chars:
            n = int(sr * (0.18 if c in "、。" else dur) / (1 if c not in "、。" else rate))
            if c in "、。":
                out.append(np.zeros(n, dtype=np.float32))
                continue
            t = np.arange(n) / sr
            f = f0 * (1 + 0.05 * rng.standard_normal())
            x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 5))
            env = np.sin(np.pi * np.arange(n) / n) ** 0.5
            out.append((0.3 * x * env).astype(np.float32))
        out.append(audio.silence(40, sr))
        tmp = Path(self.cache) / f"_tmp_{os.getpid()}.wav"
        audio.write_wav(tmp, np.concatenate(out), sr)
        data = tmp.read_bytes()
        tmp.unlink()
        return data


def get_engines(names: str, cache_dir: str | Path) -> list[TTS]:
    table = {"google": GoogleTTS, "azure": AzureTTS, "fake": FakeTTS}
    return [table[n.strip()](cache_dir) for n in names.split(",") if n.strip()]
