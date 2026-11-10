"""リアルタイム版の動画を作る：通話の音声と、D秒遅れで外に流す音声（ピー音入り）、伏せた文字を並べる。

    python tools/make_live_video.py --live masked/live_demo --stem call0102__clean --delay-ms 2000 --out docs/live_demo.mp4

`voicemask live` の書き出し（*.stream.json と *.live<D>.wav）を使う。画面の文字は、記録を届いた時刻どおりに
流し直して、その時刻までに分かっていたものだけを出す。音は、外に流す音声（D秒遅れ）。台本の人名・住所・番号はすべて架空。
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from voicemask import audio
from voicemask.asr import StreamLog
from voicemask.live import LiveMasker, feed
from voicemask.mask import CONDITIONS, Detectors

ap = argparse.ArgumentParser()
ap.add_argument("--live", default="masked/live_demo")
ap.add_argument("--stem", default="call0102__clean")
ap.add_argument("--src", default=None, help="元の音声（既定は記録に残っている場所）")
ap.add_argument("--delay-ms", type=int, default=2000)
ap.add_argument("--out", default="docs/live_demo.mp4")
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--ffmpeg", default="ffmpeg")
a = ap.parse_args()

W, H = 1280, 720
BG, FG, SUB, LINE = (255, 255, 255), (34, 34, 34), (110, 110, 110), (220, 220, 220)
WAVE, KNOWN, BEEP, BEEP_FILL = (90, 90, 90), (255, 224, 138), (200, 70, 55), (246, 200, 192)
FONT = "C:/Windows/Fonts/YuGothM.ttc"
f_title, f_lab, f_txt, f_small = (ImageFont.truetype(FONT, s) for s in (30, 20, 26, 17))

live = Path(a.live)
log = StreamLog.load(live / f"{a.stem}.stream.json")
x, sr = audio.read_wav(a.src or log.audio_path)
y, _ = audio.read_wav(live / f"{a.stem}.live{a.delay_ms}.wav")
emitted = json.loads((live / f"{a.stem}.live{a.delay_ms}.json").read_text(encoding="utf-8"))["emitted"]
cfg = CONDITIONS["C2"]
hist = feed(LiveMasker(cfg, Detectors(cfg.base), log.engine, log.audio_path), log.events).history
dur = len(x) * 1000 // sr
D = a.delay_ms
total = dur + D + 1500

# 波形：10ms ごとの振幅
hop = sr // 100
env_x = np.abs(x[: len(x) // hop * hop]).reshape(-1, hop).max(axis=1)
env_y = np.abs(y[: len(y) // hop * hop]).reshape(-1, hop).max(axis=1)
peak = max(env_x.max(), 1e-6)

SPAN_BEFORE, SPAN_AFTER = 9000, 3000  # 画面に出す時間の幅（いまの位置の前後）
X0, X1 = 60, W - 60


def tx(ms, center):
    return X0 + (ms - (center - SPAN_BEFORE)) * (X1 - X0) / (SPAN_BEFORE + SPAN_AFTER)


def strip(d, top, h, env, center, play_ms, label, masks, mask_color, note):
    """まだ聞こえていない先の音声は描かない（本物の通話では先は分からない）。"""
    d.text((X0, top - 30), label, font=f_lab, fill=FG)
    d.text((X1, top - 30), note, font=f_small, fill=SUB, anchor="ra")
    d.rectangle([X0, top, X1, top + h], outline=LINE)
    lo, hi = center - SPAN_BEFORE, center + SPAN_AFTER
    for s, e in masks:
        if e > lo and s < hi:
            d.rectangle([max(X0, tx(s, center)), top + 1, min(X1, tx(e, center)), top + h - 1], fill=mask_color)
    mid = top + h // 2
    for px in range(X0, X1, 2):
        ms = lo + (px - X0) * (hi - lo) / (X1 - X0)
        i = int(ms // 10)
        if 0 <= i < len(env) and ms <= play_ms:
            amp = env[i] / peak * (h / 2 - 4)
            d.line([px, mid - amp, px, mid + amp], fill=WAVE)
    px = tx(play_ms, center)
    if X0 <= px <= X1:
        d.line([px, top - 4, px, top + h + 4], fill=FG, width=2)


def wrap(ln, width):
    """［…］の印は割らずに、欄の幅で折り返す。"""
    pieces = [p for p in re.split(r"(［[^］]*］)", ln) if p]
    toks = [p if p.startswith("［") else c for p in pieces for c in ([p] if p.startswith("［") else list(p))]
    rows, cur, w = [], [], 0
    for t in toks:
        tw = f_txt.getlength(t)
        if cur and w + tw > width:
            rows.append(cur)
            cur, w = [], 0
        cur.append(t)
        w += tw
    if cur:
        rows.append(cur)
    return rows


def text_lines(d, top, lines):
    d.text((X0, top - 34), "AmiVoiceの認識結果から伏せた文字（その時点までに分かった分）", font=f_lab, fill=FG)
    d.text((X1, top - 30), "［要確認］：質問への答えとして伏せた語", font=f_small, fill=SUB, anchor="ra")
    bottom = H - 50
    d.rectangle([X0, top, X1, bottom], outline=LINE)
    rows = [r for ln in lines for r in wrap(ln, X1 - X0 - 32)]
    fit = (bottom - top - 12) // 42
    yy = top + 12
    for row in rows[-fit:]:
        xx = X0 + 16
        for part in row:
            w = d.textlength(part, font=f_txt)
            if part.startswith("［"):
                d.rounded_rectangle([xx - 2, yy - 2, xx + w + 2, yy + 32], radius=4, fill=(253, 236, 233))
                d.text((xx, yy), part, font=f_txt, fill=BEEP)
            else:
                d.text((xx, yy), part, font=f_txt, fill=FG)
            xx += w
        yy += 42


def frame(t):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((X0, 22), f"通話中にリアルタイムで伏せる（AmiVoice WebSocket、{D / 1000:g}秒遅れで流す）", font=f_title, fill=FG)
    k = max([i for i, h in enumerate(hist) if h.at_ms <= t], default=-1)
    known = hist[k].intervals if k >= 0 else []
    lines = hist[k].lines if k >= 0 else []
    center = t
    strip(d, 110, 90, env_x, center, min(t, dur), "通話の音声（いま話している位置）", known, KNOWN, "黄色：伏せると決めた区間")
    out_ms = min(t - D, dur)
    beeped = [(s_, min(e_, out_ms)) for s_, e_ in emitted if s_ < out_ms]  # 流し終えたところまでだけ塗る
    strip(d, 260, 90, env_y, center, out_ms, f"外に流す音声（{D / 1000:g}秒遅れ・ピー音入り）", beeped, BEEP_FILL, "赤：ピー音にした区間")
    text_lines(d, 420, lines)
    d.text((X0, H - 38), "人名・住所・番号はすべて架空です（合成音声）。github.com/Subara3/voicemask", font=f_small, fill=SUB)
    d.text((X1, H - 38), f"{max(0, t) / 1000:5.1f}秒", font=f_small, fill=SUB, anchor="ra")
    return img


out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
# 音：外に流す音声を D ミリ秒遅らせる
pad = np.zeros(D * sr // 1000, dtype=y.dtype)
yy = np.concatenate([pad, y, np.zeros(sr * 2, dtype=y.dtype)])[: total * sr // 1000]
wav = out.with_suffix(".audio.wav")
audio.write_wav(wav, yy, sr)
cmd = [a.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps),
       "-i", "-", "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-c:a", "aac", "-b:a", "96k",
       "-shortest", str(out)]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
n = total * a.fps // 1000
for i in range(n):
    p.stdin.write(frame(i * 1000 / a.fps).tobytes())
p.stdin.close()
p.wait()
wav.unlink()
print(f"{out}（{n}コマ、{total / 1000:.1f}秒）")
