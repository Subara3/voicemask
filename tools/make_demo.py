"""見本のページを作る：伏せる前と後の音声、認識した文字と伏せた文字を並べる。

    python tools/make_demo.py --calls call0102,call0103,call0105 --out docs

評価用の通話から、種類ごとに番号のいちばん若いものを選ぶ（よく伏せられた通話を選んだわけではない）。
台本の人名・住所・番号はすべて架空。
"""

import argparse
import html
import json
import subprocess
from pathlib import Path

from voicemask import audio
from voicemask.asr import load_recognition
from voicemask.evaluate import overlap, trim_to_voice
from voicemask.mask import CONDITIONS, Detectors, compute_mask
from voicemask.models import PII_LABELS_JA
from voicemask.synth import load_truths

SCENARIO_JA = {"reservation": "予約の変更", "cancel": "解約", "address_change": "住所の変更", "card_lost": "カードの紛失",
               "delivery": "配送の問い合わせ", "member_update": "会員情報の変更"}

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="data/v1/truth.jsonl")
ap.add_argument("--rec", default="runs/v1_hybrid")
ap.add_argument("--calls", default="call0102,call0103,call0105")
ap.add_argument("--condition", default="C2")
ap.add_argument("--out", default="docs")
a = ap.parse_args()

out = Path(a.out)
(out / "audio").mkdir(parents=True, exist_ok=True)
cfg = CONDITIONS[a.condition]
det = Detectors(cfg.base)
base = Path(a.truth).parent
want = a.calls.split(",")
truths = {t.call_id: t for t in load_truths(a.truth, "clean") if t.call_id in want}


def mp3(x, sr, path: Path):
    tmp = path.with_suffix(".wav")
    audio.write_wav(tmp, x, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ac", "1", "-b:a", "48k", str(path)], check=True)
    tmp.unlink()


cards = []
for cid in want:
    t = trim_to_voice(truths[cid], base)
    rec = load_recognition(Path(a.rec) / f"{t.key}.json")
    res = compute_mask(rec, cfg, detectors=det)
    x, sr = audio.read_wav(base / t.audio_path)
    mp3(x, sr, out / "audio" / f"{cid}_before.mp3")
    mp3(audio.apply_mask(x, sr, res.merged, "beep"), sr, out / "audio" / f"{cid}_after.mp3")

    # 正解の区間ごとに、伏せた割合
    spans = []
    for sp in t.spans:
        cov = overlap(res.merged, sp.start_ms, sp.end_ms) / max(1, sp.dur)
        spans.append((PII_LABELS_JA.get(sp.pii_type, sp.pii_type), cov))
    ok = sum(1 for _, c in spans if c >= 0.9)

    rows, k = [], 0
    for seg in rec.segments:
        before, after, prev = [], [], None
        for tok in seg.tokens:
            w = html.escape(tok.written)
            if res.masked[k]:
                before.append(f'<mark>{w}</mark>')
                lab = "［伏］"
                if prev != lab:
                    after.append(f'<span class="m">{lab}</span>')
                prev = lab
            else:
                before.append(w)
                after.append(w)
                prev = None
            k += 1
        m, s = divmod(seg.start_ms // 1000, 60)
        rows.append(f'<tr><td class="t"><button data-t="{seg.start_ms / 1000:.2f}">{m:02d}:{s:02d}</button></td>'
                    f'<td>{"".join(before)}</td><td>{"".join(after)}</td></tr>')
    cards.append(f'''
<section class="call" id="{cid}">
  <h2>{cid}：{SCENARIO_JA.get(t.scenario, t.scenario)}</h2>
  <p class="meta">個人情報 {len(spans)}件のうち、音声の9割以上を伏せたもの {ok}件</p>
  <div class="players">
    <label>伏せる前<audio controls preload="none" src="audio/{cid}_before.mp3"></audio></label>
    <label>伏せた後<audio controls preload="none" src="audio/{cid}_after.mp3"></audio></label>
  </div>
  <table><thead><tr><th></th><th>認識した文字（伏せる語に色）</th><th>残す文字</th></tr></thead>
  <tbody>{"".join(rows)}</tbody></table>
</section>''')

page = f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>voicemask 見本</title>
<style>
:root {{ --bg:#fff; --fg:#222; --sub:#666; --line:#ddd; --mark:#ffe08a; --m:#555; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181b; --fg:#e6e6e6; --sub:#9a9a9a; --line:#333; --mark:#6b5a1e; --m:#bbb; }} }}
body {{ background:var(--bg); color:var(--fg); font-family:system-ui,"Hiragino Sans","Yu Gothic UI",sans-serif; margin:0; padding:24px 16px; line-height:1.7; }}
main {{ max-width:1000px; margin:0 auto; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }} h2 {{ font-size:1.1rem; margin:32px 0 4px; }}
.lead, .meta {{ color:var(--sub); font-size:.9rem; margin:0 0 8px; }}
.players {{ display:flex; flex-wrap:wrap; gap:12px 24px; margin:8px 0 12px; }}
.players label {{ display:flex; flex-direction:column; font-size:.85rem; color:var(--sub); }}
audio {{ width:min(420px, 92vw); }}
table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
th, td {{ border-top:1px solid var(--line); padding:6px 8px; vertical-align:top; text-align:left; }}
th {{ font-weight:600; color:var(--sub); font-size:.8rem; }}
td.t {{ width:4.5em; }} td.t button {{ font:inherit; font-size:.8rem; color:var(--sub); background:none; border:1px solid var(--line); border-radius:4px; cursor:pointer; padding:0 6px; }}
mark {{ background:var(--mark); color:inherit; }} .m {{ color:var(--m); font-size:.85em; }}
@media (max-width:640px) {{ thead {{ display:none; }} tr {{ display:block; border-top:1px solid var(--line); padding:4px 0; }} td {{ display:block; border:0; padding:2px 0; }} }}
</style></head>
<body><main>
<h1>voicemask 見本</h1>
<p class="lead">合成した通話（人名・住所・番号はすべて架空）を AmiVoice（会話_汎用、ログ保存なし）で認識し、個人情報の語を文字と音声の両方で伏せた例です。伏せた区間はピー音に置き換えています。時刻のボタンを押すと、両方の音声がその位置に移ります。</p>
{"".join(cards)}
</main>
<script>
document.querySelectorAll('section.call').forEach(sec => {{
  const players = sec.querySelectorAll('audio');
  sec.querySelectorAll('button[data-t]').forEach(b => b.addEventListener('click', () => {{
    players.forEach(p => {{ p.currentTime = parseFloat(b.dataset.t); }});
  }}));
}});
</script>
</body></html>'''
(out / "index.html").write_text(page, encoding="utf-8")
print(f"{out / 'index.html'} と音声 {len(want) * 2}本を書きました")
