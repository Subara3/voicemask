"""採点に使ったデータを、音声なしで公開できる形に書き出す。

    python tools/export_bench.py --out bench/v1

書き出すもの
- truth.jsonl：正解（声のある範囲に詰め済み）。認識結果のある版だけ
- scripts.jsonl：台本（人名・住所・番号はすべて架空）
- rec/<エンジン>_<dev|test>/：AmiVoice の認識結果（単語・時刻・confidence）
合成音声は書き出さない。
"""

import argparse
import json
import shutil
from pathlib import Path

from voicemask.evaluate import trim_to_voice
from voicemask.synth import load_truths

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="data/v1/truth.jsonl")
ap.add_argument("--scripts", default="data/v1/scripts.jsonl")
ap.add_argument("--out", default="bench/v1")
a = ap.parse_args()

RUNS = [("hybrid", "runs/v1_hybrid"), ("e2e", "runs/v1_e2e"), ("hybrid_crm", "runs/v1_hybrid_D"),
        ("stream", "runs/v1_stream_test2"), ("stream", "runs/v1_stream_dev")]  # stream は call0200〜0299 と 0000〜0039
out = Path(a.out)
base = Path(a.truth).parent
have, n = set(), {}
for name, src in RUNS:
    for p in sorted(Path(src).glob("call*__*.json")):
        num = int(p.name[4:8])
        d = out / "rec" / f"{name}_{'dev' if num < 100 else 'test'}"
        d.mkdir(parents=True, exist_ok=True)
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["audio_path"] = Path(rec["audio_path"]).name  # 手元のフォルダ名は残さない
        (d / p.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        have.add(p.stem)
        n[d.name] = n.get(d.name, 0) + 1

cache, lines = {}, []
for t in load_truths(a.truth):
    if t.key in have:
        lines.append(json.dumps(trim_to_voice(t, base, cache).to_dict(), ensure_ascii=False))
(out / "truth.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
shutil.copy(a.scripts, out / "scripts.jsonl")
print(f"正解 {len(lines)} 件", json.dumps(n, ensure_ascii=False))
