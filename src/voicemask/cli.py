"""コマンド。`voicemask <コマンド> --help` で引数が出る。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except ImportError:
        pass


def _rec_path(rec_dir: str | Path, key: str) -> Path:
    return Path(rec_dir) / f"{key}.json"


def _pairs(truth_path: str, rec_dir: str, variant: str | None, limit: int | None = None):
    from .asr import load_recognition
    from .synth import load_truths

    from .evaluate import trim_to_voice

    out, cache, base = [], {}, Path(truth_path).parent
    for t in load_truths(truth_path, variant):
        p = _rec_path(rec_dir, t.key)
        if p.exists():
            out.append((trim_to_voice(t, base, cache), load_recognition(p)))
        if limit and len(out) >= limit:
            break
    if not out:
        sys.exit(f"{rec_dir} に認識結果がありません（variant={variant}）")
    return out


def _load_crm(scripts_path: str) -> dict[str, list[tuple[str, str]]]:
    """顧客台帳の代わり。着信番号で顧客を引けたとして、その顧客の氏名（表記, 読み）を通話ごとに返す。"""
    from .synth import load_scripts

    out: dict[str, list[tuple[str, str]]] = {}
    for sc in load_scripts(scripts_path):
        parts = [p for u in sc.utterances for p in u.parts if p.pii_type == "NAME"]
        full = next((p for p in parts if p.slot == "NAME"), None)
        if not full:
            continue
        last_r, _, first_r = full.reading.partition("、")
        last = next((p.text for p in parts if p.slot == "LAST"), "")
        words = [(full.text, last_r + first_r)]
        if last and full.text.startswith(last):
            words += [(last, last_r), (full.text[len(last):], first_r)]  # 姓だけ・名だけで呼ぶ場面もある
        out[sc.call_id] = words
    return out


def _crm_profile_words(words: list[tuple[str, str]] | None, engine: str) -> str | None:
    """顧客台帳の氏名を AmiVoice の単語登録（profileWords）にする。ハイブリッドは「表記 読み」、E2E は強調。"""
    if not words:
        return None
    from amivoice import ProfileWord, build_profile_words

    if engine.startswith("e2e") or "-a2-" in engine:
        pws = [ProfileWord(written=w, biasing=1.0) for w, _ in words]
    else:
        pws = [ProfileWord(written=w, spoken=r) for w, r in words]
    return build_profile_words(pws)


def _profile_words(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8").strip() if path else None


# ---------------------------------------------------------------- コマンド


def cmd_gen_scripts(a):
    from .scenarios import generate_scripts
    from .synth import save_scripts

    s = generate_scripts(a.n, a.seed)
    save_scripts(s, a.out)
    print(f"{len(s)} 本の台本を {a.out} に書きました")


def cmd_synth(a):
    from .synth import load_scripts, synth_dataset
    from .tts import get_engines

    engines = get_engines(a.tts, Path(a.out) / "tts_cache")
    p = synth_dataset(load_scripts(a.scripts), engines, a.out, a.variants.split(","), a.seed)
    print(f"正解を {p} に書きました")


def cmd_recognize(a):
    from .asr import recognize_file, save_recognition
    from .synth import load_truths

    truths = load_truths(a.truth, a.variant)[a.start:][: a.limit or None]
    base = Path(a.truth).parent
    crm = _load_crm(a.crm_scripts) if a.crm_scripts else None
    for i, t in enumerate(truths):
        out = _rec_path(a.out, t.key)
        if out.exists():
            continue  # 認識済みは送らない（再課金しない）
        pw = _crm_profile_words(crm.get(t.call_id), a.engine) if crm else _profile_words(a.profile_words)
        rec = recognize_file(base / t.audio_path, a.engine, max_chunk_s=a.max_chunk_s,
                             profile_words=pw, profile_id=a.profile_id)
        save_recognition(rec, out)
        print(f"\r認識 {i + 1}/{len(truths)} {t.key}", end="", flush=True)
    print()


def cmd_fake_recognize(a):
    from .asr import save_recognition
    from .fakeasr import fake_recognize
    from .synth import load_truths

    for t in load_truths(a.truth):
        save_recognition(fake_recognize(t, a.seed), _rec_path(a.out, t.key))
    print(f"疑似の認識結果を {a.out} に書きました（数値は本物ではありません）")


def cmd_eval(a):
    from .evaluate import compare_conditions, plot_sweep, sweep, table_md, write_json
    from .mask import Detectors, MaskConfig
    from .evaluate import evaluate

    pairs = _pairs(a.truth, a.rec, a.variant)
    det = Detectors(("rules", "ner") + (("llm",) if a.llm else ()))
    rows = compare_conditions(pairs, det, a.threshold, a.pad_ms, a.pad_before_ms)
    if a.llm:
        cfg = MaskConfig(base=("rules", "ner", "llm"), context="lowconf", threshold=a.threshold, pad_ms=a.pad_ms,
                         pad_before_ms=a.pad_before_ms)
        rows["C+Claude（比較用）"] = evaluate(pairs, cfg, {}, det)[0]
    out = Path(a.out)
    md = [f"# 条件ごとの結果（{a.rec}, variant={a.variant}, 閾値={a.threshold}, 余白={a.pad_before_ms if a.pad_before_ms is not None else a.pad_ms}/{a.pad_ms}ms）", "", table_md(rows), ""]
    _, by_type = evaluate(pairs, MaskConfig(context="lowconf", threshold=a.threshold, pad_ms=a.pad_ms,
                                            pad_before_ms=a.pad_before_ms), {}, det)
    md += ["## 条件Cの種類別", "", table_md(by_type, "種類", leaks_only=True), ""]
    write_json({"conditions": rows, "by_type_C": by_type}, out / "summary.json")
    if a.sweep:
        for ctx in ("lowconf",):
            sw = sweep(pairs, det, ctx, pad_ms=a.pad_ms, pad_before_ms=a.pad_before_ms)
            write_json(sw, out / f"sweep_{ctx}.json")
            if a.plot:
                plot_sweep(sw, out / f"sweep_{ctx}.png", f"閾値と漏れ・消しすぎ（{Path(a.rec).name}）")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def cmd_timing(a):
    from .evaluate import timing_errors

    print(json.dumps(timing_errors(_pairs(a.truth, a.rec, a.variant)), ensure_ascii=False, indent=1))


def cmd_mask(a):
    from .mask import CONDITIONS, Detectors, process, with_threshold

    cfg = with_threshold(CONDITIONS[a.condition], a.threshold)
    det = Detectors(cfg.base)
    base = Path(a.truth).parent
    for t, rec in _pairs(a.truth, a.rec, a.variant, a.limit):
        process(rec, base / t.audio_path, a.out, cfg, det)
    print(f"伏せた音声・文字・監査ログを {a.out} に書きました")


def cmd_recheck(a):
    """伏せた音声をもう一度認識し、個人情報が文字として出てこないかを数える。"""
    from .asr import load_recognition, recognize_file, save_recognition
    from .evaluate import write_json
    from .synth import load_truths
    from .textnorm import digits_only, normalize_for_match

    rows, found, total = [], 0, 0
    for t in load_truths(a.truth, a.variant)[: a.limit or None]:
        wav = Path(a.masked) / f"{t.key}.masked.wav"
        if not wav.exists():
            continue
        cache = Path(a.masked) / "recheck" / f"{t.key}.json"
        rec = load_recognition(cache) if cache.exists() else recognize_file(wav, a.engine, max_chunk_s=a.max_chunk_s)
        save_recognition(rec, cache)
        text = normalize_for_match(rec.text)
        digits = digits_only(rec.text)
        for sp in t.spans:
            total += 1
            v = normalize_for_match(sp.text)
            d = digits_only(sp.text)
            hit = (len(d) >= 4 and d in digits) or (not d and v and v in text)
            if hit:
                found += 1
                rows.append({"call": t.key, "type": sp.pii_type})
    res = {"件数": total, "再認識で出てきた": found, "割合": found / total if total else 0, "内訳": rows}
    write_json(res, a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "内訳"}, ensure_ascii=False))


def cmd_stream(a):
    from .asr import stream_file
    from .synth import load_truths

    base = Path(a.truth).parent
    for t in load_truths(a.truth, a.variant)[a.start:][: a.limit or None]:
        out = Path(a.out) / f"{t.key}.json"
        if out.exists():
            continue
        log = None
        for attempt in range(3):  # 接続の始めに切られることがある（11/10に1回）。送り直す
            try:
                log = stream_file(base / t.audio_path, a.engine, a.chunk_ms, profile_words=_profile_words(a.profile_words),
                                  logging_opt_out=not a.with_log)
                break
            except Exception as e:  # noqa: BLE001
                print(f"{t.key}: {attempt + 1}回目失敗（{type(e).__name__}: {e}）", flush=True)
                time.sleep(10)
        if log is None:
            continue
        log.save(out)
        n = sum(1 for e in log.events if e["type"] == "final")
        print(f"{t.key}: 確定結果 {n} 件、送り終わり {log.sent_done_ms:.0f}ms", flush=True)


def cmd_stream_eval(a):
    from .asr import StreamLog, load_recognition
    from .evaluate import write_json
    from .mask import CONDITIONS, Detectors, with_threshold
    from .stream_eval import decision_lags, partial_arrivals, segment_arrivals, simulate_arrivals, stream_leaks
    from .synth import load_truths

    from .evaluate import trim_to_voice

    items, cache = [], {}
    base = Path(a.truth).parent
    for t in load_truths(a.truth, a.variant):
        p = Path(a.rec) / f"{t.key}.json"
        if not p.exists():
            continue
        t = trim_to_voice(t, base, cache)
        if a.simulate:
            rec = load_recognition(p)
            items.append((t, rec, simulate_arrivals(rec)))
        else:
            log = StreamLog.load(p)
            rec = log.to_recognition()
            items.append((t, rec, partial_arrivals(log, rec) if a.partial else segment_arrivals(log)))
    cfg = with_threshold(CONDITIONS[a.condition], a.threshold)
    rows = stream_leaks(items, cfg, Detectors(cfg.base), [int(x) for x in a.delays.split(",")], use_partial=a.partial)
    res = {"確定までの遅れ": decision_lags(items), "ためる時間ごと": rows}
    write_json(res, Path(a.out) / "stream.json")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_live(a):
    """録音ファイルを通話に見立てて実時間で流し、遅れて出てくる音声を伏せる。"""
    from . import audio
    from .live import run_live
    from .mask import CONDITIONS, Detectors

    cfg = CONDITIONS[a.condition]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.audio).stem
    print(f"{a.delay_ms}ms ためて流します（ログ保存{'あり' if a.with_log else 'なし'}の接続先）", flush=True)
    y, sr, log, masker, emitted = run_live(a.audio, a.delay_ms, cfg, Detectors(cfg.base), a.engine, a.with_log,
                                           on_line=lambda ln: print(ln, flush=True), ahead=a.ahead)
    audio.write_wav(out / f"{stem}.live{a.delay_ms}.wav", y, sr)
    log.save(out / f"{stem}.stream.json")
    write = {"delay_ms": a.delay_ms, "emitted": emitted,
             "snapshots": [{"at_ms": h.at_ms, "intervals": h.intervals} for h in masker.history]}
    (out / f"{stem}.live{a.delay_ms}.json").write_text(json.dumps(write, ensure_ascii=False), encoding="utf-8")
    print(f"ピー音にした長さ {sum(e - s for s, e in emitted) / 1000:.1f}秒、書き出し先 {out}", flush=True)


def cmd_live_eval(a):
    """保存したイベントを届いた時刻どおりに流し直し、リアルタイムで伏せたときの漏れを測る。"""
    from .asr import StreamLog
    from .evaluate import trim_to_voice, write_json
    from .live import score_live
    from .mask import CONDITIONS, Detectors
    from .synth import load_truths

    items, cache, base = [], {}, Path(a.truth).parent
    for t in load_truths(a.truth, a.variant):
        p = Path(a.rec) / f"{t.key}.json"
        if p.exists():
            items.append((trim_to_voice(t, base, cache), StreamLog.load(p)))
    cfg = CONDITIONS[a.condition]
    rows = score_live(items, cfg, Detectors(cfg.base), [int(x) for x in a.delays.split(",")], ahead=a.ahead)
    write_json({"本数": len(items), "先回り": a.ahead, "ためる時間ごと": rows}, Path(a.out) / "live.json")
    for r in rows:
        print(f"{r['ためる時間ms']:>6}ms 区間漏れ {r['区間漏れ率']:.1%} 消しすぎ {r['消しすぎ率']:.1%}（{r['件数']}件）", flush=True)


def cmd_llm_export(a):
    """Claude Code に答えてもらう問いを書き出す（比較用の検出）。"""
    from .detect.llm import export_question
    from .detect.tokenview import TokenView

    n = 0
    for _, rec in _pairs(a.truth, a.rec, a.variant, a.limit):
        export_question(TokenView(rec), a.root)
        n += 1
    print(f"{n} 件の問いを {a.root}/ask に書きました。答えは {a.root}/answer/<同じ名前>.json に置く")


def cmd_profile_words(a):
    """CSV（表記,読み）から profileWords の文字列を作る。E2E は単語強調（表記  強調度）にする。"""
    from amivoice import ProfileWord, build_profile_words

    words = []
    with open(a.csv, encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if a.e2e:
                words.append(ProfileWord(written=row[0].strip(), biasing=a.biasing))
            else:
                words.append(ProfileWord(written=row[0].strip(), spoken=(row[1].strip() if len(row) > 1 else row[0].strip())))
    Path(a.out).write_text(build_profile_words(words), encoding="utf-8")
    print(f"{len(words)} 語を {a.out} に書きました")


def cmd_run(a):
    """録音ファイル1本を、認識→伏せる→（任意で）伏せた文字だけ LLM で要約、まで通す。"""
    from .asr import recognize_file, save_recognition
    from .mask import CONDITIONS, process, with_threshold

    src = Path(a.audio)
    out = Path(a.out)
    rec = recognize_file(src, a.engine, max_chunk_s=a.max_chunk_s, profile_words=_profile_words(a.profile_words))
    cfg = with_threshold(CONDITIONS[a.condition], a.threshold)
    res = process(rec, src, out, cfg, delete_source=a.delete_source, llm_after=a.llm_after)
    if a.keep_raw:
        save_recognition(rec, out / f"{src.stem}.raw.json")  # 伏せる前の文字。確認用で、本番では残さない
    print(res.text)
    if a.summarize:
        from .summarize import summarize

        s = summarize(res.text)
        (out / f"{src.stem}.summary.txt").write_text(s, encoding="utf-8")
        print("\n--- 伏せた文字だけから作った対応記録 ---\n" + s)


# ---------------------------------------------------------------- 引数


def main(argv: list[str] | None = None) -> None:
    _load_env()
    p = argparse.ArgumentParser(prog="voicemask")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    s = add("gen-scripts", cmd_gen_scripts, "台本を作る")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", required=True)

    s = add("synth", cmd_synth, "台本から音声と正解を作る")
    s.add_argument("--scripts", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tts", default="fake", help="fake / google,azure")
    s.add_argument("--variants", default="clean,tel")
    s.add_argument("--seed", type=int, default=1)

    for name, fn, h in (("recognize", cmd_recognize, "同期HTTPで認識する"), ("recheck", cmd_recheck, "伏せた音声をもう一度認識する")):
        s = add(name, fn, h)
        s.add_argument("--truth", required=True)
        s.add_argument("--engine", default="hybrid", help="hybrid / e2e / エンジン名")
        s.add_argument("--variant", default="clean")
        s.add_argument("--limit", type=int, default=0)
        s.add_argument("--start", type=int, default=0, help="先頭から何本飛ばすか（評価用だけ送るとき 100）")
        s.add_argument("--max-chunk-s", type=float, default=None, help="この秒数以下に区切って送る（E2E は 18 を推奨）")
        if name == "recognize":
            s.add_argument("--out", required=True)
            s.add_argument("--profile-words", default=None)
            s.add_argument("--profile-id", default=None, help=":dic1:dic2 の形")
            s.add_argument("--crm-scripts", default=None, help="台本を顧客台帳の代わりにし、その通話の顧客の氏名を単語登録する（条件D）")
        else:
            s.add_argument("--masked", required=True)
            s.add_argument("--out", required=True)

    s = add("fake-recognize", cmd_fake_recognize, "疑似の認識結果を作る（オフライン確認用）")
    s.add_argument("--truth", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)

    s = add("eval", cmd_eval, "条件ごとの表と閾値の曲線")
    s.add_argument("--truth", required=True)
    s.add_argument("--rec", required=True)
    s.add_argument("--variant", default="clean")
    s.add_argument("--out", required=True)
    s.add_argument("--threshold", type=float, default=0.6)
    s.add_argument("--pad-ms", type=int, default=120)
    s.add_argument("--pad-before-ms", type=int, default=None, help="前の余白（E2E は 500 前後）")
    s.add_argument("--sweep", action="store_true")
    s.add_argument("--plot", action="store_true")
    s.add_argument("--llm", action="store_true", help="Claude の検出を比較用に足す（伏せる前の文字を送る。上限10ドル）")

    s = add("timing", cmd_timing, "単語の時刻のずれ")
    s.add_argument("--truth", required=True)
    s.add_argument("--rec", required=True)
    s.add_argument("--variant", default="clean")

    s = add("mask", cmd_mask, "伏せた音声・文字・監査ログを書き出す")
    s.add_argument("--truth", required=True)
    s.add_argument("--rec", required=True)
    s.add_argument("--variant", default="clean")
    s.add_argument("--out", required=True)
    s.add_argument("--condition", default="C")
    s.add_argument("--threshold", type=float, default=0.6)
    s.add_argument("--limit", type=int, default=0)

    s = add("stream", cmd_stream, "WebSocket で実時間に流し、イベントの到着時刻を残す")
    s.add_argument("--truth", required=True)
    s.add_argument("--engine", default="hybrid")
    s.add_argument("--variant", default="clean")
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=1)
    s.add_argument("--chunk-ms", type=int, default=100)
    s.add_argument("--profile-words", default=None)
    s.add_argument("--start", type=int, default=0, help="何本目から流すか（100 なら call0100 から）")
    s.add_argument("--with-log", action="store_true", help="ログ保存ありの接続先を使う（架空の台本だけに使う）")

    s = add("stream-eval", cmd_stream_eval, "ためる時間ごとの漏れ率")
    s.add_argument("--truth", required=True)
    s.add_argument("--rec", required=True)
    s.add_argument("--variant", default="clean")
    s.add_argument("--out", required=True)
    s.add_argument("--condition", default="C")
    s.add_argument("--threshold", type=float, default=0.6)
    s.add_argument("--delays", default="0,250,500,1000,1500,2000,3000,5000")
    s.add_argument("--simulate", action="store_true", help="同期HTTPの結果から到着時刻を仮に作る（オフライン確認用）")
    s.add_argument("--partial", action="store_true", help="途中結果（U）に文字が出た時点で判断する")

    s = add("live", cmd_live, "通話中にリアルタイムで伏せる（録音ファイルを実時間で流して試す）")
    s.add_argument("audio")
    s.add_argument("--delay-ms", type=int, default=1500)
    s.add_argument("--out", default="masked/live")
    s.add_argument("--engine", default="hybrid")
    s.add_argument("--condition", default="C2")
    s.add_argument("--with-log", action="store_true", help="ログ保存ありの接続先を使う（架空の台本だけに使う）")
    s.add_argument("--ahead", action="store_true", help="質問のあとの発話を、文字が出る前から伏せておく")

    s = add("live-eval", cmd_live_eval, "リアルタイムで伏せる処理を、保存したイベントで流し直して測る")
    s.add_argument("--truth", required=True)
    s.add_argument("--rec", required=True)
    s.add_argument("--variant", default="clean")
    s.add_argument("--out", required=True)
    s.add_argument("--condition", default="C2")
    s.add_argument("--delays", default="0,500,1000,1500,2000,3000,5000")
    s.add_argument("--ahead", action="store_true", help="質問のあとの発話を、文字が出る前から伏せておく")

    s = add("llm-export", cmd_llm_export, "Claude Code に答えてもらう問いを書き出す（比較用）")
    s.add_argument("--truth", required=True)
    s.add_argument("--rec", required=True)
    s.add_argument("--variant", default="clean")
    s.add_argument("--root", default="data/llm")
    s.add_argument("--limit", type=int, default=0)

    s = add("profile-words", cmd_profile_words, "CSV から単語登録の文字列を作る")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--e2e", action="store_true")
    s.add_argument("--biasing", type=float, default=0.5)

    s = add("run", cmd_run, "録音ファイル1本を伏せる（デモ）")
    s.add_argument("audio")
    s.add_argument("--out", default="masked/demo")
    s.add_argument("--engine", default="hybrid")
    s.add_argument("--condition", default="C2")
    s.add_argument("--threshold", type=float, default=0.6)
    s.add_argument("--max-chunk-s", type=float, default=None)
    s.add_argument("--profile-words", default=None)
    s.add_argument("--llm-after", action="store_true", help="手元で伏せたあとの文字だけを Claude に見せ、取りこぼしを足す")
    s.add_argument("--summarize", action="store_true", help="伏せた文字だけを Claude に渡して対応記録を作る")
    s.add_argument("--delete-source", action="store_true", help="伏せたあと元の音声を消す")
    s.add_argument("--keep-raw", action="store_true", help="伏せる前の認識結果も残す（確認用）")

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
