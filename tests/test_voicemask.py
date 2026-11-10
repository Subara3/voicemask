import json
import random

import numpy as np
import pytest

from voicemask import audio
from voicemask.asr import make_settings, results_to_recognition, split_by_gaps, split_points
from voicemask.detect.context import detect_lowconf
from voicemask.detect.rules import detect_rules
from voicemask.detect.tokenview import TokenView
from voicemask.evaluate import evaluate
from voicemask.fakeasr import fake_recognize
from voicemask.mask import CONDITIONS, MaskConfig, compute_mask
from voicemask.models import Recognition, Segment, Token
from voicemask.scenarios import SCENARIOS, generate_scripts, int_reading
from voicemask.stream_eval import effective_intervals
from voicemask.synth import synth_dataset, load_truths
from voicemask.textnorm import digits_to_reading, kana_digits_to_arabic, norm_token
from voicemask.tts import FakeTTS


def rec_of(words, gap_after=()):
    """(表記, confidence) の並びから、100ms ごとの単語でできた認識結果を作る。"""
    toks, t = [], 0
    for i, (w, c) in enumerate(words):
        toks.append(Token(w, t, t + 100, c))
        t += 100 + (800 if i in gap_after else 0)
    return Recognition("x.wav", "test", split_by_gaps(toks, 600))


# ---------------------------------------------------------------- 文字と数字


def test_kana_digits():
    assert kana_digits_to_arabic("ぜろきゅうぜろ") == "090"
    assert kana_digits_to_arabic("ゼロキュウゼロ") == "090"
    assert kana_digits_to_arabic("まる") is None
    assert kana_digits_to_arabic("さとう") is None
    assert norm_token("〇九〇") == "090"
    assert norm_token("０９０") == "090"


def test_readings():
    assert digits_to_reading("090-12") == "ぜろきゅうぜろの、いちに"
    assert int_reading(1985) == "せんきゅうひゃくはちじゅうご"
    assert int_reading(300) == "さんびゃく"
    assert int_reading(2004) == "にせんよん"


# ---------------------------------------------------------------- 音声


def test_mulaw_roundtrip(tmp_path):
    x = (0.5 * np.sin(np.linspace(0, 200, 8000))).astype(np.float32)
    y = audio.mulaw_decode(audio.mulaw_encode(x))
    assert np.max(np.abs(x - y)) < 0.02
    audio.write_wav(tmp_path / "t.wav", x, 8000, mulaw=True)
    assert audio.wav_is_mulaw(tmp_path / "t.wav")
    z, sr = audio.read_wav(tmp_path / "t.wav")
    assert sr == 8000 and z.size == x.size and np.max(np.abs(z - x)) < 0.02


def test_apply_mask_only_inside():
    x = np.ones(16000, dtype=np.float32) * 0.5
    y = audio.apply_mask(x, 16000, [(250, 500)], "silence")
    assert np.all(y[: 16000 * 250 // 1000] == 0.5)
    assert np.all(y[4100:7900] == 0)
    assert np.all(y[8100:] == 0.5)


def test_split_points_cuts_in_silence():
    sr = 16000
    tone = (0.5 * np.sin(np.arange(sr * 8) * 0.3)).astype(np.float32)
    x = np.concatenate([tone, np.zeros(sr), tone, np.zeros(sr), tone])
    chunks = split_points(x, sr, 18)
    assert len(chunks) == 2
    cut = chunks[0][1] / sr
    assert 8.0 < cut < 9.0 or 17.0 < cut < 18.0


# ---------------------------------------------------------------- 台本と合成


def test_scripts_cover_all_scenarios():
    s = generate_scripts(12, seed=3)
    assert {x.scenario for x in s} == set(SCENARIOS)
    for sc in s:
        assert not any("{" in p.text for u in sc.utterances for p in u.parts)


def test_synth_truth_matches_parts(tmp_path):
    scripts = generate_scripts(2, seed=5)
    p = synth_dataset(scripts, [FakeTTS(tmp_path / "c")], tmp_path, ["clean", "tel"], seed=1)
    truths = load_truths(p)
    assert len(truths) == 4
    t = truths[0]
    assert t.spans and all(s.end_ms > s.start_ms for s in t.spans)
    tel = [x for x in truths if x.variant == "tel"][0]
    assert tel.sample_rate == 8000 and audio.wav_is_mulaw(tmp_path / tel.audio_path)
    # 正解の区間は部品の時刻と一致する
    pii_parts = [x for x in t.parts if x["pii_type"]]
    assert [(x["start_ms"], x["end_ms"]) for x in pii_parts] == [(s.start_ms, s.end_ms) for s in t.spans]


# ---------------------------------------------------------------- AmiVoice の応答


SAMPLE = {
    "results": [{
        "tokens": [
            {"written": "電話番号", "confidence": 0.99, "starttime": 500, "endtime": 900, "spoken": "でんわばんごう"},
            {"written": "は", "confidence": 0.98, "starttime": 900, "endtime": 1000, "spoken": "は"},
            {"written": "090", "confidence": 0.91, "starttime": 1000, "endtime": 1500, "spoken": "ぜろきゅうぜろ"},
            {"written": "1234", "confidence": 0.42, "starttime": 1500, "endtime": 2100, "spoken": "いちにさんよん"},
            {"written": "5678", "confidence": 0.95, "starttime": 2100, "endtime": 2700, "spoken": "ごろくななはち"},
            {"written": "です", "confidence": 0.99, "starttime": 2700, "endtime": 2900, "spoken": "です"},
            {"written": "。", "confidence": 0.9, "spoken": "_"},
            {"written": "ありがとう", "confidence": 0.99, "starttime": 4000, "endtime": 4500, "spoken": "ありがとう"},
        ],
        "confidence": 0.9, "starttime": 400, "endtime": 4600, "tags": [], "rulename": "",
        "text": "電話番号は09012345678です。ありがとう",
    }],
    "utteranceid": "x", "text": "電話番号は09012345678です。ありがとう", "code": "", "message": "",
}


def test_parse_and_split():
    rec = results_to_recognition("a.wav", "hybrid", [(0, SAMPLE)])
    assert len(rec.segments) == 2           # 1100ms の無音で区切り直す
    assert rec.tokens[6].start_ms == 2900    # 時刻のない句読点は前の単語の終わり
    rec2 = results_to_recognition("a.wav", "hybrid", [(10000, SAMPLE)])
    assert rec2.tokens[0].start_ms == 10500  # 区切って送った分のずれを足す


def test_phone_detected_by_rule():
    rec = results_to_recognition("a.wav", "hybrid", [(0, SAMPLE)])
    dets = detect_rules(TokenView(rec))
    assert any(d.pii_type == "PHONE" and (d.token_start, d.token_end) == (2, 5) for d in dets)


class _FakeHTTP:
    def __init__(self):
        self.body = b""

    def request(self, url, headers, body, params):
        self.url, self.body = url, body
        return 200, json.dumps(SAMPLE).encode()


def test_library_request_has_confidence_zero_and_nolog(tmp_path):
    from amivoice import AmiVoiceClient

    t = _FakeHTTP()
    c = AmiVoiceClient(api_key="testkey", settings=make_settings("e2e"), hrp_transport=t)
    audio.write_wav(tmp_path / "a.wav", np.zeros(1600, dtype=np.float32), 16000)
    c.transcribe(str(tmp_path / "a.wav"))
    head = t.body.split(b"RIFF")[0].decode("utf-8", "replace")
    assert "confidenceLevel=0" in head
    assert "-a2-ja-general" in head
    assert "/nolog/" in t.url


def test_e2e_profile_words_format():
    from amivoice import ProfileWord, build_profile_words

    assert build_profile_words([ProfileWord(written="佐藤", biasing=0.5)]) == "佐藤  0.5"
    assert build_profile_words([ProfileWord(written="佐藤", spoken="さとう")]) == "佐藤 さとう"


# ---------------------------------------------------------------- 検出と伏せ字


def test_lowconf_after_cue_covers_whole_number():
    # 「電話番号を」→ 次の発話で番号の真ん中が誤認識（自信なし）
    rec = rec_of([("電話番号", .99), ("を", .99), ("お願い", .99), ("します", .99),
                  ("090", .9), ("-", .9), ("加藤", .3), ("-", .9), ("5678", .9), ("です", .99)], gap_after=(3,))
    view = TokenView(rec)
    dets = detect_lowconf(view, 0.6)
    assert any(d.token_start == 4 and d.token_end == 9 for d in dets)


def test_digits_after_cue_masked_even_if_short_and_confident():
    # 実測：郵便番号「029-6386」が「029-63は、知力」（知力の confidence 0.98）になった
    rec = rec_of([("新しい", .99), ("郵便番号", .99), ("から", .99), ("お願い", .99), ("します", .99), ("。", .9),
                  ("029", 1), ("-", 1), ("63", 1), ("は", 1), ("、", .7), ("知力", .98)], gap_after=(5,))
    dets = [d for d in detect_rules(TokenView(rec)) if d.pii_type == "POSTAL"]
    assert dets and dets[0].token_start == 6


def test_spelled_email_whole_utterance():
    rec = rec_of([("メールアドレス", 1), ("を", 1), ("お願い", 1), ("します", 1), ("。", 1),
                  ("S", 1), ("O", 1), ("アンダーバー", 1), ("は", 1), ("い", .9), ("M", .6), ("@", 1), ("example", 1), (".com", 1),
                  ("です", 1)], gap_after=(4,))
    d = [x for x in detect_rules(TokenView(rec)) if x.pii_type == "EMAIL"]
    assert any(x.token_start == 5 and x.token_end == 14 for x in d)


def test_condition_a_masks_text_only():
    rec = results_to_recognition("a.wav", "hybrid", [(0, SAMPLE)])
    a = compute_mask(rec, CONDITIONS["A"])
    b = compute_mask(rec, CONDITIONS["B"])
    assert a.merged == [] and b.merged
    assert "［電話番号］" in a.text and "090" not in a.text


def test_pad_extends_interval():
    rec = results_to_recognition("a.wav", "hybrid", [(0, SAMPLE)])
    r = compute_mask(rec, MaskConfig(base=("rules",), context="none", pad_ms=100))
    assert r.merged[0] == (900, 2800)


def test_perfect_recognition_has_no_leak(tmp_path):
    scripts = generate_scripts(6, seed=2)
    p = synth_dataset(scripts, [FakeTTS(tmp_path / "c")], tmp_path, ["clean"], seed=1)
    pairs = [(t, fake_recognize(t, err_pii=0, err_other=0, jitter_ms=0)) for t in load_truths(p)]
    m, _ = evaluate(pairs, MaskConfig(base=("rules",), context="answer"))
    assert m["区間漏れ率"] <= 0.1
    assert m["件数"] > 10


def test_stream_delay_reduces_leak():
    rec = results_to_recognition("a.wav", "hybrid", [(0, SAMPLE)])
    res = compute_mask(rec, MaskConfig(base=("rules",), context="none", pad_ms=0))
    arrivals = [(1000, 2700, 3200.0)]  # 発話の終わりから500ms後に確定
    assert effective_intervals(res, arrivals, 0) == []
    assert effective_intervals(res, arrivals, 2500)[0][0] == 1000


def test_split_year_birthdate():
    # 実測：「せんきゅうひゃくななじゅうろく」が「190716年」「1907、16年」になる
    rec = rec_of([("生年月日", 1), ("を", 1), ("お願い", 1), ("いたします", 1), ("。", 1),
                  ("190716", 1), ("年", 1), ("6", 1), ("月", 1), ("18", 1), ("日", 1), ("です", 1)], gap_after=(4,))
    d = [x for x in detect_rules(TokenView(rec)) if x.pii_type == "BIRTHDATE"]
    assert any(x.token_start == 5 and x.token_end == 11 for x in d)


def test_spelled_email_across_segments():
    # 1文字ずつ読む間で発話が割れても、質問の後から伏せる。化けたドメイン（「.ネット」）も伏せる
    rec = rec_of([("メールアドレス", 1), ("を", 1), ("お願い", 1), ("いたします", 1), ("。", 1),
                  ("YOECHI", .5), (".", 1), ("Y", 1), ("A", 1), ("@", 1), ("example", 1), (".", 1), ("ネット", 1), ("です", 1)],
                 gap_after=(4, 6))
    d = [x for x in detect_rules(TokenView(rec)) if x.pii_type == "EMAIL"]
    assert any(x.token_start == 5 and x.token_end == 13 for x in d)


def test_answer_stops_before_next_question():
    from voicemask.detect.context import detect_answers
    rec = rec_of([("お名前", 1), ("を", 1), ("お願い", 1), ("いたします", 1), ("。", 1),
                  ("高橋", 1), ("充", 1), ("です", 1), ("お客様", 1), ("の", 1), ("ご住所", 1)], gap_after=(4,))
    d = detect_answers(TokenView(rec))
    assert {x.token_start for x in d} == {5, 6}


def test_bridge_silence_between_family_and_given_name():
    from voicemask.mask import _bridge_silence
    toks = [Token("田中", 0, 600, 1), Token("翔太", 1200, 1800, 1), Token("です", 1800, 2000, 1), Token("はい", 2600, 2800, 1)]
    assert _bridge_silence([(0, 700), (1100, 1900)], toks, [True, True, False, False]) == [(0, 1900)]
    # あいだに伏せていない声があればつながない
    assert _bridge_silence([(0, 700), (2500, 2900)], toks, [True, True, False, True]) == [(0, 700), (2500, 2900)]


def test_reveal_time_uses_first_partial_with_text():
    from voicemask.stream_eval import reveal_time
    parts = [(1000.0, "お名前は"), (1300.0, "お名前は佐"), (1400.0, "お名前は佐藤")]
    assert reveal_time("佐藤", parts, 5000.0) == 1400.0
    assert reveal_time("鈴木", parts, 5000.0) == 5000.0  # 途中結果に出なければ確定結果を待つ


def test_emitted_intervals_only_uses_what_was_known_at_emit_time():
    from voicemask.live import Snapshot, emitted_intervals
    hist = [Snapshot(0, []), Snapshot(3000, [(1000, 2000)])]  # 1〜2秒の区間は3秒の時点で分かった
    assert emitted_intervals(hist, 1000, 4000) == []            # 1秒遅れでは、もう流れたあと
    assert emitted_intervals(hist, 2000, 4000) == [(1000, 2000)]  # 2秒遅れなら間に合う
