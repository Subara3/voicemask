"""データの型。認識結果・正解・検出・伏せる区間。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PII_TYPES = ("NAME", "PHONE", "ADDRESS", "POSTAL", "BIRTHDATE", "CARD", "EMAIL")

PII_LABELS_JA = {
    "NAME": "氏名",
    "PHONE": "電話番号",
    "ADDRESS": "住所",
    "POSTAL": "郵便番号",
    "BIRTHDATE": "生年月日",
    "CARD": "カード番号",
    "EMAIL": "メールアドレス",
    "SUSPECT": "要確認",
}

PUNCT = set("、。，．,.？！?! 　")


# ---------------------------------------------------------------- 認識結果


@dataclass
class Token:
    written: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    spoken: str = ""

    @property
    def is_punct(self) -> bool:
        return not self.written or all(c in PUNCT for c in self.written)


@dataclass
class Segment:
    tokens: list[Token]
    start_ms: int
    end_ms: int
    confidence: float = 0.0

    @property
    def text(self) -> str:
        return "".join(t.written for t in self.tokens)


@dataclass
class Recognition:
    """1つの音声ファイルの認識結果。segments は時刻順。"""

    audio_path: str
    engine: str
    segments: list[Segment]
    raw: list[Any] = field(default_factory=list)

    @property
    def tokens(self) -> list[Token]:
        return [t for s in self.segments for t in s.tokens]

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Recognition":
        segs = [
            Segment(
                tokens=[Token(**t) for t in s["tokens"]],
                start_ms=s["start_ms"],
                end_ms=s["end_ms"],
                confidence=s.get("confidence", 0.0),
            )
            for s in d["segments"]
        ]
        return cls(d["audio_path"], d["engine"], segs, d.get("raw", []))


# ---------------------------------------------------------------- 台本と正解


@dataclass
class Part:
    """発話の部品。個人情報の値は1つの部品として独立させる。"""

    text: str           # 表記（評価で照合する文字）
    reading: str        # 合成に渡す文字（個人情報はかなで読みを固定する）
    pii_type: str = ""  # 空なら個人情報ではない
    slot: str = ""      # 値の種類（ORDER・PRICE など個人情報でない値も入る）


@dataclass
class Utterance:
    speaker: str        # "operator" / "customer"
    parts: list[Part]

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts)


@dataclass
class CallScript:
    call_id: str
    scenario: str
    utterances: list[Utterance]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CallScript":
        utts = [Utterance(u["speaker"], [Part(**p) for p in u["parts"]]) for u in d["utterances"]]
        return cls(d["call_id"], d["scenario"], utts)


@dataclass
class TruthSpan:
    start_ms: int
    end_ms: int
    pii_type: str
    text: str
    speaker: str = ""

    @property
    def dur(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class CallTruth:
    call_id: str
    audio_path: str
    duration_ms: int
    sample_rate: int
    variant: str
    scenario: str
    spans: list[TruthSpan]
    parts: list[dict]            # 部品ごとの時刻（個人情報でない部品も含む）
    voices: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CallTruth":
        return cls(
            d["call_id"], d["audio_path"], d["duration_ms"], d["sample_rate"], d["variant"],
            d["scenario"], [TruthSpan(**s) for s in d["spans"]], d["parts"], d.get("voices", {}),
        )

    @property
    def key(self) -> str:
        return f"{self.call_id}__{self.variant}"


# ---------------------------------------------------------------- 検出と伏せる区間


@dataclass
class Detection:
    pii_type: str
    token_start: int    # Recognition.tokens の添字（end は含まない）
    token_end: int
    source: str         # "rule" / "ner" / "context" / "neighbor" / "answer" / "llm"
    reason: str = ""


@dataclass
class MaskInterval:
    start_ms: int
    end_ms: int
    pii_type: str
    source: str
    reason: str
    min_confidence: float
