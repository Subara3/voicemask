"""GiNZA の固有表現抽出で、人名と地名を拾う（手元で動く）。"""

from __future__ import annotations

from ..models import Detection
from .tokenview import TokenView

LABELS = {
    "Person": "NAME",
    "Province": "ADDRESS",
    "City": "ADDRESS",
    "County": "ADDRESS",
    "GPE_Other": "ADDRESS",
    "Phone_Number": "PHONE",
    "Email": "EMAIL",
    "Postal_Address": "ADDRESS",
}


class GinzaDetector:
    def __init__(self, model: str = "ja_ginza"):
        import spacy

        self.nlp = spacy.load(model)

    def detect(self, view: TokenView) -> list[Detection]:
        out = []
        for si in range(len(view.seg_bounds)):
            text = view.segment_text(si)
            if not text:
                continue
            for ent in self.nlp(text).ents:
                typ = LABELS.get(ent.label_)
                if typ:
                    s, e = view.span_to_tokens(si, ent.start_char, ent.end_char)
                    if e > s:
                        out.append(Detection(typ, s, e, "ner", f"GiNZA {ent.label_}"))
        return out
