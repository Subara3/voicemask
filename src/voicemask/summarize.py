"""伏せた後の文字だけから、通話の対応記録を作る（デモ用）。Claude が見るのは伏せた後の文字だけ。"""

from __future__ import annotations

from pathlib import Path

PROMPT = """次はコールセンターの通話の文字起こしです。個人情報は［氏名］［電話番号］のように伏せてあります。
伏せた部分を推測で埋めずに、次の形式で対応記録を作ってください。

- 用件：
- 対応：
- 次にすること：

文字起こし：
"""


def write_request(masked_text: str, out_dir: str | Path, stem: str) -> Path:
    p = Path(out_dir) / f"{stem}.ask_summary.txt"
    p.write_text(PROMPT + masked_text + "\n", encoding="utf-8")
    return p


def summarize(masked_text: str) -> str:
    from .llm_client import ask

    return ask(PROMPT + masked_text, max_tokens=2000)
