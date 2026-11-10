"""Claude API の呼び出しと、使った金額の上限。

- 使った金額は data/llm/ledger.json に足していき、上限（既定 10 ドル、VOICEMASK_LLM_BUDGET_USD）を超えそうなら呼ばない
- 呼ぶ前に「入力の文字数から見積もった入力トークン＋max_tokens 全部」を上乗せして判定する（多めに見積もる）
- 答えはキャッシュし、同じ問いをもう一度送らない
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 1M トークンあたりのドル（入力, 出力）。2026-10 時点の公表価格
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.00, 50.00),
}
WORST = (10.00, 50.00)  # 知らないモデル（フォールバック先など）は高い方で数える

DEFAULT_MODEL = "claude-haiku-5-5"  # 本人の指定（2026-11-10）
LEDGER = Path("data/llm/ledger.json")


class BudgetExceeded(RuntimeError):
    pass


def budget_usd() -> float:
    return float(os.environ.get("VOICEMASK_LLM_BUDGET_USD", "10"))


def spent_usd() -> float:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8")).get("spent_usd", 0.0)
    return 0.0


def _record(model: str, usage, cost: float, refused: bool = False) -> None:
    d = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"spent_usd": 0.0, "calls": 0}
    d["spent_usd"] = round(d["spent_usd"] + cost, 6)
    d["calls"] = d.get("calls", 0) + 1
    d["refusals"] = d.get("refusals", 0) + (1 if refused else 0)
    d.setdefault("by_model", {}).setdefault(model, 0.0)
    d["by_model"][model] = round(d["by_model"][model] + cost, 6)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def ask(prompt: str, max_tokens: int = 4000, model: str | None = None, effort: str = "low") -> str:
    """1回呼んで文字の答えを返す。上限を超えそうなら BudgetExceeded。"""
    import anthropic

    model = model or os.environ.get("VOICEMASK_LLM_MODEL", DEFAULT_MODEL)
    pin, pout = PRICES.get(model, WORST)
    est_in = len(prompt) * 1.5  # 日本語は1文字1トークン強。多めに見る
    worst = est_in / 1e6 * pin + max_tokens / 1e6 * pout
    if spent_usd() + worst > budget_usd():
        raise BudgetExceeded(f"Claude の上限 {budget_usd()} ドルに届くので呼びません（使用済み {spent_usd():.3f} ドル）")

    client = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=max_tokens, output_config={"effort": effort},
              messages=[{"role": "user", "content": prompt}])
    if model.startswith("claude-haiku"):
        resp = client.messages.create(**kw)  # Haiku は断られたときの自動の切り替えがない
    else:
        resp = client.beta.messages.create(**kw, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    served = getattr(resp, "model", model) or model
    pin, pout = PRICES.get(served, WORST)
    u = resp.usage
    cost = (u.input_tokens or 0) / 1e6 * pin + (u.output_tokens or 0) / 1e6 * pout
    _record(served, u, cost, refused=resp.stop_reason == "refusal")
    if resp.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
