"""
knowledge_cards — 弁別知識カードの読み込みと描画

knowledge_cards.yaml（契約は docs/CARD_SPEC.md）を読み、LLM 合成層に注入できる
テキストに描画する。カードは «役割説明» ではなく «識別の決め手・近縁との差分» を
知覚層 IR のフィールドに紐づけて載せたもの。
"""

from __future__ import annotations

import pathlib

import yaml

_DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "knowledge_cards.yaml"


def load_cards(path: str | pathlib.Path | None = None) -> dict:
    """circuit_id -> card の dict を返す。ファイルが無ければ空 dict。"""
    p = pathlib.Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("cards", {}) or {}


def render_card(card: dict, name_lookup: dict | None = None) -> str:
    """1 枚のカードを LLM 可読テキストに描画する。

    name_lookup は confused_with.with（circuit_id）を回路名に解決する辞書。
    無ければ id をそのまま表示する。
    """
    name_lookup = name_lookup or {}
    lines = [f"【{card['name']}】 {card.get('role', '')}".rstrip()]

    decisive = card.get("decisive", [])
    if decisive:
        lines.append("  決め手:")
        for d in decisive:
            lines.append(f"   - {d}")

    confused = card.get("confused_with", [])
    if confused:
        lines.append("  紛らわしい近縁との差分:")
        for ref in confused:
            tgt = ref.get("with")
            nm = name_lookup.get(tgt, tgt)
            lines.append(f"   vs {nm}: {ref.get('difference', '')}")

    if card.get("pitfalls"):
        lines.append(f"  注意: {card['pitfalls']}")

    return "\n".join(lines)
