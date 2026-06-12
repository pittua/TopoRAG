"""③プロンプト再設計の回帰ガード。

判定対象クエリ側はラベル(機能タグ・説明)を伏せる（カンニング防止）。
カードのある回路は弁別カードが注入される。
"""
from circuit_ir import build_ir, render_ir
from feature_extractor import extract_all_features


def _circ(comps, ports, tags):
    return {"id": "halfwave_rect_001", "name": "半波整流回路",
            "description": "ダイオード1本で整流する回路。",
            "function_tags": tags, "components": comps, "ports": ports}


HALFWAVE = _circ(
    [{"id": "D1", "type": "D", "terminals": {"anode": "IN", "cathode": "OUT"}},
     {"id": "R1", "type": "R", "terminals": {"p": "OUT", "n": "GND"}}],
    {"input": "IN", "output": "OUT", "gnd": "GND"},
    ["rectifier", "halfwave", "ac_dc"])


def test_render_suppresses_labels():
    ir = build_ir(extract_all_features(HALFWAVE))
    shown = render_ir(ir, reveal_labels=True)
    hidden = render_ir(ir, reveal_labels=False)
    assert "機能タグ" in shown and "rectifier" in shown
    assert "機能タグ" not in hidden and "rectifier" not in hidden
    # 構造事実は両方に残る
    assert "ダイオード役割" in hidden
    assert "先頭直列部品" in hidden


def test_query_side_does_not_leak_tags(rag):
    # 判定対象クエリの IR にタグ・説明が出ないこと（直接カンニング防止）
    system, user = rag.build_prompt(HALFWAVE, top_k=3)
    q_section = user.split("## 近縁候補")[0]   # クエリ IR 部分のみ
    assert "機能タグ" not in q_section
    assert "rectifier" not in q_section
    assert "整流する回路" not in q_section          # description も漏れない
    # 構造事実は残る
    assert "ダイオード役割" in q_section


def test_card_is_injected_for_carded_circuit(rag):
    system, user = rag.build_prompt(HALFWAVE, top_k=3)
    assert "決め手" in user                          # カード本文
    assert "紛らわしい近縁との差分" in user
    assert "負クリッパ回路" in user                   # confused_with が名前解決される


def test_system_prompt_is_card_matching_framing(rag):
    system, _ = rag.build_prompt(HALFWAVE, top_k=3)
    assert "照合" in system
    assert "推論する必要はありません" in system
    assert "該当なし" in system
