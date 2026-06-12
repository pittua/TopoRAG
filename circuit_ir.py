"""
circuit_ir — 知覚層の構造IR（Intermediate Representation）契約

役割:
    feature_extractor が出力する特徴 dict を、LLM 合成層が消費する
    «決定的な構造IR» に翻訳する。方針再定義（2026-06-11）の3層
    アーキテクチャ「知覚層(決定的IR) → 構造検索 → LLM合成」の
    第1層の出力仕様を、ここに一本化する。

設計原則:
    - build_ir(features) は構造化 dict（JSON 直列化可能・版付き）を返す。
    - render_ir(ir) はそれを LLM 可読テキストに描画する（唯一の描画経路）。
    - IR は «生のネットリスト隣接» を載せない。知覚層が消化した構造事実
      （直列列・シャント・ループ・能動構成・ダイオード役割）だけを渡す。
      LLM にグラフ走査をさせないこと自体が欠陥1への解だからである。
    - 決定的: 同じ入力 features からは常に同じ IR が出る（順序も固定）。

契約の詳細は docs/IR_SPEC.md を参照。
"""

from __future__ import annotations

IR_VERSION = "1.0"

# 能動素子構成コードの日本語ラベル（描画用）
_TRANSISTOR_CONFIG_LABEL = {
    "CE": "接地エミッタ/ソース(反転増幅)",
    "CC": "コレクタ/ドレイン接地(フォロワ)",
    "CB": "ベース/ゲート接地",
}


def build_ir(features: dict) -> dict:
    """特徴 dict → 構造IR(dict)。

    `features` は extract_all_features / extract_hierarchical_features の
    出力、またはそのブロック要素を想定する（どちらも同じキー集合を持つ。
    is_hierarchical / blocks は階層回路にのみ存在するため .get で参照する）。
    """
    A = features.get("A_component", {})
    B1 = features.get("B1_order", {})
    B2 = features.get("B2_diode", {})
    B3 = features.get("B3_series_parallel", {})
    C = features.get("C_node", {})
    D = features.get("D_active", {})

    ir: dict = {
        "ir_version": IR_VERSION,
        "circuit_name": features.get("circuit_name"),
        "tags": list(features.get("function_tags", [])),
        "description": features.get("description", ""),
        "ports": _ports_ir(features.get("ports", {})),
        "components": {
            "types": list(A.get("component_types", [])),
            "normalized_counts": dict(A.get("normalized_counts", {})),
        },
        "topology": {
            "first_series": B1.get("first_series_type"),
            "series_sequence": list(B1.get("series_type_sequence", [])),
            "shunt_to_gnd": list(B1.get("shunt_type_sequence", [])),
            "sw_l_order": B1.get("sw_l_order"),
            "loop_count": C.get("cycle_count", 0),
            "has_parallel": bool(B3.get("has_parallel_components", False)),
        },
        "diode": _diode_ir(A, B2),
        "active": _active_ir(D),
        "hierarchy": _hierarchy_ir(features),
    }
    return ir


def _ports_ir(ports: dict) -> dict:
    out = {
        "input": ports.get("input"),
        "output": ports.get("output"),
        "gnd": ports.get("gnd"),
    }
    if ports.get("input2") is not None:
        out["input2"] = ports["input2"]
    return out


def _diode_ir(A: dict, B2: dict) -> dict | None:
    """ダイオードを含む回路にのみ diode セクションを出す（無い回路は None）。"""
    if not A.get("has_diode", False):
        return None
    return {
        "anode_to_gnd": bool(B2.get("diode_anode_to_gnd", False)),
        "cathode_to_out": bool(B2.get("diode_cathode_to_out", False)),
        # 役割ベース（整流/クリッパ/平滑の識別: 改善(a) dim34-37）
        "series": bool(B2.get("diode_series", False)),
        "anode_at_input": bool(B2.get("diode_anode_at_input", False)),
        "shunt": bool(B2.get("diode_shunt", False)),
        "rectifier_smoothing": bool(B2.get("rectifier_smoothing", False)),
        "has_zener": bool(A.get("has_zener", False)),
    }


def _active_ir(D: dict) -> dict | None:
    """能動素子を含む回路にのみ active セクションを出す（無い回路は None）。"""
    if not D.get("has_active", False):
        return None
    devices = []
    if D.get("has_bjt"):
        devices.append("BJT")
    if D.get("has_mosfet"):
        devices.append("MOSFET")
    if D.get("has_opamp"):
        devices.append("OpAmp")
    return {
        "devices": devices,
        "n_active": D.get("n_active", 0),
        "transistor_config": D.get("transistor_config"),
        "opamp_config": D.get("opamp_config"),
        "p_type": bool(D.get("p_type", False)),
        "has_feedback": bool(D.get("has_feedback", False)),
        "is_inverting": bool(D.get("is_inverting", False)),
        "is_follower": bool(D.get("is_follower", False)),
        "is_differential": bool(D.get("is_differential", False)),
        "has_coupled_pair": bool(D.get("has_coupled_pair", False)),
        "has_diode_connected": bool(D.get("has_diode_connected", False)),
    }


def _hierarchy_ir(features: dict) -> dict | None:
    """複合回路にのみ hierarchy セクションを出す（単一ブロックは None）。"""
    if not features.get("is_hierarchical", False):
        return None
    blocks = features.get("blocks", [])
    if not blocks:
        return None
    return {
        "n_blocks": features.get("n_blocks", len(blocks)),
        "blocks": [
            {
                "types": list(b.get("A_component", {}).get("component_types", [])),
                "series_sequence": list(
                    b.get("B1_order", {}).get("series_type_sequence", [])
                ),
                "shunt_to_gnd": list(
                    b.get("B1_order", {}).get("shunt_type_sequence", [])
                ),
            }
            for b in blocks
        ],
    }


# ── 描画（IR → LLM 可読テキスト）──────────────────────────────

def render_ir(ir: dict, reveal_labels: bool = True) -> str:
    """構造IR → テキスト。プロンプトに渡る唯一の描画経路。

    reveal_labels=False で機能タグ・説明を伏せる（«ラベル抑制モード»）。
    タグはこのプロジェクトでは実質ラベル（答え）であり、説明も答えを散文で
    述べるため、判定対象クエリ側ではこれらを隠さないとカンニングになる。
    構造事実（部品/直列列/シャント/ループ/能動/ダイオード役割）は常に残す。
    """
    lines: list[str] = []

    if reveal_labels and ir.get("tags"):
        lines.append(f"  機能タグ        : {', '.join(ir['tags'])}")
    if reveal_labels and ir.get("description"):
        lines.append(f"  説明            : {ir['description']}")

    comp = ir["components"]
    topo = ir["topology"]
    lines += [
        f"  部品種別        : {comp['types']}",
        f"  先頭直列部品    : {topo['first_series']}",
        f"  直列シーケンス  : {topo['series_sequence']}",
        f"  GND並列部品     : {topo['shunt_to_gnd']}",
        f"  SW-L順序        : {topo['sw_l_order']}",
    ]

    diode = ir.get("diode")
    if diode is not None:
        lines.append(f"  Dアノード→GND  : {diode['anode_to_gnd']}")
        lines.append(f"  Dカソード→OUT  : {diode['cathode_to_out']}")
        roles = _diode_role_labels(diode)
        if roles:
            lines.append(f"  ダイオード役割  : {' / '.join(roles)}")

    lines.append(f"  ループ数        : {topo['loop_count']}")

    active = ir.get("active")
    if active is not None:
        lines.append(f"  能動素子構成    : {_active_label(active)}")

    hier = ir.get("hierarchy")
    if hier is not None:
        lines.append(f"  ブロック構成     : {hier['n_blocks']} ブロック")
        for i, blk in enumerate(hier["blocks"]):
            lines.append(
                f"    ブロック{i + 1}: 部品={blk['types']}"
                f"  直列={blk['series_sequence']}"
                f"  GND並列={blk['shunt_to_gnd']}"
            )

    return "\n".join(lines)


def _diode_role_labels(diode: dict) -> list[str]:
    labels = []
    if diode.get("anode_at_input"):
        labels.append("整流(アノード入力)")
    elif diode.get("series"):
        labels.append("直列")
    if diode.get("shunt"):
        labels.append("シャント(クリッパ/ツェナー)")
    if diode.get("rectifier_smoothing"):
        labels.append("整流平滑(直列D＋出力C)")
    if diode.get("has_zener"):
        labels.append("ツェナー")
    return labels


def _active_label(active: dict) -> str:
    parts = []
    if active.get("transistor_config"):
        parts.append(
            _TRANSISTOR_CONFIG_LABEL.get(
                active["transistor_config"], active["transistor_config"]
            )
        )
    if active.get("opamp_config"):
        parts.append(f"OpAmp:{active['opamp_config']}")
    label = " / ".join(parts) or "不明"
    if active.get("has_feedback"):
        label += "  (帰還あり)"
    if active.get("p_type"):
        label += "  (p型)"
    if active.get("is_differential"):
        label += "  (差動)"
    return label
