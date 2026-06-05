"""
回路ブロック分解モジュール  v1
入力→出力の直列経路上の T 分岐点（GND 接続節点）でブロックを分割する

分割ルール:
  1. 入力→出力の最短経路を主直列パスとして取得
  2. 主パス上の内部節点（入出力を除く）のうち GND に接続されているものをブロック境界とする
  3. 各区間について、直列部品と区間終端のシャント部品をまとめて 1 ブロックとする

TopoSizing 的な考え方との対応:
  - 境界節点 = 機能が切り替わる「エネルギー引き渡し点」
  - 各ブロックは独立した特徴量ベクトルを持ち、ブロック間の干渉を防ぐ
"""

import networkx as nx
from feature_extractor import build_graph, ACTIVE_TYPES


def decompose_blocks(circuit: dict) -> list[dict]:
    """
    回路を機能ブロックのリストに分解する。

    Parameters
    ----------
    circuit : dict
        sample_netlists.json 形式の回路定義

    Returns
    -------
    list[dict]
        ブロック数 >= 2 の場合: サブ回路 dict のリスト（各ブロックは extract_all_features に渡せる形式）
        単一ブロックの場合: [circuit] をそのまま返す
    """
    # 能動素子を含む段は T 分岐分割の対象外（バイアス節点で分断しない）。
    # ブロック分解は受動の多段回路向けに設計されている。
    if any(c["type"] in ACTIVE_TYPES for c in circuit["components"]):
        return [circuit]

    G = build_graph(circuit)
    inp = circuit["ports"]["input"]
    out = circuit["ports"]["output"]
    gnd = circuit["ports"]["gnd"]

    # ── 主経路探索（GND を通らない最短パス） ─────────────
    # GND 経由の等長パスが選ばれると境界検出が誤動作するため、
    # GND ノードを通らないパスを優先する
    try:
        paths = list(nx.all_simple_paths(G, inp, out, cutoff=20))
    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
        return [circuit]
    if not paths:
        return [circuit]

    signal_paths = [p for p in paths if gnd not in p]
    candidates = signal_paths if signal_paths else paths
    main_path = min(candidates, key=len)
    if len(main_path) < 3:
        return [circuit]  # 内部節点なし → 分割不可

    # ── ブロック境界節点の検出 ────────────────────────────
    gnd_neighbors = set(G.neighbors(gnd)) if gnd in G else set()
    # 境界 = 主パスの内部節点（入出力を除く）かつ GND に接続されている
    boundary_nodes = [n for n in main_path[1:-1] if n in gnd_neighbors]

    if not boundary_nodes:
        return [circuit]

    # ── 区間ごとにサブ回路を構築 ──────────────────────────
    segment_endpoints = [inp] + boundary_nodes + [out]
    assigned: set[str] = set()
    sub_circuits: list[dict] = []

    for idx in range(len(segment_endpoints) - 1):
        seg_start = segment_endpoints[idx]
        seg_end   = segment_endpoints[idx + 1]

        i0 = main_path.index(seg_start)
        i1 = main_path.index(seg_end)
        seg_nodes = set(main_path[i0 : i1 + 1])

        sub_comps = []
        for comp in circuit["components"]:
            if comp["id"] in assigned:
                continue
            terms = list(comp["terminals"].values())
            t0, t1 = terms[0], terms[1]

            # 直列部品: 両端が区間内節点に収まる
            if t0 in seg_nodes and t1 in seg_nodes:
                sub_comps.append(comp)
                assigned.add(comp["id"])
            # 終端シャント部品: 区間終端 ↔ GND
            # 区間終端のシャント部品はその区間（前段）に帰属させる
            elif (t0 == gnd and t1 == seg_end) or (t1 == gnd and t0 == seg_end):
                sub_comps.append(comp)
                assigned.add(comp["id"])

        if sub_comps:
            sub_circuits.append({
                "id":            f"{circuit['id']}_b{idx}",
                "name":          f"{circuit['name']} [ブロック{idx + 1}]",
                "description":   circuit.get("description", ""),
                "function_tags": circuit.get("function_tags", []),
                "components":    sub_comps,
                "ports":         {"input": seg_start, "output": seg_end, "gnd": gnd},
            })

    # 未割り当て部品（主パス外の並列枝など）を最終ブロックへ補填
    leftover = [c for c in circuit["components"] if c["id"] not in assigned]
    if leftover and sub_circuits:
        sub_circuits[-1]["components"].extend(leftover)

    return sub_circuits if len(sub_circuits) >= 2 else [circuit]


# ─────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from feature_extractor import extract_all_features

    with open("sample_netlists.json", encoding="utf-8") as f:
        data = json.load(f)

    for circuit in data["circuits"]:
        blocks = decompose_blocks(circuit)
        if len(blocks) == 1:
            print(f"[単一] {circuit['name']}")
        else:
            print(f"\n[{len(blocks)} ブロック] {circuit['name']}")
            for i, blk in enumerate(blocks):
                feat = extract_all_features(blk)
                print(f"  ブロック{i + 1}: 部品={feat['A_component']['component_types']}"
                      f"  直列={feat['B1_order']['series_type_sequence']}"
                      f"  GND並列={feat['B1_order']['shunt_type_sequence']}")
