"""
回路ネットリスト トポロジー特徴量抽出スクリプト v2
変更点: B1に「直列経路上の各部品がGNDに接続されているか」を追加
        → RCローパス(R直列・C並列GND) と RCハイパス(C直列・R並列GND) を識別
"""

import json
import networkx as nx
from collections import defaultdict


def build_graph(circuit: dict) -> nx.Graph:
    G = nx.Graph()
    for comp in circuit["components"]:
        terminals = list(comp["terminals"].values())
        u, v = terminals[0], terminals[1]
        G.add_edge(u, v, type=comp["type"], id=comp["id"],
                   terminals=comp["terminals"])
    return G


# ── A. 部品特徴 ──────────────────────────────────────────

def extract_component_features(circuit: dict) -> dict:
    types = [c["type"] for c in circuit["components"]]
    count = defaultdict(int)
    for t in types:
        count[t] += 1
    return {
        "component_types":  sorted(set(types)),
        "component_counts": dict(count),
        "has_switch":    "SW" in count,
        "has_inductor":  "L"  in count,
        "has_diode":     "D"  in count or "DZ" in count,
        "has_capacitor": "C"  in count,
        "has_resistor":  "R"  in count,
        "has_zener":     "DZ" in count,
    }


# ── B1. 接続順序（強化版）────────────────────────────────

def extract_b1_component_order(circuit: dict, G: nx.Graph) -> dict:
    """
    入力→出力の直列経路上の部品列を取得し、
    各部品が「GNDに並列接続されているか（shunt）」も記録する。
    例:
      RCローパス: [{"type":"R","shunt":False}, {"type":"C","shunt":True}]
      RCハイパス: [{"type":"C","shunt":False}, {"type":"R","shunt":True}]
    """
    inp = circuit["ports"]["input"]
    out = circuit["ports"]["output"]
    gnd = circuit["ports"]["gnd"]

    result = {
        "sw_l_order":            None,
        "path_sequence":         [],   # [{"type":..., "shunt":bool}, ...]
        "series_type_sequence":  [],   # 直列部品の型リスト（ベクトル化用）
        "first_series_type":     None, # 最初の直列部品の型
        "shunt_type_sequence":   [],   # GND並列部品の型リスト
    }

    try:
        paths = list(nx.all_simple_paths(G, inp, out, cutoff=10))
    except nx.NetworkXNoPath:
        return result

    # GND を経由しないパスを優先（等長の GND 経由パスが選ばれるのを防ぐ）
    signal_paths = [p for p in paths if gnd not in p]
    candidates = signal_paths if signal_paths else paths

    for path in sorted(candidates, key=len):
        seq = []

        # 入力ノードのシャント部品（例: π型フィルタの入力コンデンサ）
        if gnd in G:
            gnd_edge = G.get_edge_data(path[0], gnd)
            if gnd_edge:
                seq.append({"type": gnd_edge.get("type", "?"), "shunt": True})

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = G.get_edge_data(u, v)
            if not data:
                continue
            # 信号パス上の直列部品
            seq.append({"type": data.get("type", "?"), "shunt": False})
            # 下流ノード v に直結するシャント部品
            if gnd in G:
                gnd_edge = G.get_edge_data(v, gnd)
                if gnd_edge:
                    seq.append({"type": gnd_edge.get("type", "?"), "shunt": True})

        if not seq:
            continue

        series = [s for s in seq if not s["shunt"]]
        shunts = [s for s in seq if s["shunt"]]

        result["path_sequence"]        = seq
        result["series_type_sequence"] = [s["type"] for s in series]
        result["first_series_type"]    = series[0]["type"] if series else None
        result["shunt_type_sequence"]  = [s["type"] for s in shunts]

        # SW / L 前後関係
        types = result["series_type_sequence"]
        if "SW" in types and "L" in types:
            result["sw_l_order"] = (
                "SW_before_L" if types.index("SW") < types.index("L")
                else "L_before_SW"
            )
        break

    return result


# ── B2. ダイオード向き ────────────────────────────────────

def extract_b2_diode_orientation(circuit: dict) -> dict:
    gnd = circuit["ports"]["gnd"]
    out = circuit["ports"]["output"]
    result = {
        "diode_anode_to_gnd":   False,
        "diode_cathode_to_out": False,
        "diode_anode_to_out":   False,
        "diode_cathode_to_gnd": False,
    }
    for comp in circuit["components"]:
        if comp["type"] not in ("D", "DZ"):
            continue
        anode   = comp["terminals"].get("anode")
        cathode = comp["terminals"].get("cathode")
        if anode   == gnd: result["diode_anode_to_gnd"]   = True
        if cathode == out: result["diode_cathode_to_out"]  = True
        if anode   == out: result["diode_anode_to_out"]    = True
        if cathode == gnd: result["diode_cathode_to_gnd"]  = True
    return result


# ── B3. 直列 / 並列 ──────────────────────────────────────

def extract_b3_series_parallel(circuit: dict, G: nx.Graph) -> dict:
    # nx.Graph は同一ノード対の多重辺を保持しないため、circuit dict から並列検出する
    pair_to_types: dict = defaultdict(list)
    for comp in circuit["components"]:
        terms = list(comp["terminals"].values())
        key = tuple(sorted(terms))
        pair_to_types[key].append(comp["type"])

    parallel_pairs = {str(k): v for k, v in pair_to_types.items() if len(v) >= 2}
    degree2_nodes  = [n for n in G.nodes() if G.degree(n) == 2]

    return {
        "has_parallel_components": len(parallel_pairs) > 0,
        "parallel_pairs":          parallel_pairs,
        "series_chain_nodes":      degree2_nodes,
        "series_chain_length":     len(degree2_nodes),
    }


# ── C. ノード特徴 ─────────────────────────────────────────

def extract_node_features(circuit: dict, G: nx.Graph) -> dict:
    gnd = circuit["ports"]["gnd"]
    gnd_connected = (
        [G[gnd][nb]["id"] for nb in G.neighbors(gnd) if "id" in G[gnd][nb]]
        if gnd in G else []
    )
    high_degree = {n: G.degree(n) for n in G.nodes() if G.degree(n) >= 3}
    return {
        "node_count":           G.number_of_nodes(),
        "gnd_connected_parts":  gnd_connected,
        "high_degree_nodes":    high_degree,
        "has_high_degree_node": len(high_degree) > 0,
        "cycle_count":          len(nx.cycle_basis(G)),
    }


# ── まとめて抽出 ──────────────────────────────────────────

def extract_all_features(circuit: dict) -> dict:
    G = build_graph(circuit)
    return {
        "circuit_id":         circuit["id"],
        "circuit_name":       circuit["name"],
        "description":        circuit.get("description", ""),
        "function_tags":      circuit.get("function_tags", []),
        "ports":              circuit["ports"],
        "A_component":        extract_component_features(circuit),
        "B1_order":           extract_b1_component_order(circuit, G),
        "B2_diode":           extract_b2_diode_orientation(circuit),
        "B3_series_parallel": extract_b3_series_parallel(circuit, G),
        "C_node":             extract_node_features(circuit, G),
    }


# ── 階層特徴量抽出 ─────────────────────────────────────────

def extract_hierarchical_features(circuit: dict) -> dict:
    """
    回路レベルのフラット特徴量にブロック分解結果を付加して返す。

    単一ブロック回路: is_hierarchical=False, blocks=[]
    複数ブロック回路: is_hierarchical=True,  blocks=[各ブロックのextract_all_features結果]
    """
    from block_decomposer import decompose_blocks  # 循環インポート回避

    base = extract_all_features(circuit)
    blocks = decompose_blocks(circuit)

    if len(blocks) <= 1:
        base["is_hierarchical"] = False
        base["n_blocks"]        = 1
        base["blocks"]          = []
        return base

    base["is_hierarchical"] = True
    base["n_blocks"]        = len(blocks)
    base["blocks"]          = [extract_all_features(b) for b in blocks]
    return base


if __name__ == "__main__":
    with open("sample_netlists.json", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for circuit in data["circuits"]:
        f = extract_hierarchical_features(circuit)
        results.append(f)
        print(f"\n{'='*50}")
        print(f"【{f['circuit_name']}】")
        print(f"  直列シーケンス  : {f['B1_order']['series_type_sequence']}")
        print(f"  先頭部品        : {f['B1_order']['first_series_type']}")
        print(f"  GND並列部品     : {f['B1_order']['shunt_type_sequence']}")
        print(f"  SW-L順序        : {f['B1_order']['sw_l_order']}")
        if f["is_hierarchical"]:
            print(f"  ── {f['n_blocks']} ブロック分割 ──")
            for i, blk in enumerate(f["blocks"]):
                print(f"    ブロック{i + 1}: {blk['A_component']['component_types']}"
                      f"  直列={blk['B1_order']['series_type_sequence']}")

    with open("features_db.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n→ features_db.json 更新完了")
