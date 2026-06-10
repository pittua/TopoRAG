"""
回路ネットリスト トポロジー特徴量抽出スクリプト v2
変更点: B1に「直列経路上の各部品がGNDに接続されているか」を追加
        → RCローパス(R直列・C並列GND) と RCハイパス(C直列・R並列GND) を識別
"""

import json
import networkx as nx
from collections import defaultdict


# グラフ基盤プリミティブ・素子タクソノミーは circuit_graph に集約した
# （feature_extractor ⇄ topo_kernel ⇄ block_decomposer の循環インポート解消）。
# 後方互換のため本モジュールからも再エクスポートする
# （`from feature_extractor import build_graph` 等の既存利用を維持）。
from circuit_graph import (  # noqa: F401  re-export
    ACTIVE_TYPES, BJT_TYPES, FET_TYPES, PTYPE_TYPES,
    is_device, build_graph, bounded_simple_paths,
)

# 高水準抽出は WL カーネルとブロック分解を使う。circuit_graph 経由で循環が
# 解けたため、モジュール先頭で通常 import できる（旧: 関数内 import の握り潰し）。
from topo_kernel import wl_histogram
from block_decomposer import decompose_blocks

# トランジスタの端子別名 → 役割（制御端子 / 出力側 / 共通側）への正規化
_CTRL_PINS   = {"base", "gate"}                  # 入力制御端子
_OUTPUT_PINS = {"collector", "drain"}            # 主出力端子
_COMMON_PINS = {"emitter", "source"}             # 共通／帰還端子


# ── スイッチング素子の正規化 ─────────────────────────────

def switch_mosfet_ids(circuit: dict) -> set:
    """
    スイッチング・コンバータの開閉素子として使われている MOSFET の id 集合を返す。

    増幅用 MOSFET（バイアス網で動作）と区別するため、以下を満たす場合のみ
    「スイッチ＝SW 相当」とみなす（保守的）:
      - 回路が L と D を含む（フリーホイール付き変換器の文脈）、かつ
      - その MOSFET の drain か source がインダクタ端子（＝スイッチ節点）に接する。

    これにより、SW でモデル化した DB コンバータと、実 MOSFET を使う実機コンバータが
    同一の構造ベクトルに正規化される（buck/boost の表現ミスマッチを解消）。
    増幅段の MOSFET（L/D を伴わない）や DB の SW コンバータ（FET 無し）は不変。
    """
    comps = circuit["components"]
    has_L = any(c["type"] == "L" for c in comps)
    has_D = any(c["type"] in ("D", "DZ") for c in comps)
    if not (has_L and has_D):
        return set()
    ind_nets = set()
    for c in comps:
        if c["type"] == "L":
            ind_nets.update(c["terminals"].values())
    ids = set()
    for c in comps:
        if c["type"] not in FET_TYPES:
            continue
        ds = {c["terminals"].get("drain"), c["terminals"].get("source")}
        if ds & ind_nets:
            ids.add(c["id"])
    return ids


# ── A. 部品特徴 ──────────────────────────────────────────

def extract_component_features(circuit: dict) -> dict:
    types = [c["type"] for c in circuit["components"]]
    count = defaultdict(int)
    for t in types:
        count[t] += 1
    type_set = set(types)

    # スイッチ用途の MOSFET は「増幅用 MOSFET」ではなく「スイッチ(SW相当)」として扱う
    sw_ids = switch_mosfet_ids(circuit)
    amp_fets = {c["id"] for c in circuit["components"]
                if c["type"] in FET_TYPES and c["id"] not in sw_ids}
    has_amp_mosfet = bool(amp_fets)
    has_other_active = bool(type_set & (ACTIVE_TYPES - FET_TYPES))

    # 正規化部品数: スイッチ化 MOSFET を SW にカウントし、DZ は D に合算。
    # （実機の実 MOSFET スイッチと DB の SW 表現を部品数でも統一する）
    norm_count: dict = defaultdict(int)
    for c in circuit["components"]:
        t = c["type"]
        if t in FET_TYPES and c["id"] in sw_ids:
            norm_count["SW"] += 1
        elif t == "DZ":
            norm_count["D"] += 1
        else:
            norm_count[t] += 1

    return {
        "component_types":  sorted(type_set),
        "component_counts": dict(count),
        "normalized_counts": dict(norm_count),
        "has_switch":    ("SW" in count) or bool(sw_ids),
        "has_inductor":  "L"  in count,
        "has_diode":     "D"  in count or "DZ" in count,
        "has_capacitor": "C"  in count,
        "has_resistor":  "R"  in count,
        "has_zener":     "DZ" in count,
        "has_bjt":       bool(type_set & BJT_TYPES),
        "has_mosfet":    has_amp_mosfet,
        "has_opamp":     "OPAMP" in count,
        "has_active":    has_other_active or has_amp_mosfet,
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

    # ポートノードがグラフに存在しない場合（例: OpAmp 等をスキップした変換回路で
    # 入出力が能動素子側にしか繋がっていない）は経路解析を諦めて空結果を返す。
    if inp not in G or out not in G:
        return result

    paths = bounded_simple_paths(G, inp, out, cutoff=10)  # 指数爆発ガード

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
    inp = circuit["ports"].get("input")
    result = {
        "diode_anode_to_gnd":   False,
        "diode_cathode_to_out": False,
        "diode_anode_to_out":   False,
        "diode_cathode_to_gnd": False,
        # ── 役割ベース（整流/クリッパ/平滑の識別用）──────────
        "diode_series":     False,  # 信号経路上の直列ダイオード（両端とも非GND）
        "diode_anode_at_input": False,  # アノードが入力ポート（真の整流段）
        "diode_shunt":      False,  # 片端のみGND（クリッパ/ツェナー型のシャント）
        "rectifier_smoothing": False,  # 直列D ＋ 出力→GND の平滑コンデンサ
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

        at_gnd = (anode == gnd) + (cathode == gnd)
        if at_gnd == 0:
            result["diode_series"] = True
        elif at_gnd == 1:
            result["diode_shunt"] = True
        if anode == inp:
            result["diode_anode_at_input"] = True

    # 平滑コンデンサ: 直列ダイオードがある場合に限り、出力→GND の C を検出する。
    # （直列D を条件にすることで素の RC ローパス等への誤発火を防ぐ＝非ダイオード回路は不変）
    if result["diode_series"]:
        for comp in circuit["components"]:
            if comp["type"] != "C":
                continue
            nets = set(comp["terminals"].values())
            if out in nets and gnd in nets:
                result["rectifier_smoothing"] = True
                break
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


# ── D. 能動素子の構成（BJT/MOSFET/OpAmp）─────────────────

def _passive_net_adjacency(circuit: dict) -> dict:
    """2 端子の受動/スイッチ素子のみを辿るネット隣接（能動素子は跨がない）。"""
    adj: dict = defaultdict(set)
    for c in circuit["components"]:
        if is_device(c):
            continue
        terms = list(c["terminals"].values())
        if len(terms) != 2:
            continue
        a, b = terms
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _reaches(adj: dict, start, target, max_hops: int = 3, avoid=frozenset()) -> bool:
    """
    start から受動素子のみを介して max_hops 以内で target に到達できるか。
    avoid のネット（通常は GND）は経由しない——全ネットが GND 経由で繋がって
    しまい構成判定が誤るのを防ぐ。
    """
    if start is None or target is None:
        return False
    if start == target:
        return True
    seen = {start}
    frontier = {start}
    for _ in range(max_hops):
        nxt = set()
        for n in frontier:
            for m in adj.get(n, ()):
                if m == target:
                    return True
                if m not in seen and m not in avoid:
                    seen.add(m)
                    nxt.add(m)
        frontier = nxt
        if not frontier:
            break
    return False


def _classify_transistor(pins: dict, adj: dict, inp, out, gnd) -> str | None:
    """
    端子ネット辞書 pins（base/collector/emitter 等 → ネット名）から
    増幅器の接地構成を判定する。BJT/FET を CE/CC/CB（接地端子）に正規化。

      CE: 入力=制御端子(base/gate), 出力=出力端子(collector/drain)   … 反転増幅
      CC: 入力=制御端子,           出力=共通端子(emitter/source)     … フォロワ
      CB: 入力=共通端子,           出力=出力端子                     … ベース接地
    """
    ctrl  = next((n for p, n in pins.items() if p in _CTRL_PINS), None)
    out_p = next((n for p, n in pins.items() if p in _OUTPUT_PINS), None)
    com_p = next((n for p, n in pins.items() if p in _COMMON_PINS), None)
    avoid = frozenset({gnd})

    def near(net, port):
        return net is not None and _reaches(adj, net, port, avoid=avoid)

    in_at_ctrl = near(ctrl, inp)
    in_at_com  = near(com_p, inp)
    out_at_outp = near(out_p, out)
    out_at_com  = near(com_p, out)

    if in_at_ctrl and out_at_outp:
        return "CE"
    if in_at_ctrl and out_at_com:
        return "CC"
    if in_at_com and out_at_outp:
        return "CB"
    # フォールバック: 出力端子が出力ポート寄りなら CE 相当とみなす
    if out_at_outp:
        return "CE"
    if out_at_com:
        return "CC"
    return None


def _classify_opamp(pins: dict, adj: dict, inp, out, gnd) -> str | None:
    """
    OpAmp 端子（in_p/in_n/out）から構成を判定する。
      buffer        : 出力が反転入力に直結（ボルテージフォロワ）
      inverting     : 入力信号が反転入力に入り、出力→反転入力に帰還あり
      non_inverting : 入力信号が非反転入力に入り、出力→反転入力に帰還あり
      comparator    : 帰還なし
    """
    inn = pins.get("in_n")
    inp_pin = pins.get("in_p")
    outp = pins.get("out")
    avoid = frozenset({gnd})

    feedback = _reaches(adj, outp, inn, avoid=avoid)
    direct_buffer = outp is not None and outp == inn

    in_at_inn = _reaches(adj, inn, inp, avoid=avoid)
    in_at_inp = _reaches(adj, inp_pin, inp, avoid=avoid)

    if direct_buffer:
        return "buffer"
    if not feedback:
        return "comparator"
    if in_at_inn and not in_at_inp:
        return "inverting"
    if in_at_inp:
        return "non_inverting"
    return "inverting" if feedback else None


def _transistor_pin(comp: dict, role: str):
    """role='ctrl'|'common'|'output' のネット名を返す（BJT/FET の端子別名を吸収）。"""
    pins = comp["terminals"]
    sets = {"ctrl": _CTRL_PINS, "common": _COMMON_PINS, "output": _OUTPUT_PINS}[role]
    return next((n for p, n in pins.items() if p in sets), None)


def _is_diode_connected(comp: dict) -> bool:
    """ダイオード接続（制御端子＝出力端子が同一ネット）か。カレントミラーの参照側。"""
    return (_transistor_pin(comp, "ctrl") is not None
            and _transistor_pin(comp, "ctrl") == _transistor_pin(comp, "output"))


def extract_active_features(circuit: dict) -> dict:
    """能動素子（BJT/MOSFET/OpAmp）の構成特徴を抽出する。"""
    ports = circuit["ports"]
    inp, out, gnd = ports["input"], ports["output"], ports["gnd"]
    inp2 = ports.get("input2")          # 差動回路の第2入力（任意）
    adj = _passive_net_adjacency(circuit)

    result = {
        "has_bjt":             False,
        "has_mosfet":          False,
        "has_opamp":           False,
        "has_active":          False,
        "n_active":            0,
        "p_type":              False,
        "transistor_config":   None,   # "CE" / "CC" / "CB"
        "opamp_config":        None,   # "inverting" / "non_inverting" / "buffer" / "comparator"
        "has_feedback":        False,
        "is_inverting":        False,
        "is_follower":         False,
        "has_diode_connected": False,  # カレントミラー参照側
        "has_coupled_pair":    False,  # 差動対/ロングテール（共通端子を共有する2素子）
        "is_differential":     False,  # 結合ペア＋2つの独立信号入力
    }

    transistors = []
    common_groups: dict = defaultdict(list)   # 共通端子ネット → 素子リスト
    sw_ids = switch_mosfet_ids(circuit)        # スイッチ用途 MOSFET は能動素子に数えない

    for c in circuit["components"]:
        t = c["type"]
        if t not in ACTIVE_TYPES:
            continue
        if t in FET_TYPES and c["id"] in sw_ids:
            continue  # SW 相当（スイッチング変換器の開閉素子）→ 能動扱いしない
        result["has_active"] = True
        result["n_active"] += 1
        if t in PTYPE_TYPES:
            result["p_type"] = True
        pins = c["terminals"]

        if t in BJT_TYPES or t in FET_TYPES:
            result["has_bjt"]    = result["has_bjt"]    or t in BJT_TYPES
            result["has_mosfet"] = result["has_mosfet"] or t in FET_TYPES
            cfg = _classify_transistor(pins, adj, inp, out, gnd)
            result["transistor_config"] = result["transistor_config"] or cfg
            if _is_diode_connected(c):
                result["has_diode_connected"] = True
            transistors.append(c)
            com = _transistor_pin(c, "common")
            if com is not None and com != gnd:
                common_groups[com].append(c)
        elif t == "OPAMP":
            result["has_opamp"] = True
            cfg = _classify_opamp(pins, adj, inp, out, gnd)
            result["opamp_config"] = result["opamp_config"] or cfg

    # 結合ペア（共通端子＝非GNDのテールを共有する2素子以上）の検出
    signal_inputs = {inp, inp2} - {None}
    for group in common_groups.values():
        if len(group) < 2:
            continue
        result["has_coupled_pair"] = True
        ctrl_nets = {_transistor_pin(c, "ctrl") for c in group}
        # 制御端子が異なる独立入力に2つ以上繋がっていれば差動
        if len(ctrl_nets & signal_inputs) >= 2:
            result["is_differential"] = True

    # 派生フラグ
    tc, oc = result["transistor_config"], result["opamp_config"]
    result["is_inverting"] = (tc == "CE") or (oc == "inverting")
    result["is_follower"]  = (tc == "CC") or (oc == "buffer")
    result["has_feedback"] = oc in ("inverting", "non_inverting", "buffer")
    return result


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
        "D_active":           extract_active_features(circuit),
        "wl_features":        wl_histogram(circuit, G=G),
    }


# ── 階層特徴量抽出 ─────────────────────────────────────────

def extract_hierarchical_features(circuit: dict) -> dict:
    """
    回路レベルのフラット特徴量にブロック分解結果を付加して返す。

    単一ブロック回路: is_hierarchical=False, blocks=[]
    複数ブロック回路: is_hierarchical=True,  blocks=[各ブロックのextract_all_features結果]
    """
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
