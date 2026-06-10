"""
circuit_graph.py — グラフ基盤プリミティブ（依存なしの最下層モジュール）

回路 dict → networkx グラフ化と、素子タクソノミー・単純パス列挙ガードを提供する。
ここに依存を集約することで feature_extractor / topo_kernel / block_decomposer の
循環インポート（旧: 関数内 import で握り潰していた）を解消する。

レイヤ:
    circuit_graph  (本モジュール, networkx のみに依存)
        ↑              ↑                 ↑
  feature_extractor  topo_kernel  block_decomposer
        ↑
    circuit_rag / evaluate / validate_real / ...
"""

import networkx as nx


# ── 能動素子（3端子以上）の種別定義 ───────────────────────
ACTIVE_TYPES = {"NPN", "PNP", "NMOS", "PMOS", "OPAMP"}
BJT_TYPES    = {"NPN", "PNP"}
FET_TYPES    = {"NMOS", "PMOS"}
PTYPE_TYPES  = {"PNP", "PMOS"}          # p 型（極性反転）


# ── 単純パス列挙の爆発ガード ──────────────────────────────
# nx.all_simple_paths はノード数・密度に対し指数的に増え得る。トイDBでは無害でも
# 実機の密なネットリスト（ブリッジ整流＋多段等）でハングし得るため上限を設ける。
MAX_SIMPLE_PATHS = 10000
DEFAULT_PATH_CUTOFF = 20


def is_device(comp: dict) -> bool:
    """3 端子以上（=辺で表現できない）能動素子か。"""
    return comp["type"] in ACTIVE_TYPES or len(comp.get("terminals", {})) >= 3


def build_graph(circuit: dict) -> nx.Graph:
    """
    ネットリストを無向グラフ化する。

    2 端子素子（R/C/L/SW/D/DZ など）: 従来通り「ネット間の辺」として表現する。
      → 既存の 2 端子回路のグラフは一切変化しない（後方互換）。
    3 端子以上の素子（NPN/PNP/NMOS/PMOS/OPAMP など）: 辺では表せないため、
      デバイスノード `__dev_<id>` を作り、各端子ネットへ放射状に辺を張る。
      各辺には pin（端子名）を持たせ、能動素子の構成解析に用いる。
    """
    G = nx.Graph()
    for comp in circuit["components"]:
        terminals = list(comp["terminals"].values())
        if len(terminals) == 2 and comp["type"] not in ACTIVE_TYPES:
            u, v = terminals[0], terminals[1]
            G.add_edge(u, v, type=comp["type"], id=comp["id"],
                       terminals=comp["terminals"])
        else:
            dev = f"__dev_{comp['id']}"
            G.add_node(dev, is_device=True, type=comp["type"], id=comp["id"])
            for pin, net in comp["terminals"].items():
                G.add_edge(dev, net, type=comp["type"], id=comp["id"],
                           pin=pin, terminals=comp["terminals"])
    return G


def bounded_simple_paths(G: nx.Graph, source, target,
                         cutoff: int = DEFAULT_PATH_CUTOFF,
                         max_paths: int = MAX_SIMPLE_PATHS) -> list[list]:
    """
    nx.all_simple_paths をパス数上限つきで列挙する（指数爆発ガード）。

    現行 DB（疎なトイ回路）では総パス数が上限を大きく下回るため、`list(...)` と
    結果は同一になる。密な実機グラフで上限に達した場合のみ打ち切る（ハング防止）。
    source/target がグラフに無い、または到達不能なら空リストを返す。
    """
    if source not in G or target not in G:
        return []
    out: list[list] = []
    try:
        for p in nx.all_simple_paths(G, source, target, cutoff=cutoff):
            out.append(p)
            if len(out) >= max_paths:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return out
