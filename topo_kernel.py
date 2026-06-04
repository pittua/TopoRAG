"""
topo_kernel.py — 回路グラフの構造類似度カーネル (sgnb ブランチで追加)

Weisfeiler-Lehman (WL) サブツリーカーネルで回路トポロジーの構造的類似度を測る。

背景:
  既存の vectorize() は 38 次元の手設計ベクトル + コサイン類似度で、
  「部品の有無・粗い直列順序・ダイオード役割」など集約済みの特徴しか見ない。
  接続の細部（どの部品がどのネットを共有し、能動素子のどの端子がどこに
  繋がるか）は失われる。この平坦化が「トポロジーのみ Hit@1 77.4%」
  （7件取り違え）の主因。

  WL カーネルは回路グラフの各ノードに「役割 + 局所接続」の初期ラベルを与え、
  近傍ラベルを反復集約することで、半径 h ホップのサブツリーパターンの
  ヒストグラムを構築する。ポート役割(IN/OUT/GND)と能動素子の端子(pin)を
  ラベルに織り込むため、同じ部品集合でも接続の向き・位置が違う回路
  (RC ローパス↔ハイパス、整流の極性、CE↔CC 構成) を構造レベルで分離できる。

設計:
  - wl_histogram(circuit): 回路 → WL サブツリー特徴の疎ヒストグラム(dict)。
    features dict に "wl_features" として埋め込み、features_db.json に保存される
    （検索時に毎回グラフを再構築せず内積で高速照合するため）。
  - wl_kernel(h1, h2): 2 ヒストグラムの正規化線形カーネル(コサイン, 0..1)。
  circuit_rag._topo_similarity / search が beta でコサインとブレンドする。

後方互換:
  wl_features を持たない旧 features_db.json でも、circuit_rag 側が
  「wl_features が無ければコサインのみ」にフォールバックする。
"""

import math
from collections import Counter

from feature_extractor import build_graph

WL_ITERATIONS = 3


def _edge_label(data: dict) -> str:
    """辺ラベル = 部品種別(+能動素子の端子役割 pin)。"""
    t = str(data.get("type", "?"))
    pin = data.get("pin")
    return f"{t}.{pin}" if pin else t


def _node_role(node, ports: dict, G) -> str:
    """ノードの役割ラベル。ポート(IN/OUT/GND)と能動素子デバイスノードを区別する。"""
    if isinstance(node, str) and node.startswith("__dev_"):
        return "DEV:" + str(G.nodes[node].get("type", "?"))
    if node == ports.get("gnd"):
        return "GND"
    if node == ports.get("input"):
        return "IN"
    if node == ports.get("input2"):
        return "IN2"
    if node == ports.get("output"):
        return "OUT"
    return "NET"


def _initial_labels(G, ports: dict) -> dict:
    """初期ノードラベル = 役割 + 接続辺ラベルの整列マルチセット。"""
    labels = {}
    for n in G.nodes():
        role = _node_role(n, ports, G)
        inc = sorted(_edge_label(d) for _, _, d in G.edges(n, data=True))
        labels[n] = f"{role}[{','.join(inc)}]"
    return labels


def wl_histogram(circuit: dict, n_iter: int = WL_ITERATIONS, G=None) -> dict:
    """
    回路の WL サブツリー特徴ヒストグラム {ラベル: 出現数} を返す。
    各反復で全ノードのラベルを近傍集約し、全反復のラベル分布を加算する
    （半径 0..n_iter ホップのサブツリーを全て特徴として数える）。
    """
    if G is None:
        G = build_graph(circuit)
    ports = circuit.get("ports", {})
    if G.number_of_nodes() == 0:
        return {}

    labels = _initial_labels(G, ports)
    hist: Counter = Counter(labels.values())

    for _ in range(n_iter):
        new_labels = {}
        for n in G.nodes():
            neigh = []
            for m in G.neighbors(n):
                d = G.get_edge_data(n, m) or {}
                neigh.append(f"{_edge_label(d)}>{labels[m]}")
            new_labels[n] = labels[n] + "#" + "|".join(sorted(neigh))
        labels = new_labels
        hist.update(labels.values())

    return dict(hist)


def wl_kernel(h1: dict, h2: dict) -> float:
    """2 つの WL ヒストグラムの正規化線形カーネル（コサイン, 0..1）。"""
    if not h1 or not h2:
        return 0.0
    # 小さい方を走査して内積を取る
    if len(h1) > len(h2):
        h1, h2 = h2, h1
    dot = 0.0
    for k, v in h1.items():
        w = h2.get(k)
        if w:
            dot += v * w
    if dot == 0.0:
        return 0.0
    n1 = math.sqrt(sum(v * v for v in h1.values()))
    n2 = math.sqrt(sum(v * v for v in h2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


if __name__ == "__main__":
    # 動作確認: RC ローパス vs RC ハイパス（同じ部品 R,C・接続が逆）が
    # 構造カーネルで分離されることを示す。
    rc_lp = {
        "id": "lp", "name": "RC LP",
        "components": [
            {"id": "R1", "type": "R", "terminals": {"p": "IN", "n": "OUT"}},
            {"id": "C1", "type": "C", "terminals": {"p": "OUT", "n": "GND"}},
        ],
        "ports": {"input": "IN", "output": "OUT", "gnd": "GND"},
    }
    rc_hp = {
        "id": "hp", "name": "RC HP",
        "components": [
            {"id": "C1", "type": "C", "terminals": {"p": "IN", "n": "OUT"}},
            {"id": "R1", "type": "R", "terminals": {"p": "OUT", "n": "GND"}},
        ],
        "ports": {"input": "IN", "output": "OUT", "gnd": "GND"},
    }
    h_lp = wl_histogram(rc_lp)
    h_hp = wl_histogram(rc_hp)
    print(f"WL(LP,LP) = {wl_kernel(h_lp, h_lp):.3f}  (自己=1.0期待)")
    print(f"WL(LP,HP) = {wl_kernel(h_lp, h_hp):.3f}  (1.0未満＝構造差を検出)")
    print(f"WL特徴数: LP={len(h_lp)}  HP={len(h_hp)}")
