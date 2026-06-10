"""circuit_graph: build_graph / is_device / bounded_simple_paths のガード。"""
import networkx as nx

from circuit_graph import build_graph, is_device, bounded_simple_paths


def _rc():
    return {
        "id": "rc", "name": "rc",
        "components": [
            {"id": "R1", "type": "R", "terminals": {"p": "IN", "n": "OUT"}},
            {"id": "C1", "type": "C", "terminals": {"p": "OUT", "n": "GND"}},
        ],
        "ports": {"input": "IN", "output": "OUT", "gnd": "GND"},
    }


def test_is_device():
    assert is_device({"type": "NPN", "terminals": {"b": "1", "c": "2", "e": "3"}})
    assert is_device({"type": "OPAMP", "terminals": {"in_p": "1", "in_n": "2", "out": "3"}})
    assert not is_device({"type": "R", "terminals": {"p": "1", "n": "2"}})
    # 種別が受動でも端子3本以上ならデバイス扱い
    assert is_device({"type": "X", "terminals": {"a": "1", "b": "2", "c": "3"}})


def test_build_graph_two_terminal_is_edge():
    G = build_graph(_rc())
    # ネットノードのみ（デバイスノードなし）
    assert set(G.nodes()) == {"IN", "OUT", "GND"}
    assert G.has_edge("IN", "OUT")
    assert G["IN"]["OUT"]["type"] == "R"


def test_build_graph_three_terminal_is_device_node():
    circuit = {
        "id": "q", "name": "q",
        "components": [
            {"id": "Q1", "type": "NPN",
             "terminals": {"base": "NB", "collector": "NC", "emitter": "NE"}},
        ],
        "ports": {"input": "NB", "output": "NC", "gnd": "NE"},
    }
    G = build_graph(circuit)
    assert "__dev_Q1" in G.nodes()
    # デバイスノードから3端子ネットへ放射状の辺、各辺に pin ラベル
    pins = {G["__dev_Q1"][n]["pin"] for n in ("NB", "NC", "NE")}
    assert pins == {"base", "collector", "emitter"}


def test_bounded_simple_paths_basic():
    G = build_graph(_rc())
    paths = bounded_simple_paths(G, "IN", "GND", cutoff=10)
    assert ["IN", "OUT", "GND"] in paths


def test_bounded_simple_paths_missing_node():
    G = build_graph(_rc())
    assert bounded_simple_paths(G, "IN", "NOPE") == []


def test_bounded_simple_paths_caps_explosion():
    # 完全グラフは単純パス数が階乗的に増える。max_paths で打ち切られること。
    G = nx.complete_graph(8)
    capped = bounded_simple_paths(G, 0, 7, cutoff=8, max_paths=5)
    assert len(capped) == 5
