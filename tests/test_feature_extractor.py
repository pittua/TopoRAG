"""B1 経路探索・部品特徴・ベクトル次元のユニットテスト。"""
from feature_extractor import (
    build_graph, extract_b1_component_order, extract_component_features,
    extract_all_features,
)
from circuit_rag import vectorize


def _circ(comps, ports):
    return {"id": "t", "name": "t", "components": comps, "ports": ports}


def test_b1_rc_lowpass_series_shunt():
    # R 直列 → C シャント。先頭直列=R, シャント=C
    c = _circ(
        [{"id": "R1", "type": "R", "terminals": {"p": "IN", "n": "OUT"}},
         {"id": "C1", "type": "C", "terminals": {"p": "OUT", "n": "GND"}}],
        {"input": "IN", "output": "OUT", "gnd": "GND"})
    b1 = extract_b1_component_order(c, build_graph(c))
    assert b1["series_type_sequence"] == ["R"]
    assert b1["first_series_type"] == "R"
    assert b1["shunt_type_sequence"] == ["C"]


def test_b1_rc_highpass_is_distinct_from_lowpass():
    # C 直列 → R シャント（ローパスと部品集合は同じだが配置が逆）
    c = _circ(
        [{"id": "C1", "type": "C", "terminals": {"p": "IN", "n": "OUT"}},
         {"id": "R1", "type": "R", "terminals": {"p": "OUT", "n": "GND"}}],
        {"input": "IN", "output": "OUT", "gnd": "GND"})
    b1 = extract_b1_component_order(c, build_graph(c))
    assert b1["series_type_sequence"] == ["C"]
    assert b1["first_series_type"] == "C"
    assert b1["shunt_type_sequence"] == ["R"]


def test_b1_missing_port_node_no_crash():
    # 出力ポートがグラフに存在しない → 空結果（例外を投げない）
    c = _circ(
        [{"id": "R1", "type": "R", "terminals": {"p": "IN", "n": "MID"}}],
        {"input": "IN", "output": "NOPE", "gnd": "GND"})
    b1 = extract_b1_component_order(c, build_graph(c))
    assert b1["series_type_sequence"] == []
    assert b1["first_series_type"] is None


def test_component_active_flags():
    c = _circ(
        [{"id": "Q1", "type": "NPN",
          "terminals": {"base": "NB", "collector": "NC", "emitter": "GND"}}],
        {"input": "NB", "output": "NC", "gnd": "GND"})
    a = extract_component_features(c)
    assert a["has_bjt"] and a["has_active"]
    assert not a["has_mosfet"] and not a["has_opamp"]


def test_vectorize_dimension_is_43(samples):
    v = vectorize(extract_all_features(samples["rc_lowpass_001"]))
    assert len(v) == 43


def test_passive_circuit_zeroes_active_and_diode_dims(samples):
    # RC ローパス（受動・ダイオード無し）は能動次元19-33・ダイオード役割34-37が全0
    v = vectorize(extract_all_features(samples["rc_lowpass_001"]))
    assert list(v[19:38]) == [0.0] * 19
