"""構造IR(circuit_ir)のスキーマ・決定性・版の回帰ガード。"""
from circuit_ir import IR_VERSION, build_ir, render_ir
from feature_extractor import extract_all_features, extract_hierarchical_features


def _circ(comps, ports):
    return {"id": "t", "name": "t", "components": comps, "ports": ports}


RC_LOWPASS = _circ(
    [{"id": "R1", "type": "R", "terminals": {"p": "IN", "n": "OUT"}},
     {"id": "C1", "type": "C", "terminals": {"p": "OUT", "n": "GND"}}],
    {"input": "IN", "output": "OUT", "gnd": "GND"})

NPN_CE = _circ(
    [{"id": "Q1", "type": "NPN",
      "terminals": {"base": "IN", "collector": "OUT", "emitter": "GND"}}],
    {"input": "IN", "output": "OUT", "gnd": "GND"})

HALFWAVE = _circ(
    [{"id": "D1", "type": "D", "terminals": {"anode": "IN", "cathode": "OUT"}},
     {"id": "R1", "type": "R", "terminals": {"p": "OUT", "n": "GND"}}],
    {"input": "IN", "output": "OUT", "gnd": "GND"})


def test_ir_version_is_stamped():
    ir = build_ir(extract_all_features(RC_LOWPASS))
    assert ir["ir_version"] == IR_VERSION


def test_ir_is_deterministic():
    f = extract_all_features(RC_LOWPASS)
    assert build_ir(f) == build_ir(f)


def test_ir_is_json_serializable():
    import json
    json.dumps(build_ir(extract_all_features(NPN_CE)))


def test_passive_circuit_omits_diode_and_active_and_hierarchy():
    ir = build_ir(extract_hierarchical_features(RC_LOWPASS))
    assert ir["diode"] is None
    assert ir["active"] is None
    assert ir["hierarchy"] is None
    assert ir["topology"]["first_series"] == "R"
    assert ir["topology"]["shunt_to_gnd"] == ["C"]


def test_diode_section_present_for_diode_circuit():
    ir = build_ir(extract_all_features(HALFWAVE))
    assert ir["diode"] is not None
    # アノードが入力ポート＝真の整流段
    assert ir["diode"]["anode_at_input"] is True


def test_active_section_present_for_transistor_circuit():
    ir = build_ir(extract_all_features(NPN_CE))
    assert ir["active"] is not None
    assert "BJT" in ir["active"]["devices"]
    assert ir["active"]["transistor_config"] == "CE"


def test_render_includes_core_topology_lines():
    out = render_ir(build_ir(extract_all_features(RC_LOWPASS)))
    assert "部品種別" in out
    assert "先頭直列部品" in out
    assert "ループ数" in out


def test_render_omits_diode_line_for_passive():
    out = render_ir(build_ir(extract_all_features(RC_LOWPASS)))
    assert "アノード" not in out


def test_build_ir_handles_db_records(db_features):
    # features_db.json 全件で例外なく IR 化・描画でき、版が揃う
    for rec in db_features:
        ir = build_ir(rec)
        assert ir["ir_version"] == IR_VERSION
        assert isinstance(render_ir(ir), str)
