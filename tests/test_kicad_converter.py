"""KiCad コンバータ: 能動素子の端子構築と電源レール除外。"""
from kicad_sch_to_toporag import (
    _build_terminals, _infer_ports, IN_RE, OUT_RE, POWER_RAIL_RE,
)


def test_build_terminals_bjt():
    t = _build_terminals("NPN", {"1": "C", "2": "B", "3": "E"},
                         {"1": "NC", "2": "NB", "3": "NE"})
    assert t == {"collector": "NC", "base": "NB", "emitter": "NE"}


def test_build_terminals_mosfet():
    t = _build_terminals("NMOS", {"1": "D", "2": "G", "3": "S"},
                         {"1": "ND", "2": "NG", "3": "NS"})
    assert t == {"drain": "ND", "gate": "NG", "source": "NS"}


def test_build_terminals_opamp_output_is_unroled_pin():
    # +/- は入力、V+/V- は電源（スキップ）、役割名なしの pin5 が出力
    t = _build_terminals(
        "OPAMP",
        {"1": "+", "2": "-", "3": "V+", "4": "V-"},
        {"1": "GND", "2": "NINV", "3": "VP", "4": "VN", "5": "OUT"})
    assert t == {"in_p": "GND", "in_n": "NINV", "out": "OUT"}


def test_power_rail_regex_separates_supply_from_signal():
    # 信号入力にマッチすべきもの
    assert IN_RE.search("Vsig") and IN_RE.search("In")
    # 電源レールは信号入力にマッチしない（誤って入力にされない）
    assert not IN_RE.search("Vdc")
    assert not IN_RE.search("VCC")
    # 電源レール判定
    for rail in ("Vdc", "VCC", "VDD", "V+", "V-"):
        assert POWER_RAIL_RE.search(rail), rail
    assert not POWER_RAIL_RE.search("Vsig")
    assert OUT_RE.search("Out")


def test_infer_ports_picks_signal_over_supply_rail():
    # Vsig(信号源) と Vdc(電源) が両方ある増幅器で、入力は Vsig が選ばれる
    all_nets = {"GND", "Vsig", "Vdc", "Out", "NB"}
    pin_to_net = {
        ("R1", "1"): "Vsig", ("R1", "2"): "NB",
        ("R2", "1"): "Vdc",  ("R2", "2"): "NB",
        ("R3", "1"): "Out",  ("R3", "2"): "GND",
        ("V1", "1"): "Vsig",
    }
    orig_comps = [{"ref": "V1", "sim_device": "V", "pin_role": {"1": "+", "2": "-"}}]
    ports = _infer_ports(all_nets, pin_to_net, orig_comps)
    assert ports["input"] == "Vsig"   # 電源 Vdc ではなく信号 Vsig
    assert ports["output"] == "Out"
    assert ports["gnd"] == "GND"
