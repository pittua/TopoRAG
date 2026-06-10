"""ブロック分解の境界帰属（T分岐点分割）。"""
from block_decomposer import decompose_blocks


def test_single_rc_is_one_block(samples):
    assert len(decompose_blocks(samples["rc_lowpass_001"])) == 1


def test_rc_cascade_two_blocks(samples):
    blocks = decompose_blocks(samples["rc_cascade_lpf_001"])
    assert len(blocks) == 2
    # 各ブロックは R 直列 + C シャントの 1 段
    for b in blocks:
        types = sorted(c["type"] for c in b["components"])
        assert types == ["C", "R"]


def test_buck_lc_filter_three_blocks_shunt_attribution(samples):
    # {SW,D} + {L,C} + {L,C}。区間終端のシャント C は前段ブロックに帰属する。
    blocks = decompose_blocks(samples["buck_lc_filter_001"])
    assert len(blocks) == 3
    block_types = [sorted(c["type"] for c in b["components"]) for b in blocks]
    assert block_types[0] == ["D", "SW"]
    assert block_types[1] == ["C", "L"]
    assert block_types[2] == ["C", "L"]


def test_active_circuit_not_fragmented(samples):
    # 能動段はバイアス節点で分断しない（単一ブロック）
    assert len(decompose_blocks(samples["ce_amp_npn_001"])) == 1
    assert len(decompose_blocks(samples["two_stage_ce_001"])) == 1
