"""能動素子の構成分類ヒューリスティック（_classify_transistor / _classify_opamp）。

DB サンプル回路で接地構成・帰還・差動/カレントミラーの判定を固定する。
分類ロジックを触ったときの回帰検出が目的。
"""
import pytest

from feature_extractor import extract_all_features


def _d(samples, cid):
    return extract_all_features(samples[cid])["D_active"]


@pytest.mark.parametrize("cid,cfg", [
    ("ce_amp_npn_001", "CE"),            # エミッタ接地 = 反転増幅
    ("ce_amp_pnp_001", "CE"),
    ("emitter_follower_npn_001", "CC"),  # コレクタ接地 = フォロワ
    ("cs_amp_nmos_001", "CE"),           # ソース接地（CS）も CE に正規化
    ("source_follower_nmos_001", "CC"),
])
def test_transistor_config(samples, cid, cfg):
    assert _d(samples, cid)["transistor_config"] == cfg


@pytest.mark.parametrize("cid,cfg", [
    ("opamp_inverting_001", "inverting"),
    ("opamp_noninverting_001", "non_inverting"),
    ("opamp_buffer_001", "buffer"),
])
def test_opamp_config(samples, cid, cfg):
    d = _d(samples, cid)
    assert d["opamp_config"] == cfg
    assert d["has_feedback"] is True


def test_pnp_is_p_type(samples):
    assert _d(samples, "ce_amp_pnp_001")["p_type"] is True
    assert _d(samples, "ce_amp_npn_001")["p_type"] is False


def test_current_mirror_diode_connected(samples):
    d = _d(samples, "current_mirror_npn_001")
    assert d["has_diode_connected"] is True
    assert d["has_coupled_pair"] is False


def test_differential_pair(samples):
    d = _d(samples, "diff_pair_npn_001")
    assert d["has_coupled_pair"] is True
    assert d["is_differential"] is True
    assert d["has_diode_connected"] is False
