"""知識カード(knowledge_cards.yaml)のスキーマ・参照整合の回帰ガード。

契約は docs/CARD_SPEC.md。カードは検索ヒット(=DB id)に紐づくため、
キーと confused_with.with は実在の circuit_id でなければならない。
"""
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIRED = ("name", "role", "decisive")


@pytest.fixture(scope="session")
def cards():
    with open(ROOT / "knowledge_cards.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["cards"]


def test_yaml_loads_and_nonempty(cards):
    assert isinstance(cards, dict) and cards


def test_card_keys_are_real_db_ids(cards, samples):
    for cid in cards:
        assert cid in samples, f"カード id {cid} が DB に存在しない"


def test_required_fields_present(cards):
    for cid, card in cards.items():
        for field in REQUIRED:
            assert card.get(field), f"{cid}: 必須フィールド {field} が空"
        assert isinstance(card["decisive"], list) and card["decisive"]


def test_role_is_single_sentence(cards):
    # role は一文（冗長な役割説明の禁止を機械的に担保）
    for cid, card in cards.items():
        assert card["role"].count("。") <= 1, f"{cid}: role は一文にする"


def test_card_name_matches_db(cards, samples):
    for cid, card in cards.items():
        assert card["name"] == samples[cid]["name"], (
            f"{cid}: カード name が DB と不一致")


def test_confused_with_targets_resolve(cards, samples):
    for cid, card in cards.items():
        for ref in card.get("confused_with", []):
            tgt = ref["with"]
            assert tgt in samples, f"{cid}: confused_with.with={tgt} が DB に無い"
            assert ref.get("difference"), f"{cid}->{tgt}: difference が空"
            assert tgt != cid, f"{cid}: 自分自身を confused_with に含めている"
