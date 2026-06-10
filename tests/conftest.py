"""共有フィクスチャ。cwd に依存せずリポジトリ直下のデータを読む。"""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_json(name):
    with open(ROOT / name, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def circuits():
    """sample_netlists.json の回路リスト。"""
    return _load_json("sample_netlists.json")["circuits"]


@pytest.fixture(scope="session")
def samples(circuits):
    """id -> 回路 dict。"""
    return {c["id"]: c for c in circuits}


@pytest.fixture(scope="session")
def db_features():
    """features_db.json（抽出済み特徴の list）。"""
    return _load_json("features_db.json")


@pytest.fixture(scope="session")
def rag(db_features):
    """features_db.json をロードした CircuitRAG。"""
    from circuit_rag import CircuitRAG
    r = CircuitRAG()
    for rec in db_features:
        r.add(rec)
    return r
