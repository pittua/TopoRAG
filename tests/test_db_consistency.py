"""features_db.json が sample_netlists.json の再抽出と一致することを保証する。

生成物 features_db.json をコミットしつつ評価系がその場再抽出も行うため、
再生成忘れで両者が無言で乖離し得る（監査指摘 #7）。本テストがその乖離を CI で検出する。
不一致時は `python feature_extractor.py` で再生成すること。
"""
import json

from feature_extractor import extract_hierarchical_features


def _norm(obj):
    # tuple↔list 等の差を JSON 経由で正規化して厳密比較する
    return json.loads(json.dumps(obj, ensure_ascii=False))


def test_features_db_matches_reextraction(circuits, db_features):
    assert len(db_features) == len(circuits), "回路数が features_db と sample で不一致"
    reextracted = [_norm(extract_hierarchical_features(c)) for c in circuits]
    for committed, fresh in zip(db_features, reextracted):
        assert committed["circuit_id"] == fresh["circuit_id"]
        assert _norm(committed) == fresh, (
            f"{fresh['circuit_id']}: features_db.json が古い。"
            "`python feature_extractor.py` で再生成してください。"
        )
