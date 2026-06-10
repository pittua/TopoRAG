"""検索・棄却の不変条件。"""
import circuit_rag
from circuit_rag import CircuitRAG, vectorize, cosine_similarity
from feature_extractor import extract_hierarchical_features


def test_self_search_hit_at_1(rag, circuits):
    """DB 各回路を自分で検索すると rank1 に自分が来る（自己検索 Hit@1=100%）。"""
    for c in circuits:
        feat = extract_hierarchical_features(c)
        hits = rag.search(feat, top_k=1)
        assert hits[0]["features"]["circuit_id"] == c["id"], (
            f"{c['id']} が自己検索で rank1 に来ない"
        )


def test_cosine_identity():
    import numpy as np
    v = np.array([1.0, 2.0, 0.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9
    assert cosine_similarity(v, np.zeros(4)) == 0.0


def test_block_match_optimal_renamed():
    # _block_match_greedy（実体は Hungarian）は _block_match_optimal に改名済み
    assert hasattr(CircuitRAG, "_block_match_optimal")
    assert not hasattr(CircuitRAG, "_block_match_greedy")


def test_reject_threshold_is_provisional():
    # 棄却閾値は「未校正の暫定値」として明示的に改名されている
    assert circuit_rag.PROVISIONAL_REJECT_THRESHOLD == 0.83
    assert not hasattr(circuit_rag, "RECOMMENDED_REJECT_THRESHOLD")


def test_search_with_rejection_shape(rag, samples):
    feat = extract_hierarchical_features(samples["rc_lowpass_001"])
    hits, accepted, conf = rag.search_with_rejection(feat)
    assert isinstance(accepted, bool)
    assert 0.0 <= conf <= 1.0
    # DB 厳密一致クエリは高スコアで受理される
    assert accepted and hits[0]["features"]["circuit_id"] == "rc_lowpass_001"
