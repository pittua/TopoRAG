"""
ablation.py — 特徴グループの寄与分析（Phase 1 / 研究の ablation 章）

各特徴グループを順にゼロ化（ablate）し、性能低下から寄与を定量化する。
2 つのデータセットで測る:
  - 自己検索   : DB 47 回路を自分自身で検索（識別の上限性能）
  - 実機in-scope: real_corpus.json の in-scope 15 回路（汎化性能）

特徴グループ（43 次元コサインベクトル + タグ軸）:
  A   部品presence   (0-4)    R/C/L/SW/D の有無
  B1  順序           (5-8)    SW-L 順序・先頭直列部品
  B2o ダイオード向き (9-12)   アノード/カソード × GND/OUT
  B3C 構造           (13-17)  直列並列・ノード・ループ
  DZ  ツェナー       (18)
  D   能動           (19-33)  BJT/MOSFET/OpAmp 構成・帰還・差動 等
  B2r ダイオード役割 (34-37)  直列/シャント/アノード位置/平滑
  CNT 部品数         (38-42)  R/C/L/D/SW の正規化個数（sgnb で追加）
  TAG 機能タグ                alpha で重み付けされるタグ類似度（ベクトル外）

WL カーネル(sgnb)はコサインベクトルと独立なため、本 ablation は **beta=1.0
（コサインのみ）** で実施し、コサイン各グループの寄与を測る。WL カーネル自体の寄与は
evaluate.py --beta-sweep（コサインのみ 78.7% → WL ブレンド 100%）で別途評価する。
ベクトル系グループは alpha=1.0（タグ非依存）で評価し、TAG だけ alpha を振って測る。

使い方:
  python ablation.py

依存: circuit_rag.vectorize をマスク版に差し替えて DB/クエリ双方に同一マスクを適用する。
"""
from __future__ import annotations

import numpy as np

import circuit_rag
from feature_extractor import extract_hierarchical_features
from validate_real import load_circuits, load_expected, build_rag

DIM = 43
SEP = "─" * 70

# ── vectorize をマスク可能にする（DB側 add とクエリ側 search の両方に効く）──
_ORIG_VECTORIZE = circuit_rag.vectorize
_MASK = np.ones(DIM)


def _masked_vectorize(features: dict) -> np.ndarray:
    v = _ORIG_VECTORIZE(features)
    if len(v) == len(_MASK):
        return v * _MASK
    return v


circuit_rag.vectorize = _masked_vectorize


def set_mask(drop_idx=None) -> None:
    global _MASK
    _MASK = np.ones(DIM)
    if drop_idx is not None:
        for i in drop_idx:
            _MASK[i] = 0.0


GROUPS = {
    "A   部品presence (0-4)": range(0, 5),
    "B1  順序 (5-8)": range(5, 9),
    "B2o ダイオード向き (9-12)": range(9, 13),
    "B3C 構造 (13-17)": range(13, 18),
    "DZ  ツェナー (18)": [18],
    "D   能動 (19-33)": range(19, 34),
    "B2r ダイオード役割 (34-37)": range(34, 38),
    "CNT 部品数 (38-42)": range(38, 43),
}


def _ranks_self(db: list[dict], alpha: float) -> list[int]:
    rag = build_rag(db)
    n = len(db)
    ranks = []
    for c in db:
        q = extract_hierarchical_features(c)
        hits = rag.search(q, top_k=n, alpha=alpha, beta=1.0)
        r = next((h["rank"] for h in hits
                  if h["features"]["circuit_id"] == c["id"]), n)
        ranks.append(r)
    return ranks


def _ranks_real(db: list[dict], queries: list[dict], expected: dict,
                alpha: float) -> list[int]:
    rag = build_rag(db)
    n = len(db)
    ranks = []
    for q in queries:
        spec = expected.get(q["id"], {})
        if spec.get("scope") != "in":
            continue
        expect = set(spec.get("expect", []) or [])
        qf = extract_hierarchical_features(q)
        hits = rag.search(qf, top_k=n, alpha=alpha, beta=1.0)
        r = next((h["rank"] for h in hits
                  if h["features"]["circuit_id"] in expect), n)
        ranks.append(r)
    return ranks


def hit1_mrr(ranks: list[int]) -> tuple[float, float]:
    n = len(ranks)
    if n == 0:
        return 0.0, 0.0
    return sum(r == 1 for r in ranks) / n, sum(1.0 / r for r in ranks) / n


def main() -> int:
    db = load_circuits("sample_netlists.json")
    queries = load_circuits("real_corpus.json")
    expected = load_expected("real_expected.yaml")
    n_in = sum(expected.get(q["id"], {}).get("scope") == "in" for q in queries)

    # ── ベースライン（全特徴, alpha=1.0）──
    set_mask(None)
    base_self = hit1_mrr(_ranks_self(db, 1.0))
    base_real = hit1_mrr(_ranks_real(db, queries, expected, 1.0))

    print(f"\nAblation 分析  （DB={len(db)} / 実機in-scope={n_in}, alpha=1.0 beta=1.0 コサインのみ）")
    print("=" * 72)
    print(f"{'特徴グループを除外':28} | {'自己Hit@1':>9} {'自己MRR':>8} | "
          f"{'実機Hit@1':>9} {'実機MRR':>8}")
    print(SEP)
    print(f"{'(なし=ベースライン)':28} | {base_self[0]*100:8.1f}% {base_self[1]:8.3f} | "
          f"{base_real[0]*100:8.1f}% {base_real[1]:8.3f}")
    print(SEP)

    # ── 各グループを除外 ──
    rows = []
    for name, idx in GROUPS.items():
        set_mask(idx)
        s = hit1_mrr(_ranks_self(db, 1.0))
        r = hit1_mrr(_ranks_real(db, queries, expected, 1.0))
        ds = (base_self[0] - s[0]) * 100
        dr = (base_real[0] - r[0]) * 100
        rows.append((name, s, r, ds, dr))
        print(f"−{name:27} | {s[0]*100:8.1f}% {s[1]:8.3f} | "
              f"{r[0]*100:8.1f}% {r[1]:8.3f}   "
              f"(Δself {ds:+.1f} / Δreal {dr:+.1f})")
    set_mask(None)

    # ── タグ軸（自己検索のみ。実機クエリはタグ非保持で alpha 不感）──
    print(SEP)
    sweep = {}
    for step in range(11):
        a = round(step / 10.0, 1)
        sweep[a] = hit1_mrr(_ranks_self(db, a))[0]
    best_a = max(sweep, key=lambda a: sweep[a])
    tag_gain = (sweep[best_a] - sweep[1.0]) * 100
    print(f"TAG 機能タグ寄与（自己検索）: alpha=1.0 Hit@1 {sweep[1.0]*100:.1f}% → "
          f"最良 alpha={best_a} {sweep[best_a]*100:.1f}%  (タグ寄与 +{tag_gain:.1f}pt)")
    print("  ※ 実機クエリはタグ非保持のため alpha 不感（タグはDB自己検索のみに効く）")

    # ── まとめ ──
    print("\n" + "=" * 72)
    print("[寄与の大きいグループ（実機Hit@1 低下が大きい順）]")
    for name, s, r, ds, dr in sorted(rows, key=lambda x: -x[4]):
        print(f"  Δreal {dr:+5.1f}pt / Δself {ds:+5.1f}pt  : {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
