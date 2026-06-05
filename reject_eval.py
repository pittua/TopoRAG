"""
reject_eval.py — 棄却シグナルの分離性能評価（Phase 0-B）

背景:
  自作 DB は自己類似でスコア 1.0、実機クエリはタグ無しでトポロジーのみ ≤ alpha に
  なるため、両者は別分布で単一の絶対閾値 θ は原理的に不適（Phase 0-A の知見）。
  本スクリプトは「未知回路（out-of-scope）を棄却できるか」を OOD 検知問題として捉え、
  タグ非依存（トポロジーのみ）で複数の信頼度シグナルを比較する。

評価対象シグナル（in-scope=受理クラス, out-of-scope=棄却クラス）:
  - top1        : top-1 トポロジースコア（高いほど in-scope らしい）
  - margin      : top1 - top2（上位の突出度。高いほど確信＝in-scope らしい）
  - ratio_gap   : 1 - top2/top1（同上を相対化）

各シグナルについて:
  - AUC（in を out より高くランクできる確率＝Mann-Whitney U / (n_in·n_out)）
  - balanced accuracy を最大化する閾値と、その混同行列

使い方:
  python reject_eval.py
  python reject_eval.py --corpus real_corpus.json --expected real_expected.yaml

設計: docs/EVALUATION_DESIGN.md §7「真の未知回路テスト」への回答（Phase 0-B）
"""
from __future__ import annotations

import argparse

from feature_extractor import extract_hierarchical_features
from circuit_rag import CircuitRAG
from validate_real import load_circuits, load_expected, build_rag

SEP = "─" * 64


def auc(pos: list[float], neg: list[float]) -> float:
    """pos（in-scope）が neg（out-of-scope）より高い確率。0.5=分離不能, 1.0=完全分離。"""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float, dict]:
    """
    「signal >= θ なら受理」で balanced accuracy を最大化する θ を返す。
    返り値: (θ, balanced_acc, {tp,fn,tn,fp})
    """
    cands = sorted(set(pos + neg))
    # 閾値候補は各値の少し下も含めるため、隣接中点も加える
    mids = [(a + b) / 2 for a, b in zip(cands, cands[1:])]
    cands = sorted(set(cands + mids + [min(cands) - 1e-9, max(cands) + 1e-9]))

    best = (cands[0], -1.0, {})
    for th in cands:
        tp = sum(p >= th for p in pos)   # in を受理
        fn = len(pos) - tp               # in を誤棄却
        tn = sum(n < th for n in neg)    # out を棄却
        fp = len(neg) - tn               # out を誤受理
        tpr = tp / len(pos) if pos else 0.0
        tnr = tn / len(neg) if neg else 0.0
        bacc = 0.5 * (tpr + tnr)
        if bacc > best[1]:
            best = (th, bacc, {"tp": tp, "fn": fn, "tn": tn, "fp": fp})
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="TopoRAG 棄却シグナル評価")
    ap.add_argument("--corpus", default="real_corpus.json")
    ap.add_argument("--expected", default="real_expected.yaml")
    ap.add_argument("--db", default="sample_netlists.json")
    args = ap.parse_args()

    db = load_circuits(args.db)
    rag = build_rag(db)
    queries = load_circuits(args.corpus)
    expected = load_expected(args.expected)

    # タグ非依存にするため alpha=1.0（トポロジーのみ）で評価する
    rows = []  # (qid, scope, top1, margin, ratio_gap)
    for q in queries:
        spec = expected.get(q["id"], {})
        scope = spec.get("scope")
        if scope not in ("in", "out"):
            continue
        qf = extract_hierarchical_features(q)
        hits = rag.search(qf, top_k=2, alpha=1.0)
        top1 = hits[0]["score"]
        top2 = hits[1]["score"] if len(hits) > 1 else 0.0
        margin = top1 - top2
        ratio_gap = (1.0 - top2 / top1) if top1 > 0 else 0.0
        rows.append((q["id"], scope, top1, margin, ratio_gap))

    n_in = sum(r[1] == "in" for r in rows)
    n_out = sum(r[1] == "out" for r in rows)
    print(f"\n棄却評価（トポロジーのみ alpha=1.0）  in-scope={n_in}  out-of-scope={n_out}")
    print("=" * 66)

    # 一覧（top1 降順）
    print(f"\n{'query':22} {'scope':4} {'top1':>7} {'margin':>7} {'ratio_gap':>9}")
    for qid, scope, t1, mg, rg in sorted(rows, key=lambda r: -r[2]):
        print(f"{qid:22} {scope:4} {t1:7.4f} {mg:7.4f} {rg:9.4f}")

    # 各シグナルの分離性能
    signals = {
        "top1 (絶対スコア)": 2,
        "margin (top1-top2)": 3,
        "ratio_gap (1-top2/top1)": 4,
    }
    print("\n" + "=" * 66)
    print("[シグナル別 分離性能]")
    for name, idx in signals.items():
        pos = [r[idx] for r in rows if r[1] == "in"]
        neg = [r[idx] for r in rows if r[1] == "out"]
        a = auc(pos, neg)
        th, bacc, cm = best_threshold(pos, neg)
        print(f"\n● {name}")
        print(f"   AUC = {a:.3f}   （0.5=分離不能 / 1.0=完全分離）")
        print(f"   最良閾値 θ = {th:.4f} で受理 → balanced acc = {bacc:.3f}")
        print(f"   受理: in {cm['tp']}/{len(pos)}（誤棄却 {cm['fn']}）  "
              f"棄却: out {cm['tn']}/{len(neg)}（誤受理 {cm['fp']}）")

    # 旧方式（LOO θ=0.9786 をタグ付き絶対スコアに適用）の破綻も併記
    print("\n" + "=" * 66)
    print("[参考] 旧 LOO 閾値 θ=0.9786 を実機 top1(alpha=0.9) に適用すると…")
    rej_all = 0
    for q in queries:
        spec = expected.get(q["id"], {})
        if spec.get("scope") not in ("in", "out"):
            continue
        qf = extract_hierarchical_features(q)
        s = rag.search(qf, top_k=1, alpha=0.9)[0]["score"]
        if s < 0.9786:
            rej_all += 1
    print(f"   {rej_all}/{n_in + n_out} 件が棄却される（in-scope 含め大半を誤棄却＝実用不能）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
