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
import math

import numpy as np

from feature_extractor import extract_hierarchical_features
from circuit_rag import CircuitRAG
from validate_real import load_circuits, load_expected, build_rag

SEP = "─" * 64

# 統計の再現性のため乱数シードは固定（タスク指定 = 42）
SEED = 42
N_BOOT = 1000  # ブートストラップ反復回数


def auc(pos: list[float], neg: list[float]) -> float:
    """pos（in-scope）が neg（out-of-scope）より高い確率。0.5=分離不能, 1.0=完全分離。"""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def auc_bootstrap_ci(
    pos: list[float], neg: list[float], n_boot: int = N_BOOT, seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    層化ブートストラップで AUC の (1-alpha) 信頼区間を返す。
    in / out をそれぞれ独立に復元抽出（クラス比を保つ＝層化）して AUC を再計算し、
    パーセンタイル法で下限・上限を取る。n_out=6 と小標本なので点推定 AUC だけの
    断定は過大主張になる（CI 併記が必須）。
    """
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    pos_a = np.asarray(pos, dtype=float)
    neg_a = np.asarray(neg, dtype=float)
    n_p, n_n = len(pos_a), len(neg_a)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        bp = pos_a[rng.integers(0, n_p, n_p)]
        bn = neg_a[rng.integers(0, n_n, n_n)]
        stats[b] = auc(bp.tolist(), bn.tolist())
    lo = float(np.percentile(stats, 100 * (alpha / 2)))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """
    二項比率 k/n の Wilson スコア 95%CI（既定 z=1.96）。
    小標本でも 0/1 付近で破綻しない（正規近似より頑健）。
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def loo_logreg_scores(
    X: np.ndarray, y: np.ndarray, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-One-Out で各サンプルの out-of-fold 予測確率（in-scope=陽性クラスの確率）を返す。
    特徴は LOO の訓練 fold 統計で標準化し、リークを避ける。
    返り値: (proba_oof, y) — proba_oof[i] は i 番目を抜いて学習したモデルの予測。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut

    proba = np.empty(len(y), dtype=float)
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        # in-scope を陽性(=1)として確率を取り出す
        cls = list(clf.classes_)
        pi = cls.index(1) if 1 in cls else 0
        proba[te[0]] = clf.predict_proba(scaler.transform(X[te]))[0, pi]
    return proba, y


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


def nested_loo_bacc(pos_idx: list[float], neg_idx: list[float]) -> tuple[float, dict]:
    """
    ネスト（外側 LOO）交差検証で「楽観バイアスを除いた」balanced accuracy を返す。
    手順: 外側で 1 件抜き → 残りデータだけで best_threshold（閾値選択）→ 抜いた 1 件を判定。
    これを全件で回し、外側で集計した混同行列から bacc を算出する。
    「同一データで閾値選択し同一データで報告」する naive 値は楽観的に膨らむため、
    汎化性能の正直な推定としてこちらを併記する。
    """
    pos = list(pos_idx)
    neg = list(neg_idx)
    tp = fn = tn = fp = 0

    # in-scope を 1 件ずつ外す（残りで閾値選択 → 抜いた in を判定）
    for i in range(len(pos)):
        held = pos[i]
        tr_pos = pos[:i] + pos[i + 1:]
        if not tr_pos or not neg:
            continue
        th, _, _ = best_threshold(tr_pos, neg)
        if held >= th:
            tp += 1   # in を正しく受理
        else:
            fn += 1   # in を誤棄却

    # out-of-scope を 1 件ずつ外す（残りで閾値選択 → 抜いた out を判定）
    for j in range(len(neg)):
        held = neg[j]
        tr_neg = neg[:j] + neg[j + 1:]
        if not pos or not tr_neg:
            continue
        th, _, _ = best_threshold(pos, tr_neg)
        if held < th:
            tn += 1   # out を正しく棄却
        else:
            fp += 1   # out を誤受理

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bacc = 0.5 * (tpr + tnr)
    return bacc, {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "tpr": tpr, "tnr": tnr}


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
        a_lo, a_hi = auc_bootstrap_ci(pos, neg)
        th, bacc, cm = best_threshold(pos, neg)
        # 最良閾値での TPR/TNR に Wilson 95%CI（小標本なので必須）
        tpr_lo, tpr_hi = wilson_ci(cm["tp"], len(pos))
        tnr_lo, tnr_hi = wilson_ci(cm["tn"], len(neg))
        print(f"\n● {name}")
        print(f"   AUC = {a:.3f}   95%CI [{a_lo:.3f}, {a_hi:.3f}]"
              f"   （層化ブートストラップ {N_BOOT}回, seed={SEED}）")
        print(f"      ※0.5=分離不能 / 1.0=完全分離。CI が 0.5 を跨ぐと有意な分離とは言えない")
        print(f"   最良閾値 θ = {th:.4f} で受理 → balanced acc = {bacc:.3f}（naive）")
        print(f"   受理: in {cm['tp']}/{len(pos)}（誤棄却 {cm['fn']}）  "
              f"棄却: out {cm['tn']}/{len(neg)}（誤受理 {cm['fp']}）")
        print(f"      TPR(in受理率) = {cm['tp']}/{len(pos)} = {cm['tp']/len(pos):.3f}"
              f"  Wilson95%CI [{tpr_lo:.3f}, {tpr_hi:.3f}]")
        print(f"      TNR(out棄却率) = {cm['tn']}/{len(neg)} = {cm['tn']/len(neg):.3f}"
              f"  Wilson95%CI [{tnr_lo:.3f}, {tnr_hi:.3f}]")

    # ── [1] 複数シグナル融合: top1 + margin の2次元を LogisticRegression で融合 ──
    # LOO 交差検証で out-of-fold 予測確率を出し、単一シグナルより分離が上がるか検証。
    print("\n" + "=" * 66)
    print("[複数シグナル融合]  top1 + margin → LogisticRegression（LOO交差検証）")
    X = np.array([[r[2], r[3]] for r in rows], dtype=float)   # [top1, margin]
    y = np.array([1 if r[1] == "in" else 0 for r in rows], dtype=int)
    proba, _ = loo_logreg_scores(X, y)
    fpos = [float(p) for p, lab in zip(proba, y) if lab == 1]
    fneg = [float(p) for p, lab in zip(proba, y) if lab == 0]
    f_auc = auc(fpos, fneg)
    f_lo, f_hi = auc_bootstrap_ci(fpos, fneg)
    f_th, f_bacc, f_cm = best_threshold(fpos, fneg)
    ftpr_lo, ftpr_hi = wilson_ci(f_cm["tp"], len(fpos))
    ftnr_lo, ftnr_hi = wilson_ci(f_cm["tn"], len(fneg))
    # 単一シグナルの最良 AUC（top1/margin/ratio_gap のうち最大）と比較する
    single_aucs = {
        nm: auc([r[ix] for r in rows if r[1] == "in"],
                [r[ix] for r in rows if r[1] == "out"])
        for nm, ix in signals.items()
    }
    best_single = max(single_aucs, key=single_aucs.get)
    print(f"\n● 融合（top1+margin, LOO out-of-fold 予測確率）")
    print(f"   AUC = {f_auc:.3f}   95%CI [{f_lo:.3f}, {f_hi:.3f}]"
          f"   （層化ブートストラップ {N_BOOT}回, seed={SEED}）")
    print(f"   最良閾値で balanced acc = {f_bacc:.3f}（naive, LOO予測確率上で選択）")
    print(f"      TPR = {f_cm['tp']}/{len(fpos)} = {f_cm['tp']/len(fpos):.3f}"
          f"  Wilson95%CI [{ftpr_lo:.3f}, {ftpr_hi:.3f}]")
    print(f"      TNR = {f_cm['tn']}/{len(fneg)} = {f_cm['tn']/len(fneg):.3f}"
          f"  Wilson95%CI [{ftnr_lo:.3f}, {ftnr_hi:.3f}]")
    print(f"   比較: 単一シグナル最良 = {best_single}（AUC {single_aucs[best_single]:.3f}）"
          f"  →  融合 AUC {f_auc:.3f}  "
          f"（{'+' if f_auc >= single_aucs[best_single] else ''}{f_auc - single_aucs[best_single]:.3f}）")
    verdict = "分離が向上" if f_auc > single_aucs[best_single] + 1e-9 else (
        "ほぼ同等" if abs(f_auc - single_aucs[best_single]) <= 1e-9 else "むしろ低下")
    print(f"   → 2次元融合は単一シグナル最良に対し {verdict}")

    # ── [2] ネスト交差検証: 楽観バイアスを除いた「正直な」棄却性能 ──
    # 各シグナルについて naive（同一データで閾値選択&報告）と nested（外側LOOで分離）を比較。
    print("\n" + "=" * 66)
    print("[ネスト交差検証]  外側LOOで閾値選択と評価を分離 → 最適化バイアス除去")
    print("  naive  : 全データで閾値選択し同一データで報告（楽観的に膨らむ）")
    print("  nested : 1件抜き→残りで閾値選択→抜いた1件を判定（汎化性能の正直な推定）")
    print(f"\n  {'シグナル':28} {'naive bacc':>11} {'nested bacc':>12} {'バイアス':>9}")
    nested_fusion_input = []  # 融合も後段で評価するため確保
    for name, idx in signals.items():
        pos = [r[idx] for r in rows if r[1] == "in"]
        neg = [r[idx] for r in rows if r[1] == "out"]
        _, naive_bacc, _ = best_threshold(pos, neg)
        nest_bacc, ncm = nested_loo_bacc(pos, neg)
        print(f"  {name:28} {naive_bacc:>11.3f} {nest_bacc:>12.3f} "
              f"{naive_bacc - nest_bacc:>+9.3f}")
    # 融合シグナル（LOO予測確率）にも同じ外側LOO閾値分離を適用
    fnaive_bacc = f_bacc
    fnest_bacc, fncm = nested_loo_bacc(fpos, fneg)
    print(f"  {'融合 top1+margin (LR)':28} {fnaive_bacc:>11.3f} {fnest_bacc:>12.3f} "
          f"{fnaive_bacc - fnest_bacc:>+9.3f}")
    print(f"\n  ※ナイーブ−ネストの正の差 = 同一データ閾値選択による楽観バイアスの大きさ。")
    print(f"    ネスト bacc が棄却機構の汎化性能（外部データに対する正直な見積り）。")

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
