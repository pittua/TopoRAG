"""
validate_real.py — 実機回路コーパスによる汎化検証（Phase 0-A）

目的:
  自作 sample_netlists.json（DB）に対し、第三者が描いた実機回路
  （KiCad 同梱 demos / feastorg サンプル等）をクエリとして投入し、
  - in-scope 回路を正しく rank 1 で同定できるか（Hit@1 / Hit@3）
  - 能動素子の構成分類（CE/CC/差動/カレントミラー/OpAmp構成）が
    実機回路でも崩れないか
  - out-of-scope 回路（DB 未収録）のスコアが in-scope と分離可能か
    （= 棄却閾値 θ が機能しうるか／Phase 0-B への布石）
  を測定する。

入力:
  real_corpus.json   … kicad_sch_to_toporag.py で変換した実機回路
                       （{"circuits": [...]}; id はファイル名スラグ）
  real_expected.yaml … 各クエリの正解ラベル（scope/expect/note）
  sample_netlists.json … 検索対象 DB

使い方:
  python validate_real.py
  python validate_real.py --corpus real_corpus.json --alpha 0.9
  python validate_real.py --sim         # 受動回路はシミュレーション種別も表示

設計: docs/EVALUATION_DESIGN.md（§3 実機検証は本スクリプトで担当）
"""
from __future__ import annotations

import json
import argparse

from feature_extractor import extract_hierarchical_features
from circuit_rag import CircuitRAG

CORPUS_PATH = "real_corpus.json"
EXPECTED_PATH = "real_expected.yaml"
DB_PATH = "sample_netlists.json"

SEP = "─" * 64


def load_circuits(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["circuits"]


def build_rag(db_circuits: list[dict]) -> CircuitRAG:
    rag = CircuitRAG()
    for c in db_circuits:
        rag.add(extract_hierarchical_features(c))
    return rag


def load_expected(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("⚠ pyyaml 未導入のため正解照合をスキップ（pip install pyyaml）")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"⚠ {path} が無いため正解照合をスキップ（top-3 のみ表示）")
        return {}


def active_signature(feat: dict) -> str:
    """D_active から構成診断の1行サマリを作る。受動回路は空文字。"""
    d = feat.get("D_active", {})
    if not d.get("has_active"):
        return ""
    parts = []
    if d.get("has_bjt"):
        parts.append("BJT")
    if d.get("has_mosfet"):
        parts.append("MOS")
    if d.get("has_opamp"):
        parts.append("OpAmp")
    tcfg = d.get("transistor_config")
    ocfg = d.get("opamp_config")
    if tcfg:
        parts.append(f"tcfg={tcfg}")
    if ocfg:
        parts.append(f"ocfg={ocfg}")
    if d.get("has_feedback"):
        parts.append("fb")
    if d.get("is_differential"):
        parts.append("diff")
    if d.get("has_diode_connected"):
        parts.append("mirror?")
    if d.get("p_type"):
        parts.append("p-type")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="TopoRAG 実機コーパス汎化検証")
    ap.add_argument("--corpus", default=CORPUS_PATH)
    ap.add_argument("--expected", default=EXPECTED_PATH)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--alpha", "-a", type=float, default=0.9,
                    help="トポロジー重み（実機クエリはタグ無しのため既定0.9）")
    ap.add_argument("--top-k", "-k", type=int, default=3)
    ap.add_argument("--sim", action="store_true",
                    help="受動回路のシミュレーション種別も表示")
    args = ap.parse_args()

    db = load_circuits(args.db)
    rag = build_rag(db)
    queries = load_circuits(args.corpus)
    expected = load_expected(args.expected)

    sim_cls = None
    if args.sim:
        try:
            from circuit_simulator import CircuitSimulator
            sim_cls = CircuitSimulator
        except ImportError:
            print("⚠ circuit_simulator 不在のため --sim を無視")

    print(f"\nDB={len(db)}回路  クエリ={len(queries)}回路  alpha={args.alpha}")
    print("=" * 66)

    in_ranks: list[int] = []          # in-scope クエリの正解順位
    in_total = 0
    out_top1_scores: list[float] = []  # out-of-scope クエリの top1 スコア
    in_top1_scores: list[float] = []   # in-scope で正解した場合の top1 スコア
    rows = []

    for q in queries:
        qid = q["id"]
        qf = extract_hierarchical_features(q)
        hits = rag.search(qf, top_k=max(args.top_k, 3), alpha=args.alpha)

        spec = expected.get(qid, {})
        scope = spec.get("scope")
        expect_ids = set(spec.get("expect", []) or [])

        top1 = hits[0]
        top1_id = top1["features"]["circuit_id"]
        top1_score = top1["score"]

        # 正解判定
        verdict = ""
        if scope == "in":
            in_total += 1
            rank = next((h["rank"] for h in hits
                         if h["features"]["circuit_id"] in expect_ids), 99)
            in_ranks.append(rank)
            verdict = "✓ Hit@1" if rank == 1 else (
                f"△ Hit@{rank}" if rank <= args.top_k else "✗ miss")
            if rank == 1:
                in_top1_scores.append(top1_score)
        elif scope == "out":
            out_top1_scores.append(top1_score)
            verdict = f"out-of-scope (top1={top1_score:.3f})"

        # 表示
        print(f"\n### {q['name']}  [{scope or '未ラベル'}]  {verdict}")
        sig = active_signature(qf)
        if sig:
            print(f"  能動構成: {sig}")
        if expect_ids:
            print(f"  期待: {sorted(expect_ids)}")
        if spec.get("note"):
            print(f"  注: {spec['note']}")
        if sim_cls is not None and not sig:
            try:
                r = sim_cls(q).extract_simulation_features()
                flags = [k for k in ("is_lowpass", "is_highpass", "is_bandpass",
                                      "is_bandstop") if r.get(k)]
                print(f"  sim: {r.get('simulation_type')} flags={flags} "
                      f"fc={r.get('cutoff_freq_hz')}")
            except Exception as e:
                print(f"  sim: 失敗 ({e})")
        for h in hits[:args.top_k]:
            mark = "←" if h["features"]["circuit_id"] in expect_ids else " "
            print(f"    [{h['rank']}] {h['score']:.4f} {mark} "
                  f"{h['features']['circuit_name']}")
        rows.append((qid, scope, verdict))

    # 集計
    print("\n" + "=" * 66)
    print("[集計]")
    if in_total:
        hit1 = sum(r == 1 for r in in_ranks)
        hit3 = sum(r <= 3 for r in in_ranks)
        mrr = sum(1.0 / r for r in in_ranks) / in_total
        print(f"  in-scope: {in_total}件  "
              f"Hit@1={hit1}/{in_total}={hit1/in_total*100:.1f}%  "
              f"Hit@3={hit3}/{in_total}={hit3/in_total*100:.1f}%  MRR={mrr:.3f}")
    if out_top1_scores:
        print(f"  out-of-scope: {len(out_top1_scores)}件  "
              f"top1スコア max={max(out_top1_scores):.4f} "
              f"min={min(out_top1_scores):.4f}")
    if in_top1_scores and out_top1_scores:
        sep = min(in_top1_scores) - max(out_top1_scores)
        print(f"  分離余裕(min in-scope正解スコア − max out-scopeスコア): {sep:+.4f}")
        print("    → 正なら単一閾値で in/out を分離可能。負なら重なりあり（θ校正の限界）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
