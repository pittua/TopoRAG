"""
evaluate.py — TopoRAG 評価スクリプト

設計: EVALUATION_DESIGN.md

検索段（ベクトル検索）と LLM 段を分離して評価する。

  [Section 1] 自己検索テスト        … 常時実行（Hit@1 / Hit@3 / MRR）
  [Section 2] Alpha グリッドサーチ  … --alpha-sweep
  [Section 3] 摂動ロバスト性テスト  … 常時実行（ノード名・部品ID変更）
  [Section 4] 棄却閾値校正(LOO)     … 常時実行（推奨閾値 θ）
  [Section 5] LLM 判定精度          … --llm（LLM_PROVIDER 環境変数が必要）
  [Section 6] シミュレーション精度  … --sim（circuit_simulator.py + eval_expected.yaml）

評価対象は sample_netlists.json をその場で特徴量化して構成するため、
回路を追加・削除すると自動的に評価対象が変わる（features_db.json の再生成は不要）。

使い方:
  python evaluate.py                              # Section 1,3,4
  python evaluate.py --alpha-sweep                # + Section 2
  LLM_PROVIDER=claude python evaluate.py --llm    # + Section 5
  python evaluate.py --sim                        # + Section 6
  LLM_PROVIDER=claude python evaluate.py --alpha-sweep --llm --sim   # 全実行
"""

from __future__ import annotations

import os
import re
import sys
import copy
import json
import argparse

# Windows コンソール(cp932)でも ✓/✗/罫線などを出力できるよう UTF-8 に固定する
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from feature_extractor import extract_hierarchical_features
from circuit_rag import CircuitRAG

SAMPLES_PATH = "sample_netlists.json"
EXPECTED_PATH = "eval_expected.yaml"

SEP = "─" * 60


# ─────────────────────────────────────────────────────────
# 共通ユーティリティ
# ─────────────────────────────────────────────────────────

def load_circuits(path: str = SAMPLES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["circuits"]


def build_rag(circuits: list[dict], llm=None) -> CircuitRAG:
    """サンプル回路をその場で特徴量化して検索可能な RAG を構築する。"""
    rag = CircuitRAG(llm=llm)
    for c in circuits:
        rag.add(extract_hierarchical_features(c))
    return rag


def _self_rank(rag: CircuitRAG, circuit: dict, alpha: float,
               n_db: int) -> tuple[int, float, dict]:
    """circuit を投入し、(自分自身の順位, 自分自身のスコア, top1のfeatures) を返す。"""
    q = extract_hierarchical_features(circuit)
    hits = rag.search(q, top_k=n_db, alpha=alpha)
    self_rank, self_score = n_db, 0.0
    for h in hits:
        if h["features"]["circuit_id"] == circuit["id"]:
            self_rank, self_score = h["rank"], h["score"]
            break
    return self_rank, self_score, hits[0]["features"]


# ─────────────────────────────────────────────────────────
# Section 1: 自己検索テスト
# ─────────────────────────────────────────────────────────

def section1_self_search(circuits: list[dict], alpha: float) -> dict:
    rag = build_rag(circuits)
    n = len(circuits)
    ranks, self_scores, failures = [], [], []

    for c in circuits:
        rank, score, top1 = _self_rank(rag, c, alpha, n)
        ranks.append(rank)
        self_scores.append(score)
        if rank != 1:
            failures.append((c["id"], rank, top1["circuit_id"], top1["circuit_name"]))

    hit1 = sum(r == 1 for r in ranks) / n
    hit3 = sum(r <= 3 for r in ranks) / n
    mrr = sum(1.0 / r for r in ranks) / n

    print(f"\n[Section 1] 自己検索テスト  (alpha={alpha}) {SEP}")
    print(f"  Hit@1 : {sum(r == 1 for r in ranks)}/{n} = {hit1*100:.1f}%")
    print(f"  Hit@3 : {sum(r <= 3 for r in ranks)}/{n} = {hit3*100:.1f}%")
    print(f"  MRR   : {mrr:.3f}")
    for cid, rank, t_id, t_name in failures:
        print(f"  ⚠ 失敗: {cid} → rank {rank}  (top1={t_id} / {t_name})")
    if not failures:
        print("  ✓ 全回路が rank 1")

    return {"hit1": hit1, "hit3": hit3, "mrr": mrr,
            "self_scores": self_scores, "failures": failures}


# ─────────────────────────────────────────────────────────
# Section 2: Alpha グリッドサーチ
# ─────────────────────────────────────────────────────────

def section2_alpha_sweep(circuits: list[dict]) -> float:
    rag = build_rag(circuits)
    n = len(circuits)
    print(f"\n[Section 2] Alpha グリッドサーチ {SEP}")

    rates: dict[float, float] = {}
    for step in range(11):
        alpha = round(step / 10.0, 1)
        hit1 = sum(_self_rank(rag, c, alpha, n)[0] == 1 for c in circuits)
        rates[alpha] = hit1 / n
        print(f"  alpha={alpha:.1f} : Hit@1 = {hit1}/{n} = {rates[alpha]*100:.1f}%")

    best = max(rates.values())
    plateau = sorted(a for a, r in rates.items() if r == best)
    # プラトー内では最もトポロジー寄り(=最大 alpha)を推奨する。
    # 実クエリはタグを持たない場合があり、タグ依存(低 alpha)は脆いため。
    recommended = max(plateau)
    topo_only = rates[1.0]   # トポロジーのみ。特徴量ベクトルの識別力の診断値

    print(f"  → 最高 Hit@1 = {best*100:.1f}%  "
          f"（alpha {plateau[0]:.1f}〜{plateau[-1]:.1f} が同率）")
    print(f"  → 推奨 alpha = {recommended:.1f}  "
          f"（同率内で最もトポロジー寄り＝タグ欠落クエリに頑健）")
    print(f"  ◇ トポロジーのみ(alpha=1.0)の Hit@1 = {topo_only*100:.1f}%")
    if topo_only < best:
        print(f"    → 構造ベクトル単独では {n - round(topo_only*n)} 件を取り違える。"
              f"特徴量次元の拡張余地（タスク: DB拡張）を示す診断。")
    return recommended


# ─────────────────────────────────────────────────────────
# Section 3: 摂動ロバスト性テスト
# ─────────────────────────────────────────────────────────

def _remap_circuit(circuit: dict, node_map: dict | None = None,
                   id_map: dict | None = None) -> dict:
    c = copy.deepcopy(circuit)
    if node_map:
        for comp in c["components"]:
            comp["terminals"] = {k: node_map.get(v, v)
                                 for k, v in comp["terminals"].items()}
        c["ports"] = {k: node_map.get(v, v) for k, v in c["ports"].items()}
    if id_map:
        for comp in c["components"]:
            comp["id"] = id_map.get(comp["id"], comp["id"])
    return c


def _perturb_nodes(circuit: dict) -> dict:
    """全ノード名（GND・ポート含む）を一貫してリネームする。"""
    nodes = set()
    for comp in circuit["components"]:
        nodes.update(comp["terminals"].values())
    nodes.update(circuit["ports"].values())
    node_map = {n: f"renamed_node_{i}" for i, n in enumerate(sorted(nodes))}
    return _remap_circuit(circuit, node_map=node_map)


def _perturb_ids(circuit: dict) -> dict:
    """全部品 ID をリネームする。"""
    id_map = {comp["id"]: f"PART_{i}" for i, comp in enumerate(circuit["components"])}
    return _remap_circuit(circuit, id_map=id_map)


def section3_perturbation(circuits: list[dict], alpha: float) -> int:
    rag = build_rag(circuits)
    n = len(circuits)
    print(f"\n[Section 3] 摂動ロバスト性テスト  (alpha={alpha}) {SEP}")

    problems = 0
    for label, perturb in (("ノード名変更", _perturb_nodes),
                           ("部品ID変更", _perturb_ids)):
        ok, fails = 0, []
        for c in circuits:
            pc = perturb(c)
            pc["id"] = c["id"]  # 自己照合のため ID は元のまま
            rank, _, _ = _self_rank(rag, pc, alpha, n)
            if rank == 1:
                ok += 1
            else:
                fails.append((c["id"], rank))
        print(f"  {label}: {ok}/{n} PASS")
        for cid, rank in fails:
            print(f"    ⚠ {cid} → rank {rank}（特徴抽出にノード名/ID依存の疑い）")
        problems += len(fails)

    if problems == 0:
        print("  ✓ 摂動に対して検索段は不変")
    return problems


# ─────────────────────────────────────────────────────────
# Section 4: 棄却閾値校正（Leave-one-out）
# ─────────────────────────────────────────────────────────

def section4_threshold(circuits: list[dict], alpha: float,
                       self_scores: list[float]) -> float | None:
    n = len(circuits)
    print(f"\n[Section 4] 棄却閾値校正 (LOO)  (alpha={alpha}) {SEP}")

    loo_scores = []
    worst = None  # (loo_score, query_id, impostor_name)
    for i, c in enumerate(circuits):
        others = circuits[:i] + circuits[i + 1:]
        rag = build_rag(others)
        q = extract_hierarchical_features(c)
        hits = rag.search(q, top_k=1, alpha=alpha)
        s = hits[0]["score"]
        loo_scores.append(s)
        if worst is None or s > worst[0]:
            worst = (s, c["id"], hits[0]["features"]["circuit_name"])

    min_self = min(self_scores)
    max_loo = max(loo_scores)
    theta = (min_self + max_loo) / 2.0

    print(f"  既知回路スコア   (min self) : {min_self:.4f}")
    print(f"  LOO 擬似未知     (max loo)  : {max_loo:.4f}"
          f"   ← {worst[1]} が {worst[2]} になりすまし")
    if max_loo >= min_self:
        print("  ⚠ 未知スコアが既知スコアを上回るペアがある（分離不能）。")
        print("    閾値だけでは棄却できない。特徴量の識別性能を要確認。")
    print(f"  → 推奨閾値 θ = {theta:.4f}  (中点)")
    print(f"    circuit_rag.py の search_with_threshold(threshold={theta:.4f}) に反映可能")
    return theta


# ─────────────────────────────────────────────────────────
# Section 5: LLM 判定精度（RAGのみ vs RAG+シミュレーション）
# ─────────────────────────────────────────────────────────

def _norm_name(s: str) -> str:
    """括弧書き・空白を除去して粗い文字列マッチ用に正規化する。"""
    s = re.sub(r"（.*?）|\(.*?\)", "", s)
    return s.replace(" ", "").replace("　", "")


def _name_match(expected_name: str, output: str) -> bool:
    return _norm_name(expected_name) in _norm_name(output or "")


def _make_llm():
    """circuit_rag.py と同じ規則で LLM クライアントを選択する。"""
    from llm_client import LLMClient, CLILLMClient, MockLLMClient
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider in ("claude", "gemini"):
        return CLILLMClient(provider=provider)
    if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        return LLMClient(provider="anthropic")
    if provider == "gemini-sdk" and os.environ.get("GEMINI_API_KEY"):
        return LLMClient(provider="gemini")
    return MockLLMClient()


def section5_llm(circuits: list[dict], alpha: float, top_k: int) -> None:
    llm = _make_llm()
    print(f"\n[Section 5] LLM 判定精度  (LLM={llm}) {SEP}")
    if type(llm).__name__ == "MockLLMClient":
        print("  ⚠ LLM_PROVIDER 未設定のためモック動作。実測には")
        print("    LLM_PROVIDER=claude python evaluate.py --llm のように指定する。")

    rag = build_rag(circuits, llm=llm)
    rag_only_ok, rag_sim_ok = 0, 0
    fails = []
    for c in circuits:
        name = c["name"]
        # RAG のみ（シミュレーション結果を渡さない）
        out_rag = rag.judge(c, sim_result=None, top_k=top_k, alpha=alpha)
        m_rag = _name_match(name, out_rag)
        rag_only_ok += m_rag
        # RAG + シミュレーション
        out_sim = rag.analyze(c, top_k=top_k, alpha=alpha)["identification"]
        m_sim = _name_match(name, out_sim)
        rag_sim_ok += m_sim
        if not (m_rag and m_sim):
            fails.append((c["id"], name, m_rag, m_sim))

    n = len(circuits)
    print(f"  RAGのみ          : {rag_only_ok}/{n} = {rag_only_ok/n*100:.1f}%")
    print(f"  RAG+シミュレーション: {rag_sim_ok}/{n} = {rag_sim_ok/n*100:.1f}%")
    print(f"  → シミュレーションの寄与: {rag_sim_ok - rag_only_ok:+d} 件")
    for cid, name, m_rag, m_sim in fails:
        tag = []
        if not m_rag:
            tag.append("RAGのみ✗")
        if not m_sim:
            tag.append("RAG+Sim✗")
        print(f"  ⚠ {cid}: 出力に「{_norm_name(name)}」が含まれない（{', '.join(tag)}）")


# ─────────────────────────────────────────────────────────
# Section 6: シミュレーション精度
# ─────────────────────────────────────────────────────────

def section6_simulation(circuits: list[dict]) -> int:
    print(f"\n[Section 6] シミュレーション精度 {SEP}")
    try:
        from circuit_simulator import CircuitSimulator
    except ImportError:
        print("  ⚠ circuit_simulator.py が見つかりません。スキップします。")
        return 0

    try:
        import yaml
    except ImportError:
        print("  ⚠ PyYAML 未インストール（pip install pyyaml）。スキップします。")
        return 0

    if not os.path.exists(EXPECTED_PATH):
        print(f"  ⚠ {EXPECTED_PATH} が見つかりません。スキップします。")
        return 0

    with open(EXPECTED_PATH, encoding="utf-8") as f:
        expected = yaml.safe_load(f)

    passed, checked, skipped, problems = 0, 0, 0, 0
    flag_keys = ("is_lowpass", "is_highpass", "is_bandpass", "is_bandstop")

    for c in circuits:
        cid = c["id"]
        if cid not in expected:
            continue
        exp = expected[cid]
        result = CircuitSimulator(c).extract_simulation_features()
        got_type = result.get("simulation_type")
        exp_type = exp.get("simulation_type")

        # ngspice 未検出は環境要因 → 失敗ではなくスキップ扱い
        if exp_type == "ac_passive" and got_type in (
                "skipped_no_ngspice",):
            print(f"  - {cid:24} SKIP（ngspice/PySpice 未検出のため照合不能）")
            skipped += 1
            continue

        checked += 1
        errs = []
        if got_type != exp_type:
            errs.append(f"type: 期待={exp_type} 実際={got_type}")

        for flag in flag_keys:
            if flag in exp and result.get(flag) != exp[flag]:
                errs.append(f"{flag}: 期待={exp[flag]} 実際={result.get(flag)}")

        if "cutoff_freq_hz" in exp:
            fc = result.get("cutoff_freq_hz")
            lo, hi = exp["cutoff_freq_hz"]
            if fc is None:
                errs.append(f"cutoff: 期待[{lo},{hi}] 実際=None")
            elif not (lo <= fc <= hi):
                errs.append(f"cutoff: 期待[{lo},{hi}] 実際={fc}")

        if errs:
            problems += 1
            print(f"  ✗ {cid:24} {got_type}")
            for e in errs:
                print(f"        {e}")
        else:
            passed += 1
            extra = ""
            if got_type == "ac_passive":
                extra = f"  fc={result.get('cutoff_freq_hz')}Hz conf={result.get('confidence')}"
            print(f"  ✓ {cid:24} {got_type}{extra}")

    print(f"  ── 照合 {passed}/{checked} PASS"
          + (f" / SKIP {skipped}" if skipped else ""))
    return problems


# ─────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TopoRAG 評価スクリプト")
    parser.add_argument("--samples", default=SAMPLES_PATH,
                        help=f"評価対象ネットリスト（デフォルト: {SAMPLES_PATH}）")
    parser.add_argument("--alpha", "-a", type=float, default=0.7,
                        help="トポロジースコアの重み（デフォルト: 0.7）")
    parser.add_argument("--top-k", "-k", type=int, default=3,
                        help="LLM 判定で参照する上位件数（デフォルト: 3）")
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="Section 2: Alpha グリッドサーチを実行")
    parser.add_argument("--llm", action="store_true",
                        help="Section 5: LLM 判定精度を測定（LLM_PROVIDER 必要）")
    parser.add_argument("--sim", action="store_true",
                        help="Section 6: シミュレーション精度を測定")
    args = parser.parse_args()

    circuits = load_circuits(args.samples)
    print(f"評価対象: {len(circuits)} 回路  ({args.samples})  alpha={args.alpha}")

    hard_problems = 0

    # Section 1（常時）
    s1 = section1_self_search(circuits, args.alpha)

    # Section 2（任意）
    if args.alpha_sweep:
        section2_alpha_sweep(circuits)

    # Section 3（常時）— 摂動失敗は実バグの可能性が高いため hard fail に数える
    hard_problems += section3_perturbation(circuits, args.alpha)

    # Section 4（常時）
    section4_threshold(circuits, args.alpha, s1["self_scores"])

    # Section 5（任意）
    if args.llm:
        section5_llm(circuits, args.alpha, args.top_k)

    # Section 6（任意）— 期待値ミスマッチは hard fail
    if args.sim:
        hard_problems += section6_simulation(circuits)

    print(f"\n{SEP}")
    if hard_problems == 0:
        print("総合: 回帰なし（摂動・シミュレーション照合で hard failure 0 件）")
    else:
        print(f"総合: hard failure {hard_problems} 件 — 上記 ⚠/✗ を確認してください")
    raise SystemExit(1 if hard_problems else 0)


if __name__ == "__main__":
    main()
