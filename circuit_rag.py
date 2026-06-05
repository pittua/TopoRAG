"""
回路ネットリスト 類似検索RAGシステム v3
  - LLMClient 経由で Anthropic / Gemini を切り替え可能
  - ハイパス/ローパス識別対応
"""

import json
import numpy as np
from feature_extractor import extract_all_features, extract_hierarchical_features
from llm_client import LLMClient, CLILLMClient, MockLLMClient
from circuit_simulator import CircuitSimulator


# ─────────────────────────────────────────────────────────
# シミュレーション結果のフォーマット
# ─────────────────────────────────────────────────────────

_SIM_TYPE_LABEL = {
    "ac_passive":             "AC解析（パッシブ）",
    "tran_nonlinear":         "過渡解析（大信号）",
    "skipped_switch":         "スキップ（スイッチング回路）",
    "skipped_active":         "スキップ（能動素子・小信号解析未実装）",
    "skipped_nonlinear":      "スキップ（ダイオード/非線形・過渡解析未実装）",
    "skipped_missing_values": "スキップ（部品値欠落）",
    "skipped_no_ngspice":     "スキップ（ngspice/PySpice 未検出）",
    "skipped_error":          "スキップ（解析失敗）",
}


def _sim_filter_label(sim: dict) -> str:
    for flag, label in (
        ("is_lowpass", "ローパス"), ("is_highpass", "ハイパス"),
        ("is_bandpass", "バンドパス"), ("is_bandstop", "ノッチ/バンドストップ"),
    ):
        if sim.get(flag):
            return label
    return "特定なし"


def _format_sim_summary(sim: dict | None) -> str:
    """人が読めるシミュレーション要約。"""
    if not sim:
        return "【シミュレーション解析】\n  実行なし"
    stype = sim.get("simulation_type", "")
    label = _SIM_TYPE_LABEL.get(stype, stype)

    if stype.startswith("skipped"):
        reason = (sim.get("warnings") or [""])[0]
        return f"【シミュレーション解析】\n  解析種別 : {label}\n  理由     : {reason}"

    lines = [
        "【シミュレーション解析】",
        f"  解析種別          : {label}",
        f"  DC 利得           : {sim.get('dc_gain_db')} dB",
        f"  高周波利得         : {sim.get('hf_gain_db')} dB",
        f"  フィルタ特性       : {_sim_filter_label(sim)}",
        f"  カットオフ周波数    : {sim.get('cutoff_freq_hz')} Hz",
        f"  共振              : {'あり' if sim.get('has_resonance') else 'なし'}",
        f"  信頼度            : {sim.get('confidence')}",
    ]
    if sim.get("warnings"):
        lines.append(f"  注意              : {', '.join(sim['warnings'])}")
    return "\n".join(lines)


def _format_sim_for_prompt(sim: dict | None) -> str:
    """LLM プロンプトに挿入するシミュレーション解析セクション。"""
    if not sim:
        return ""
    stype = sim.get("simulation_type", "")
    label = _SIM_TYPE_LABEL.get(stype, stype)

    if stype.startswith("skipped"):
        reason = (sim.get("warnings") or [""])[0]
        return (
            "\n\n## シミュレーション解析結果\n"
            f"  解析種別      : {label}\n"
            f"  理由          : {reason}\n"
            "  ※ シミュレーション値は得られていません。トポロジー情報のみで判定してください。"
        )

    lines = [
        "\n\n## シミュレーション解析結果",
        f"  解析種別      : {label}",
        f"  フィルタ特性  : {_sim_filter_label(sim)}",
        f"  DC 利得       : {sim.get('dc_gain_db')} dB",
        f"  高周波利得    : {sim.get('hf_gain_db')} dB",
        f"  カットオフ    : {sim.get('cutoff_freq_hz')} Hz",
        f"  共振          : {'あり' if sim.get('has_resonance') else 'なし'}",
        f"  信頼度        : {sim.get('confidence')}",
    ]
    if sim.get("warnings"):
        lines.append(f"  注意          : {', '.join(sim['warnings'])}")
    if sim.get("confidence") == "low":
        lines.append(
            "  ※ この結果は信頼度が低い（近似・境界ケース等）。"
            "フィルタ特性フラグは参考値として扱ってください。")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# ベクトル化（34次元）
#   0–18  : 受動トポロジー（R/C/L/SW/D/DZ）。能動素子追加後も不変。
#   19–33 : 能動素子（BJT/MOSFET/OpAmp）の構成・多素子トポロジー。
#           受動回路では全て 0 になり、末尾ゼロ拡張のためコサイン類似度は不変
#           （= 既存19回路の自己検索/摂動テストは回帰しない）。
# ─────────────────────────────────────────────────────────

def vectorize(features: dict) -> np.ndarray:
    A  = features["A_component"]
    B1 = features["B1_order"]
    B2 = features["B2_diode"]
    B3 = features["B3_series_parallel"]
    C  = features["C_node"]
    D  = features.get("D_active", {})   # 旧DB互換: 無ければ全て False/None

    order_map = {"SW_before_L": 1.0, "L_before_SW": -1.0, None: 0.0}
    first = B1.get("first_series_type")
    tc = D.get("transistor_config")
    oc = D.get("opamp_config")

    return np.array([
        float(A["has_resistor"]),           # 0
        float(A["has_capacitor"]),          # 1
        float(A["has_inductor"]),           # 2
        float(A["has_switch"]),             # 3
        float(A["has_diode"]),              # 4
        order_map.get(B1["sw_l_order"], 0.0),  # 5
        float(first == "R"),               # 6
        float(first == "C"),               # 7
        float(first == "L"),               # 8
        float(B2["diode_anode_to_gnd"]),   # 9
        float(B2["diode_cathode_to_out"]), # 10
        float(B2["diode_anode_to_out"]),   # 11
        float(B2["diode_cathode_to_gnd"]), # 12
        float(B3["has_parallel_components"]),      # 13
        min(B3["series_chain_length"] / 5.0, 1.0), # 14
        min(C["node_count"] / 10.0, 1.0), # 15
        float(C["has_high_degree_node"]),  # 16
        min(C["cycle_count"] / 5.0, 1.0), # 17
        float(A.get("has_zener", False)),  # 18
        float(D.get("has_bjt", False)),    # 19
        float(D.get("has_mosfet", False)), # 20
        float(D.get("has_opamp", False)),  # 21
        float(tc == "CE"),                 # 22  接地エミッタ/ソース（反転増幅）
        float(tc == "CC"),                 # 23  コレクタ/ドレイン接地（フォロワ）
        float(tc == "CB"),                 # 24  ベース/ゲート接地
        float(oc == "inverting"),          # 25  反転アンプ
        float(oc == "non_inverting"),      # 26  非反転アンプ
        float(oc == "buffer"),             # 27  ボルテージフォロワ
        float(D.get("has_feedback", False)), # 28  帰還の有無
        float(D.get("p_type", False)),     # 29  PNP/PMOS（極性）
        min(D.get("n_active", 0) / 4.0, 1.0), # 30  能動素子数（正規化）
        float(D.get("has_diode_connected", False)), # 31  ダイオード接続（カレントミラー）
        float(D.get("has_coupled_pair", False)),    # 32  結合ペア（差動/ロングテール）
        float(D.get("is_differential", False)),     # 33  差動（2入力）
        float(B2.get("diode_series", False)),         # 34  直列ダイオード（整流/昇圧の本線）
        float(B2.get("diode_anode_at_input", False)), # 35  アノード＝入力（真の整流段）
        float(B2.get("diode_shunt", False)),          # 36  シャントダイオード（クリッパ/ツェナー）
        float(B2.get("rectifier_smoothing", False)),  # 37  整流＋出力平滑コンデンサ
    ], dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def tag_similarity(tags_a: list, tags_b: list) -> float:
    """Jaccard 係数によるタグ類似度（0.0〜1.0）"""
    a, b = set(tags_a), set(tags_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# 棄却閾値: 実機20回路(in14/out6)を reject_eval.py で校正した値（balanced acc 最大、
# AUC 0.893）。トポロジーのみ(alpha=1.0)の top-1 スコアに対して適用する。
# 旧 LOO 校正値(0.9786)は「素のDB」基準で実機を全棄却するため使用しない。
RECOMMENDED_REJECT_THRESHOLD = 0.83


# ─────────────────────────────────────────────────────────
# CircuitRAG
# ─────────────────────────────────────────────────────────

class CircuitRAG:

    def __init__(self, llm: LLMClient | CLILLMClient | MockLLMClient | None = None):
        self.db: list[dict]                   = []
        self.vectors: list[np.ndarray]        = []
        self.block_vectors: list[list[np.ndarray]] = []  # ブロック単位ベクトル（単一ブロック回路は []）
        self.llm = llm

    def add(self, features: dict):
        self.db.append(features)
        self.vectors.append(vectorize(features))
        self.block_vectors.append([vectorize(b) for b in features.get("blocks", [])])

    def load_from_file(self, path: str):
        with open(path, encoding="utf-8") as f:
            for rec in json.load(f):
                self.add(rec)
        print(f"DB読み込み: {len(self.db)} 回路")

    # ── トポロジー類似度（階層対応） ─────────────────────

    @staticmethod
    def _block_match_greedy(q_vecs: list, d_vecs: list) -> float:
        """
        グリーディ最適ブロックマッチング。
        一方のブロック数が多い場合は余剰ブロックをスコア 0 として扱い、
        max(n_q, n_d) で正規化することでブロック数ミスマッチにペナルティを与える。
        """
        n_q, n_d = len(q_vecs), len(d_vecs)
        sims = [[cosine_similarity(q, d) for d in d_vecs] for q in q_vecs]
        used_q: set[int] = set()
        used_d: set[int] = set()
        total = 0.0
        for _ in range(min(n_q, n_d)):
            best, bi, bj = -1.0, -1, -1
            for i in range(n_q):
                if i in used_q:
                    continue
                for j in range(n_d):
                    if j in used_d:
                        continue
                    if sims[i][j] > best:
                        best, bi, bj = sims[i][j], i, j
            if bi == -1:
                break
            total += best
            used_q.add(bi)
            used_d.add(bj)
        return total / max(n_q, n_d)

    def _topo_similarity(self,
                         q_vec: np.ndarray, q_bvecs: list[np.ndarray],
                         d_vec: np.ndarray, d_bvecs: list[np.ndarray]) -> float:
        """
        階層構造を考慮したトポロジー類似度。

        ケース                        処理
        ─────────────────────────────────────────────────────
        両方フラット                  従来のコサイン類似度
        クエリがブロック / DBがフラット  各クエリブロック vs DB回路のコサイン最大値
        クエリがフラット / DBがブロック  クエリ vs 各DBブロックのコサイン最大値
        両方ブロック                  グリーディブロックマッチング
        """
        has_q = len(q_bvecs) > 0
        has_d = len(d_bvecs) > 0

        if not has_q and not has_d:
            return cosine_similarity(q_vec, d_vec)
        if has_q and not has_d:
            return max(cosine_similarity(bv, d_vec) for bv in q_bvecs)
        if not has_q and has_d:
            return max(cosine_similarity(q_vec, bv) for bv in d_bvecs)
        return self._block_match_greedy(q_bvecs, d_bvecs)

    # ── 類似検索 ─────────────────────────────────────────

    def search(self, query_features: dict, top_k: int = 3,
               alpha: float = 0.7) -> list[dict]:
        """
        alpha: トポロジースコアの重み（0.0〜1.0）
               残り (1-alpha) がタグスコアの重み
               クエリにタグがなければ alpha=1.0 と同じ結果になる
        """
        q_vec   = vectorize(query_features)
        q_bvecs = [vectorize(b) for b in query_features.get("blocks", [])]
        q_tags  = query_features.get("function_tags", [])
        scores  = []
        for db_feat, db_vec, db_bvecs in zip(self.db, self.vectors, self.block_vectors):
            topo = self._topo_similarity(q_vec, q_bvecs, db_vec, db_bvecs)
            tag  = tag_similarity(q_tags, db_feat.get("function_tags", []))
            scores.append(alpha * topo + (1.0 - alpha) * tag)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            {"rank": r + 1, "score": round(scores[i], 4), "features": self.db[i]}
            for r, (i, _) in enumerate(ranked[:top_k])
        ]

    def search_with_rejection(self, query_features: dict, top_k: int = 3,
                              alpha: float = 0.7,
                              reject_threshold: float = RECOMMENDED_REJECT_THRESHOLD
                              ) -> tuple[list[dict], bool, float]:
        """
        未知（DB 未収録）回路を棄却する検索。

        棄却判定は **トポロジーのみ(alpha=1.0) の top-1 スコア** で行う。
        これは reject_eval.py の比較で最も分離性能が高かったシグナルで（AUC 0.893）、
        margin/ratio は実機ではほぼ無力だったため採用しない。タグ非依存にするのは、
        実機クエリがタグを持たず、タグ込み絶対スコアが校正用DBと実機で別分布になるため。

        返り値:
          hits        : 受理時は通常検索(指定 alpha)の上位 top_k。棄却時は []（識別不能）。
          accepted    : 受理されたか。
          topo_conf   : 判定に用いたトポロジーのみ top-1 スコア（信頼度）。
        """
        topo_conf = self.search(query_features, top_k=1, alpha=1.0)[0]["score"]
        accepted = topo_conf >= reject_threshold
        hits = self.search(query_features, top_k=top_k, alpha=alpha) if accepted else []
        return hits, accepted, topo_conf

    # ── プロンプト生成 ────────────────────────────────────

    def _fmt(self, f: dict) -> str:
        B1, B2 = f["B1_order"], f["B2_diode"]
        tags = f.get("function_tags", [])
        desc = f.get("description", "")
        lines = []
        if tags:
            lines.append(f"  機能タグ        : {', '.join(tags)}")
        if desc:
            lines.append(f"  説明            : {desc}")
        lines += [
            f"  部品種別        : {f['A_component']['component_types']}",
            f"  先頭直列部品    : {B1.get('first_series_type')}",
            f"  直列シーケンス  : {B1['series_type_sequence']}",
            f"  GND並列部品     : {B1.get('shunt_type_sequence', [])}",
            f"  SW-L順序        : {B1['sw_l_order']}",
            f"  Dアノード→GND  : {B2['diode_anode_to_gnd']}",
            f"  Dカソード→OUT  : {B2['diode_cathode_to_out']}",
            f"  ループ数        : {f['C_node']['cycle_count']}",
        ]
        D = f.get("D_active", {})
        if D.get("has_active"):
            _CFG = {"CE": "接地エミッタ/ソース(反転増幅)", "CC": "コレクタ/ドレイン接地(フォロワ)",
                    "CB": "ベース/ゲート接地"}
            parts = []
            if D.get("transistor_config"):
                parts.append(_CFG.get(D["transistor_config"], D["transistor_config"]))
            if D.get("opamp_config"):
                parts.append(f"OpAmp:{D['opamp_config']}")
            lines.append(
                f"  能動素子構成    : {' / '.join(parts) or '不明'}"
                f"{'  (帰還あり)' if D.get('has_feedback') else ''}"
                f"{'  (p型)' if D.get('p_type') else ''}"
            )
        if f.get("is_hierarchical") and f.get("blocks"):
            lines.append(f"  ブロック構成     : {f['n_blocks']} ブロック")
            for i, blk in enumerate(f["blocks"]):
                lines.append(
                    f"    ブロック{i + 1}: 部品={blk['A_component']['component_types']}"
                    f"  直列={blk['B1_order']['series_type_sequence']}"
                    f"  GND並列={blk['B1_order']['shunt_type_sequence']}"
                )
        return "\n".join(lines)

    def build_prompt(self, query_circuit: dict, hits: list[dict] | None = None,
                     sim_result: dict | None = None,
                     top_k: int = 3, alpha: float = 0.7) -> tuple[str, str]:
        q_feat = extract_hierarchical_features(query_circuit)

        # 複合回路はブロック単位 RAG に切り替え
        if q_feat.get("is_hierarchical") and q_feat.get("blocks"):
            return self._build_hierarchical_prompt(
                q_feat, sim_result=sim_result, top_k=top_k, alpha=alpha)

        # 単一ブロック：従来の全体照合（hits 未指定なら検索。二重検索を避ける）
        if hits is None:
            hits = self.search(q_feat, top_k=top_k, alpha=alpha)

        system = (
            "あなたは電子回路の専門家です。\n"
            "与えられたネットリストの特徴量、検索された類似回路の情報、"
            "およびシミュレーション解析結果（あれば）をもとに、"
            "入力回路のトポロジー（回路種別）を判定してください。\n"
            "回答は必ず以下の形式で出力してください：\n"
            "【判定】<回路名>\n"
            "【根拠】<接続構造とシミュレーション特性に基づく理由を2〜3文>\n"
            "【類似度の解釈】<検索結果との比較コメント>"
        )

        refs = "\n\n".join(
            f"[{h['rank']}位 類似度:{h['score']:.2f}] {h['features']['circuit_name']}\n"
            + self._fmt(h["features"])
            for h in hits
        )

        user = (
            "## 参照：類似回路（RAG検索結果）\n\n" + refs
            + "\n\n## 判定対象回路の特徴量\n\n" + self._fmt(q_feat)
            + _format_sim_for_prompt(sim_result)
            + "\n\n上記をもとに回路のトポロジーを判定してください。"
        )
        return system, user

    def _build_hierarchical_prompt(self, q_feat: dict,
                                   sim_result: dict | None = None,
                                   top_k: int = 2,
                                   alpha: float = 0.7) -> tuple[str, str]:
        """
        複合回路向けプロンプト。
        ブロック単位で DB 照合し、各ブロックの照合結果を並べて提示する。
        LLM への要求は「照合結果を読んでブロック名を組み合わせる」だけ——回路推論は不要。
        """
        system = (
            "あなたは電子回路の専門家です。\n"
            "入力回路は複数の機能ブロックに分解されており、"
            "各ブロックの最類似回路がDB照合によってすでに特定されています。\n"
            "回路トポロジーの推論は不要です。照合結果を読み取り、"
            "ブロック名を組み合わせて全体回路の種別を答えてください。\n"
            "回答は必ず以下の形式で出力してください：\n"
            "【判定】<全体の回路種別（例：降圧チョッパ＋LC出力フィルタ）>\n"
            "【ブロック構成】<各ブロック番号と照合先回路名・役割>\n"
            "【根拠】<照合結果に基づく説明を1〜2文>"
        )

        block_sections = []
        for i, blk in enumerate(q_feat["blocks"]):
            inp = blk.get("ports", {}).get("input", "?") if "ports" in blk else "?"
            out = blk.get("ports", {}).get("output", "?") if "ports" in blk else "?"
            hits = self.search(blk, top_k=top_k, alpha=alpha)

            lines = [f"### ブロック{i + 1}  ({inp} → {out})"]
            lines.append(
                f"  部品種別      : {blk['A_component']['component_types']}\n"
                f"  直列シーケンス: {blk['B1_order']['series_type_sequence']}\n"
                f"  GND並列部品   : {blk['B1_order']['shunt_type_sequence']}"
            )
            lines.append("  〔DB照合結果〕")
            for h in hits:
                lines.append(
                    f"    {h['rank']}位 score={h['score']:.3f}  "
                    f"{h['features']['circuit_name']}"
                )
            block_sections.append("\n".join(lines))

        user = (
            f"## 複合回路のブロック分解結果  ({q_feat['n_blocks']} ブロック)\n\n"
            + "\n\n".join(block_sections)
            + "\n\n## 回路全体の特徴量\n\n"
            + self._fmt(q_feat)
            + _format_sim_for_prompt(sim_result)
            + "\n\n上記ブロック照合結果から、全体回路の種別と構成を特定してください。"
        )
        return system, user

    # ── LLM判定 ──────────────────────────────────────────

    def judge(self, circuit: dict, hits: list[dict] | None = None,
              sim_result: dict | None = None,
              top_k: int = 3, alpha: float = 0.7) -> str:
        if self.llm is None:
            raise RuntimeError(
                "LLMが設定されていません。\n"
                "CircuitRAG(llm=LLMClient('anthropic')) のように渡してください。"
            )
        system, user = self.build_prompt(
            circuit, hits=hits, sim_result=sim_result, top_k=top_k, alpha=alpha)
        return self.llm.chat(system=system, user=user)

    # ── シミュレーション統合判定 ──────────────────────────

    def analyze(self, circuit: dict, top_k: int = 3, alpha: float = 0.7) -> dict:
        """
        シミュレーション解析 → RAG 検索 → LLM 判定 の順に実行し、結果を統合して返す。
        RAG 検索は 1 回だけ実行し、その結果を judge() に渡す（二重検索を避ける）。

        Returns:
            {
                "identification":     str,         # LLM による識別結果テキスト
                "top_hits":           list[dict],  # RAG 検索上位結果
                "simulation":         dict,        # シミュレーション特徴量
                "simulation_summary": str,         # 人が読める要約
            }
        """
        # Step 1: シミュレーション（LLM より先に実行）
        sim_result = CircuitSimulator(circuit).extract_simulation_features()

        # Step 2: RAG 検索（ここで 1 回だけ実行）
        q_feat = extract_hierarchical_features(circuit)
        hits = self.search(q_feat, top_k=top_k, alpha=alpha)

        # Step 3: LLM 判定（検索結果とシミュレーション結果を渡す。再検索しない）
        identification = self.judge(circuit, hits=hits, sim_result=sim_result,
                                    top_k=top_k, alpha=alpha)

        return {
            "identification":     identification,
            "top_hits":           hits,
            "simulation":         sim_result,
            "simulation_summary": _format_sim_summary(sim_result),
        }


# ─────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="回路ネットリスト RAG判定")
    parser.add_argument(
        "--query", "-q",
        default="query_netlists.json",
        help="判定対象ネットリストのJSONファイル（デフォルト: query_netlists.json）",
    )
    parser.add_argument(
        "--db",
        default="features_db.json",
        help="特徴量DBファイル（デフォルト: features_db.json）",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="類似検索で参照する上位件数（デフォルト: 3）",
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.7,
        help="トポロジースコアの重み（0.0〜1.0、デフォルト: 0.7）。残り(1-alpha)がタグスコアの重み",
    )
    args = parser.parse_args()

    # ── LLMクライアントの選択 ─────────────────────────────
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if provider == "claude":
        llm = CLILLMClient(provider="claude")
        print(f"使用LLM: {llm}")
    elif provider == "gemini":
        llm = CLILLMClient(provider="gemini")
        print(f"使用LLM: {llm}")
    elif provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        llm = LLMClient(provider="anthropic")
        print(f"使用LLM: {llm}")
    elif provider == "gemini-sdk" and os.environ.get("GEMINI_API_KEY"):
        llm = LLMClient(provider="gemini")
        print(f"使用LLM: {llm}")
    else:
        llm = MockLLMClient()
        print(f"使用LLM: {llm}（未設定のためモック動作）")
        print("  CLIで動かすには（APIキー不要）:")
        print("  Claude CLI: export LLM_PROVIDER=claude")
        print("  Gemini CLI: export LLM_PROVIDER=gemini")

    # ── RAG構築 ──────────────────────────────────────────
    rag = CircuitRAG(llm=llm)
    rag.load_from_file(args.db)

    # ── クエリ回路の読み込み ──────────────────────────────
    with open(args.query, encoding="utf-8") as f:
        queries = json.load(f)["circuits"]
    print(f"クエリ読み込み: {len(queries)} 回路  ({args.query})")

    # ── 類似検索 ─────────────────────────────────────────
    print("\n--- 類似検索 ---")
    for t in queries:
        feat = extract_hierarchical_features(t)
        hits = rag.search(feat, top_k=args.top_k, alpha=args.alpha)
        n_blk = feat.get("n_blocks", 1)
        label = f"({n_blk} ブロック)" if feat.get("is_hierarchical") else ""
        print(f"\n  クエリ: {t['name']} {label}")
        for h in hits:
            db_blk = h["features"].get("n_blocks", 1)
            db_label = f"({db_blk}B)" if h["features"].get("is_hierarchical") else ""
            print(f"    [{h['rank']}位] score={h['score']:.4f}  {h['features']['circuit_name']} {db_label}")

    # ── LLM判定（シミュレーション統合）──────────────────
    print("\n--- LLM判定（トポロジー ＋ シミュレーション）---")
    for t in queries:
        print(f"\n{'='*55}\n入力: {t['name']}\n{'='*55}")
        result = rag.analyze(t, top_k=args.top_k, alpha=args.alpha)
        print(result["identification"])
        print("\n" + result["simulation_summary"])
