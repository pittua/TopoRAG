"""
回路ネットリスト 類似検索RAGシステム v3
  - LLMClient 経由で Anthropic / Gemini を切り替え可能
  - ハイパス/ローパス識別対応
"""

import json
import numpy as np
from feature_extractor import extract_all_features, extract_hierarchical_features
from circuit_ir import build_ir, render_ir
from knowledge_cards import load_cards, render_card
from llm_client import LLMClient, CLILLMClient, MockLLMClient
from circuit_simulator import CircuitSimulator
from topo_kernel import wl_kernel
from scipy.optimize import linear_sum_assignment


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
# ベクトル化（43次元）
#   0–18  : 受動トポロジー（R/C/L/SW/D/DZ）。能動素子追加後も不変。
#   19–33 : 能動素子（BJT/MOSFET/OpAmp）の構成・多素子トポロジー。
#   34–37 : ダイオードの役割（直列/整流段/シャント/整流＋平滑）。
#   38–42 : 正規化部品数（R/C/L/D/SW）。presence では区別できない素子数を補う。
#           受動回路では 19–33 が全て 0 になり、末尾拡張のためコサイン類似度は
#           既存カテゴリで不変（= 既存19回路の自己検索/摂動テストは回帰しない）。
# ─────────────────────────────────────────────────────────

def vectorize(features: dict) -> np.ndarray:
    A  = features["A_component"]
    B1 = features["B1_order"]
    B2 = features["B2_diode"]
    B3 = features["B3_series_parallel"]
    C  = features["C_node"]
    D  = features.get("D_active", {})   # 旧DB互換: 無ければ全て False/None
    NC = A.get("normalized_counts", {}) # 正規化部品数（旧DB互換: 無ければ空＝0）

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
        # ── 部品数（sgnb 追加 38–42）─────────────────────────
        #   presence(0–4) は D1個↔D2個↔D4個や SW 数を区別できず、半波↔全波整流・
        #   buck↔ブリッジ整流が誤同定する。正規化部品数を末尾に追加して矯正する。
        #   既存カテゴリは部品数が一致するため自己検索は不変（末尾拡張で後方互換）。
        min(NC.get("R", 0) / 4.0, 1.0),   # 38  抵抗数
        min(NC.get("C", 0) / 4.0, 1.0),   # 39  コンデンサ数
        min(NC.get("L", 0) / 3.0, 1.0),   # 40  インダクタ数
        min(NC.get("D", 0) / 4.0, 1.0),   # 41  ダイオード数
        min(NC.get("SW", 0) / 2.0, 1.0),  # 42  スイッチ数
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


# 棄却閾値（実機コーパス in28/out20 で reject_eval.py により校正）。
# トポロジーのみ(alpha=1.0)の top-1 スコアに対して適用する。
#
# 事前登録ゲート（docs/HANDOFF_2026-06-09.md §7.3）を **合格**:
#   - n_out=20（≥20 要件を満たす）
#   - top-1 棄却 AUC 0.868 [95%CI 0.757–0.963]（CI 下限 > 0.5）
#   - nested CV balanced acc 0.800（≥0.65 要件を満たす。楽観バイアスは +0.025 に縮小）
#   → 判定: 「棄却は有意に機能・硬い二値判定を継続」
#
# θ=0.8863 での実績: in 受理 21/28（TPR 0.75）・out 棄却 18/20（TNR 0.90）。
# ⚠ 残存する限界（過信しないこと）:
#   - 誤受理 2/20: simulation-diode-characteristics(1.0000) と opamp-freerunning(0.9517) は
#     DB 収録回路と構造が一致し閾値で切れない（分離余裕 −0.33）。前者は測定試験ベンチで
#     コーパス品質寄り、後者は OpAmp 発振器。硬い棄却の構造的な天井を示す。
#   - 誤棄却 7/28: DB に近傍が無い hard in-scope（多段アンプ・コンバータ）。
#   - nested bacc は汎化推定であり完璧ではない。境界用途では top-k＋スコア提示（ランカー）も検討。
CALIBRATED_REJECT_THRESHOLD = 0.8863

# トポロジースコア内の配合: topo = beta*コサイン + (1-beta)*WLカーネル。
# beta=1.0 で従来(コサインのみ)に一致。beta=0.8 で自己検索(トポロジーのみ)の
# Hit@1 が 77.4%→100% に改善（WLカーネルが DB 内の構造衝突7件を解消）。
# 正式 search での beta-sweep（DB47 + 部品数特徴）の結果、beta=0.95 が最適点:
# 自己検索(トポロジーのみ) 100% かつ 実機 in-scope Hit@1 73.3%・MRR 0.822（ともに最良）。
# コサイン＋部品数を主軸にしつつ WL カーネルを薄く効かせて DB 内の構造衝突(自己79→100%)を
# 解消する。WL を厚くすると実機汎化が落ち、コサインのみ(beta=1.0)では自己検索が 79% に低下。
DEFAULT_BETA = 0.95

# alpha: トポロジースコアの重み。score = alpha*topo + (1-alpha)*tag。
# 既定は 1.0（トポロジーのみ＝タグ非使用）。理由:
#   - 実機クエリ（KiCad 変換）はタグを持たず、タグ項は元々寄与しない。
#   - WL カーネル導入後はトポロジー単独でも自己検索 Hit@1=100%（alpha 0.0〜1.0 全て同率。
#     evaluate.py --alpha-sweep で確認済み）→ タグは識別に不要。
#   - タグ込みだと DB 自己一致だけスコアが底上げされ、自己検索の水増しと、実機クエリとの
#     スコア二分布化（棄却θ校正の阻害）を招く。トポロジー単独に統一すると同一スケールに乗る。
# function_tags は DB 側の任意メタデータとして温存する。alpha<1.0 を明示指定すれば
# タグ類似度をブレンドに復活できる（後方互換・タグ付きクエリを投げる将来用途のため）。
DEFAULT_ALPHA = 1.0


# ─────────────────────────────────────────────────────────
# CircuitRAG
# ─────────────────────────────────────────────────────────

class CircuitRAG:

    def __init__(self, llm: LLMClient | CLILLMClient | MockLLMClient | None = None):
        self.db: list[dict]                   = []
        self.vectors: list[np.ndarray]        = []
        self.block_vectors: list[list[np.ndarray]] = []  # ブロック単位ベクトル（単一ブロック回路は []）
        self.llm = llm
        self.cards = load_cards()             # circuit_id -> 弁別カード（docs/CARD_SPEC.md）

    def _name_lookup(self) -> dict:
        """circuit_id -> circuit_name（カードの confused_with 解決用）。"""
        return {f.get("circuit_id"): f.get("circuit_name") for f in self.db}

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
    def _block_match_optimal(q_vecs: list, d_vecs: list) -> float:
        """
        ブロック単位の最適割当（Hungarian 法 / scipy.linear_sum_assignment）。

        従来のグリーディ割当は局所最適に陥り得る（最初に高スコアのペアを確定
        すると残りで最適化できない）。Hungarian 法は割当全体の類似度和を大域
        最適化する。一方のブロック数が多い場合は余剰ブロックをスコア 0 とみなし、
        max(n_q, n_d) で正規化してブロック数ミスマッチにペナルティを与える
        （従来と同じ正規化＝互換動作）。
        """
        n_q, n_d = len(q_vecs), len(d_vecs)
        if n_q == 0 or n_d == 0:
            return 0.0
        sim = np.array([[cosine_similarity(q, d) for d in d_vecs] for q in q_vecs])
        # linear_sum_assignment はコスト最小化なので符号反転（長方形行列も可）
        row, col = linear_sum_assignment(-sim)
        total = float(sim[row, col].sum())
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
        両方ブロック                  最適割当（Hungarian）ブロックマッチング
        """
        has_q = len(q_bvecs) > 0
        has_d = len(d_bvecs) > 0

        # 全回路 vs 全回路の類似度。ブロックマッチングはこれを「置換」せず
        # max で底上げする補助シグナルとして扱う（フラットなクエリが分解済み
        # DB の部分ブロックとしか比較されず系統的に過小評価される問題を防ぐ。PR #6）。
        whole = cosine_similarity(q_vec, d_vec)

        if not has_q and not has_d:
            return whole
        if has_q and not has_d:
            block = max(cosine_similarity(bv, d_vec) for bv in q_bvecs)
        elif not has_q and has_d:
            block = max(cosine_similarity(q_vec, bv) for bv in d_bvecs)
        else:
            block = self._block_match_optimal(q_bvecs, d_bvecs)
        return max(whole, block)

    # ── 類似検索 ─────────────────────────────────────────

    def search(self, query_features: dict, top_k: int = 3,
               alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> list[dict]:
        """
        alpha: トポロジースコアの重み（0.0〜1.0、既定 DEFAULT_ALPHA=1.0＝トポロジーのみ）。
               残り (1-alpha) がタグ類似度の重み。既定ではタグを使わない（理由は DEFAULT_ALPHA
               のコメント参照）。タグ付きクエリで意図を効かせたい場合のみ alpha<1.0 を明示する。
               クエリにタグが無ければ alpha 値に関わらずタグ項は 0（順位は alpha に不感）。
        beta : トポロジースコア内での コサイン vs WL カーネルの配合
               （topo = beta*cosine + (1-beta)*wl_kernel）。
               クエリが wl_features を持たない（旧DB互換）場合はコサインのみ。
        """
        q_vec   = vectorize(query_features)
        q_bvecs = [vectorize(b) for b in query_features.get("blocks", [])]
        q_tags  = query_features.get("function_tags", [])
        q_wl    = query_features.get("wl_features") or {}
        scores  = []
        for db_feat, db_vec, db_bvecs in zip(self.db, self.vectors, self.block_vectors):
            cos_topo = self._topo_similarity(q_vec, q_bvecs, db_vec, db_bvecs)
            if q_wl:
                wl_topo = wl_kernel(q_wl, db_feat.get("wl_features") or {})
                topo = beta * cos_topo + (1.0 - beta) * wl_topo
            else:
                topo = cos_topo
            tag = tag_similarity(q_tags, db_feat.get("function_tags", []))
            scores.append(alpha * topo + (1.0 - alpha) * tag)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            {"rank": r + 1, "score": round(scores[i], 4), "features": self.db[i]}
            for r, (i, _) in enumerate(ranked[:top_k])
        ]

    def search_with_rejection(self, query_features: dict, top_k: int = 3,
                              alpha: float = DEFAULT_ALPHA,
                              reject_threshold: float = CALIBRATED_REJECT_THRESHOLD
                              ) -> tuple[list[dict], bool, float]:
        """
        未知（DB 未収録）回路を棄却する検索。

        既定の reject_threshold は実機 in28/out20 で校正済み（事前登録ゲート合格・
        CALIBRATED_REJECT_THRESHOLD のコメント参照）。TNR 0.90 / TPR 0.75 で動作するが、
        DB と構造が一致する out（測定試験ベンチ・発振器など）は誤受理し得る（分離余裕 −0.33）。
        誤受理を絶対に避けたい用途や境界回路が多い用途では、硬い二値判定ではなく
        top-k＋スコア提示（ランカー）を人が確認する運用も検討すること。

        棄却判定は **トポロジーのみ(alpha=1.0) の top-1 スコア** で行う。
        これは reject_eval.py の比較で最も分離性能が高かったシグナルで、margin/ratio は
        実機ではほぼ無力だったため採用しない。タグ非依存にするのは、実機クエリがタグを持たず、
        タグ込み絶対スコアが校正用 DB と実機で別分布になるため。

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

    def _fmt(self, f: dict, reveal_labels: bool = True) -> str:
        # 知覚層の構造IRに翻訳してから描画する（circuit_ir が唯一の契約・描画経路）
        # reveal_labels=False で機能タグ・説明を伏せる（判定対象クエリ側のカンニング防止）
        return render_ir(build_ir(f), reveal_labels=reveal_labels)

    def _render_candidate(self, hit: dict, name_lookup: dict) -> str:
        """検索ヒットを候補として描画する。弁別カードがあればそれを、無ければ
        ラベルを伏せた構造IRを示す。ヒットの正体（回路名）を見せること自体は
        RAG の正常動作であり、伏せるのは判定対象クエリ側のラベルのみ。"""
        cid = hit["features"].get("circuit_id")
        card = self.cards.get(cid)
        header = f"[{hit['rank']}位 類似度:{hit['score']:.2f}] {hit['features']['circuit_name']}"
        if card:
            return header + "\n" + render_card(card, name_lookup)
        # カード未整備の回路: ラベル抑制した構造IRで構造だけ提示
        return header + "\n" + self._fmt(hit["features"], reveal_labels=False)

    def build_prompt(self, query_circuit: dict, hits: list[dict] | None = None,
                     sim_result: dict | None = None,
                     top_k: int = 3, alpha: float = DEFAULT_ALPHA) -> tuple[str, str]:
        q_feat = extract_hierarchical_features(query_circuit)

        # 複合回路はブロック単位 RAG に切り替え
        if q_feat.get("is_hierarchical") and q_feat.get("blocks"):
            return self._build_hierarchical_prompt(
                q_feat, sim_result=sim_result, top_k=top_k, alpha=alpha)

        # 単一ブロック：従来の全体照合（hits 未指定なら検索。二重検索を避ける）
        if hits is None:
            hits = self.search(q_feat, top_k=top_k, alpha=alpha)

        system = (
            "あなたは電子回路の認識器です。\n"
            "判定対象回路の«構造IR»（決定的な知覚層の出力）と、構造検索で見つかった"
            "近縁候補が与えられます。候補には«弁別カード»（識別の決め手・"
            "紛らわしい近縁との差分）が付くことがあります。\n"
            "回路をネットリストから推論する必要はありません——構造はIRが消化済みです。\n"
            "あなたの仕事は、IR が示す構造事実とカードの«決め手»を一つずつ照合し、"
            "どの候補に該当するか、どれにも当てはまらなければ『該当なし』を裁定する"
            "ことです。最も紛らわしい候補は、カードの差分を使って明示的に退けてください。\n"
            "回答は必ず以下の形式で出力してください：\n"
            "【判定】<回路名 または 該当なし>\n"
            "【根拠】<IR のどの事実がどの決め手と一致/不一致したか。2〜3文>\n"
            "【近縁との区別】<最も紛らわしい候補をどの差分で退けたか>"
        )

        name_lookup = self._name_lookup()
        refs = "\n\n".join(
            self._render_candidate(h, name_lookup) for h in hits
        )

        user = (
            "## 判定対象回路の構造IR\n\n" + self._fmt(q_feat, reveal_labels=False)
            + _format_sim_for_prompt(sim_result)
            + "\n\n## 近縁候補（構造検索 top-{}）\n\n".format(len(hits)) + refs
            + "\n\n上記の構造IRと候補カードを照合し、判定してください。"
        )
        return system, user

    def _build_hierarchical_prompt(self, q_feat: dict,
                                   sim_result: dict | None = None,
                                   top_k: int = 2,
                                   alpha: float = DEFAULT_ALPHA) -> tuple[str, str]:
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
            + "\n\n## 回路全体の構造IR\n\n"
            + self._fmt(q_feat, reveal_labels=False)
            + _format_sim_for_prompt(sim_result)
            + "\n\n上記ブロック照合結果から、全体回路の種別と構成を特定してください。"
        )
        return system, user

    # ── LLM判定 ──────────────────────────────────────────

    def judge(self, circuit: dict, hits: list[dict] | None = None,
              sim_result: dict | None = None,
              top_k: int = 3, alpha: float = DEFAULT_ALPHA) -> str:
        if self.llm is None:
            raise RuntimeError(
                "LLMが設定されていません。\n"
                "CircuitRAG(llm=LLMClient('anthropic')) のように渡してください。"
            )
        system, user = self.build_prompt(
            circuit, hits=hits, sim_result=sim_result, top_k=top_k, alpha=alpha)
        return self.llm.chat(system=system, user=user)

    # ── シミュレーション統合判定 ──────────────────────────

    def analyze(self, circuit: dict, top_k: int = 3, alpha: float = DEFAULT_ALPHA) -> dict:
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
        default=DEFAULT_ALPHA,
        help="トポロジースコアの重み（0.0〜1.0、デフォルト: 1.0＝トポロジーのみ）。"
             "残り(1-alpha)がタグ類似度の重み。タグ付きクエリで意図を効かせる場合のみ <1.0 を指定",
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
