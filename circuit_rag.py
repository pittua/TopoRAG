"""
回路ネットリスト 類似検索RAGシステム v3
  - LLMClient 経由で Anthropic / Gemini を切り替え可能
  - ハイパス/ローパス識別対応
"""

import json
import numpy as np
from feature_extractor import extract_all_features, extract_hierarchical_features
from llm_client import LLMClient, CLILLMClient, MockLLMClient


# ─────────────────────────────────────────────────────────
# ベクトル化（20次元）
# ─────────────────────────────────────────────────────────

def vectorize(features: dict) -> np.ndarray:
    A  = features["A_component"]
    B1 = features["B1_order"]
    B2 = features["B2_diode"]
    B3 = features["B3_series_parallel"]
    C  = features["C_node"]

    order_map = {"SW_before_L": 1.0, "L_before_SW": -1.0, None: 0.0}
    first = B1.get("first_series_type")
    total = sum(A["component_counts"].values())

    return np.array([
        float(A["has_resistor"]),           # 0
        float(A["has_capacitor"]),          # 1
        float(A["has_inductor"]),           # 2
        float(A["has_switch"]),             # 3
        float(A["has_diode"]),              # 4
        min(total / 10.0, 1.0),            # 5
        order_map.get(B1["sw_l_order"], 0.0),  # 6
        float(first == "R"),               # 7
        float(first == "C"),               # 8
        float(first == "L"),               # 9
        float(B2["diode_anode_to_gnd"]),   # 10
        float(B2["diode_cathode_to_out"]), # 11
        float(B2["diode_anode_to_out"]),   # 12
        float(B2["diode_cathode_to_gnd"]), # 13
        float(B3["has_parallel_components"]),      # 14
        min(B3["series_chain_length"] / 5.0, 1.0), # 15
        min(C["node_count"] / 10.0, 1.0), # 16
        float(C["has_high_degree_node"]),  # 17
        min(C["cycle_count"] / 5.0, 1.0), # 18
        float(A.get("has_zener", False)),  # 19
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
        if f.get("is_hierarchical") and f.get("blocks"):
            lines.append(f"  ブロック構成     : {f['n_blocks']} ブロック")
            for i, blk in enumerate(f["blocks"]):
                lines.append(
                    f"    ブロック{i + 1}: 部品={blk['A_component']['component_types']}"
                    f"  直列={blk['B1_order']['series_type_sequence']}"
                    f"  GND並列={blk['B1_order']['shunt_type_sequence']}"
                )
        return "\n".join(lines)

    def build_prompt(self, query_circuit: dict, top_k: int = 3,
                     alpha: float = 0.7) -> tuple[str, str]:
        q_feat = extract_hierarchical_features(query_circuit)

        # 複合回路はブロック単位 RAG に切り替え
        if q_feat.get("is_hierarchical") and q_feat.get("blocks"):
            return self._build_hierarchical_prompt(q_feat, top_k=top_k, alpha=alpha)

        # 単一ブロック：従来の全体照合
        hits = self.search(q_feat, top_k=top_k, alpha=alpha)

        system = (
            "あなたは電子回路の専門家です。\n"
            "与えられたネットリストの特徴量と、検索された類似回路の情報をもとに、"
            "入力回路のトポロジー（回路種別）を判定してください。\n"
            "回答は必ず以下の形式で出力してください：\n"
            "【判定】<回路名>\n"
            "【根拠】<接続構造に基づく理由を2〜3文>\n"
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
            + "\n\n上記をもとに回路のトポロジーを判定してください。"
        )
        return system, user

    def _build_hierarchical_prompt(self, q_feat: dict, top_k: int = 2,
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
            + "\n\n上記ブロック照合結果から、全体回路の種別と構成を特定してください。"
        )
        return system, user

    # ── LLM判定 ──────────────────────────────────────────

    def judge(self, circuit: dict, top_k: int = 3, alpha: float = 0.7) -> str:
        if self.llm is None:
            raise RuntimeError(
                "LLMが設定されていません。\n"
                "CircuitRAG(llm=LLMClient('anthropic')) のように渡してください。"
            )
        system, user = self.build_prompt(circuit, top_k=top_k, alpha=alpha)
        return self.llm.chat(system=system, user=user)


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

    # ── LLM判定 ──────────────────────────────────────────
    print("\n--- LLM判定 ---")
    for t in queries:
        print(f"\n{'='*55}\n入力: {t['name']}\n{'='*55}")
        print(rag.judge(t, top_k=args.top_k, alpha=args.alpha))
