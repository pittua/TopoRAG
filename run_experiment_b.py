"""run_experiment_b.py — 実験B: 条件③（k=6 カード裁定）評価

設計:
  - in 判定（θ=0.8863 通過）クエリのみ LLM 裁定起動
  - out 判定は閾値棄却のまま（裁定層関与なし）
  - k=6・試行3・多数決（実験 A・② と同一集約規則）
  - 裁定層に「該当なし」逃げを禁止（候補 k 件からの選択に限定）
  - 生応答は results/experiment_b_<date>.jsonl に保存

凍結物（変更禁止）:
  - θ=0.8863（CALIBRATED_REJECT_THRESHOLD）
  - features_db.json / sample_netlists.json (DB)
  - real_corpus.json / real_expected.yaml (コーパス・正解)
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Windows コンソールが cp932 でも日本語・特殊文字を出力できるよう UTF-8 に固定
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

THETA = 0.8863  # 凍結
K = 6           # 凍結（回復対象が rank 5-6 にいるため）
TRIALS = 3      # 凍結（実験 A・②と同一）
DB_PATH = "sample_netlists.json"
CORPUS_PATH = "real_corpus.json"
EXPECTED_PATH = "real_expected.yaml"


# ── データ読み込み ──────────────────────────────────────────────────────────

def load_json_circuits(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "circuits" in data:
        return data["circuits"]
    return data


def load_expected(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── circuit_id 逆引き ────────────────────────────────────────────────────────

def build_name2id(db_raw: list[dict]) -> dict[str, str]:
    """DB 生回路から {circuit_name: circuit_id} を構築。"""
    m = {}
    for c in db_raw:
        cid = c.get("circuit_id") or c.get("id")
        cname = c.get("circuit_name") or c.get("name")
        if cid and cname:
            m[cname] = cid
    return m


def build_id2name(db_raw: list[dict]) -> dict[str, str]:
    return {v: k for k, v in build_name2id(db_raw).items()}


# ── LLM 応答から circuit_id 抽出 ────────────────────────────────────────────

def _lookup_name(text: str, name2id: dict[str, str]) -> str | None:
    """DB 回路名を含む文字列から circuit_id を逆引き（最長一致）。"""
    text = text.strip()
    # 「該当なし」「不明」「unknown」は None
    if re.search(r"該当なし|不明|unknown", text, re.IGNORECASE):
        return None
    # circuit_id が直接書かれている場合（例: halfwave_rect_001）
    if re.match(r"^[a-z][a-z0-9_]+$", text):
        return text if text in {v for v in name2id.values()} else None
    # 完全一致
    if text in name2id:
        return name2id[text]
    # 部分一致（DB 名が text に含まれる場合）—最長の DB 名を優先
    candidates = [(name, cid) for name, cid in name2id.items() if name in text]
    if candidates:
        return max(candidates, key=lambda x: len(x[0]))[1]
    return None


def extract_judgment(raw: str, name2id: dict[str, str]) -> str | None:
    """LLM 応答から circuit_id を抽出。複数フォーマットに対応。

    対応フォーマット:
      1. 【判定】回路名（本来の形式）
      2. **選択：[X位] 回路名** または **回路名**（マークダウン形式）
      3. **最有力：回路名** / **結論：回路名** / **選択：回路名**（マークダウン亜種）
    """
    # 1. 【判定】形式
    m = re.search(r"【判定】(.+)", raw)
    if m:
        return _lookup_name(m.group(1).strip(), name2id)

    # 2. マークダウン ** 形式（LLM が system 指示に従わず ## 見出しで返した場合）
    #    パターン例: **選択：[5位] 非反転増幅回路（OpAmp）**
    #                **結論：反転増幅回路（OpAmp）**
    #                **非反転増幅回路（OpAmp）**
    for pat in [
        r"\*\*(?:選択|結論|最有力|判定)[：:]\s*(?:\[\d+位\]\s*)?(.+?)\*\*",
        r"\*\*(?:\[\d+位\]\s*)?(.+?)\*\*",
    ]:
        for m in re.finditer(pat, raw):
            cid = _lookup_name(m.group(1).strip(), name2id)
            if cid is not None:
                return cid

    return None


def majority_vote(judgments: list[str | None]) -> str | None:
    valid = [j for j in judgments if j is not None]
    if not valid:
        return None
    c = Counter(valid)
    return c.most_common(1)[0][0]


# ── 統計 ────────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def fmt_ci(hit: int, total: int, label: str) -> str:
    lo, hi = wilson_ci(hit, total)
    pct = hit / total * 100 if total else 0.0
    return f"{label}: {hit}/{total} {pct:.1f}% [{lo*100:.1f}–{hi*100:.1f}%]"


# ── プロンプト上書き（「該当なし」禁止版）──────────────────────────────────

_SYSTEM_TMPL = (
    "あなたは電子回路の認識器です。\n"
    "判定対象回路の«構造IR»（決定的な知覚層の出力）と、構造検索で見つかった"
    "近縁候補が与えられます。候補には«弁別カード»（識別の決め手・"
    "紛らわしい近縁との差分）が付くことがあります。\n"
    "回路をネットリストから推論する必要はありません——構造はIRが消化済みです。\n"
    "あなたの仕事は、IR が示す構造事実とカードの«決め手»を一つずつ照合し、"
    "提示された候補の中から最も適合する1つを選ぶことです。\n"
    "★必ず候補の中から1つを選んでください。「該当なし」は選択禁止です。\n"
    "  （候補 k={k} 件の中に正解がある前提で検索を行っています。）\n"
    "最も紛らわしい候補は、カードの差分を使って明示的に退けてください。\n"
    "回答は必ず以下の形式で出力してください：\n"
    "【判定】<候補の回路名>\n"
    "【根拠】<IR のどの事実がどの決め手と一致/不一致したか。2〜3文>\n"
    "【近縁との区別】<最も紛らわしい候補をどの差分で退けたか>"
)


# ── メイン ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="実験B 条件③ (k=6 カード裁定) 評価")
    ap.add_argument("--provider", default="claude", choices=["claude", "mock"])
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--only", default=None,
                    help="デバッグ用: 特定クエリ id をカンマ区切りで指定")
    args = ap.parse_args()

    from feature_extractor import extract_hierarchical_features
    from circuit_rag import CircuitRAG, CALIBRATED_REJECT_THRESHOLD
    from llm_client import CLILLMClient, MockLLMClient

    # ── セットアップ ──────────────────────────────────────────
    db_raw = load_json_circuits(DB_PATH)
    corpus = load_json_circuits(CORPUS_PATH)
    expected = load_expected(EXPECTED_PATH)

    if args.only:
        corpus = [c for c in corpus if c["id"] in args.only.split(",")]

    name2id = build_name2id(db_raw)

    llm = (CLILLMClient(provider="claude") if args.provider == "claude"
           else MockLLMClient())
    rag = CircuitRAG(llm=llm)
    for c in db_raw:
        rag.add(extract_hierarchical_features(c))

    theta = CALIBRATED_REJECT_THRESHOLD
    date_str = datetime.date.today().isoformat()
    out_path = RESULTS_DIR / f"experiment_b_{date_str}.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)

    system_prompt = _SYSTEM_TMPL.format(k=args.k)
    results: dict[str, dict] = {}

    SLEEP_BETWEEN_CALLS = 2.0  # レート制限対策
    MAX_RETRY = 2              # エラー時のリトライ上限

    # ── 既存 JSONL の読み込み（再開サポート）─────────────────
    completed_trials: dict[str, list[dict]] = {}  # qid -> [rec, ...]
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fin:
            for line in fin:
                try:
                    rec = json.loads(line)
                    qid = rec.get("query_id")
                    if qid:
                        completed_trials.setdefault(qid, []).append(rec)
                except json.JSONDecodeError:
                    pass
        print(f"既存 JSONL 読込: {len(completed_trials)} クエリの記録あり")

    # ── 評価ループ ────────────────────────────────────────────
    print(f"実験B 開始  θ={theta}  k={args.k}  trials={args.trials}")
    print(f"DB={len(db_raw)}回路  クエリ={len(corpus)}件")
    print(f"生応答保存: {out_path}\n")

    with out_path.open("a", encoding="utf-8") as fout:
        for circ in corpus:
            qid = circ["id"]
            q_feat = extract_hierarchical_features(circ)

            # 棄却判定: トポロジーのみ(alpha=1.0) top-1 スコア
            top1 = rag.search(q_feat, top_k=1, alpha=1.0)
            topo_conf = top1[0]["score"]
            accepted = topo_conf >= theta

            if not accepted:
                # 棄却済みならスキップ
                prev = completed_trials.get(qid, [])
                if any(r.get("accepted") is False for r in prev):
                    results[qid] = {
                        "accepted": False,
                        "topo_conf": topo_conf,
                        "judgment": None,
                        "trial_judgments": [],
                    }
                    print(f"[棄却 skip] {qid}")
                    continue
                rec = {
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "query_id": qid,
                    "topo_conf": round(topo_conf, 4),
                    "accepted": False,
                    "trial": None,
                    "raw": None,
                    "judgment": None,
                    "error": None,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                results[qid] = {
                    "accepted": False,
                    "topo_conf": topo_conf,
                    "judgment": None,
                    "trial_judgments": [],
                }
                print(f"[棄却] {qid}  topo={topo_conf:.4f}")
                continue

            # 既存の試行を確認（再開サポート）
            prev_trials = [r for r in completed_trials.get(qid, [])
                           if r.get("accepted") is True and r.get("trial") is not None]
            done_trial_nums = {r["trial"] for r in prev_trials}
            prev_judgments = [extract_judgment(r.get("raw") or "", name2id)
                              for r in sorted(prev_trials, key=lambda x: x["trial"])]

            # 全 trial 完了済みならスキップ
            if len(done_trial_nums) >= args.trials:
                final = majority_vote(prev_judgments)
                results[qid] = {
                    "accepted": True,
                    "topo_conf": topo_conf,
                    "judgment": final,
                    "trial_judgments": prev_judgments,
                }
                print(f"[skip 完了済] {qid}  final={final}")
                continue

            # in 判定: k=6 検索 → カード裁定
            hits = rag.search(q_feat, top_k=args.k, alpha=1.0)
            _, user_prompt = rag.build_prompt(circ, hits=hits, top_k=args.k)

            trial_judgments: list[str | None] = list(prev_judgments)
            for trial in range(1, args.trials + 1):
                if trial in done_trial_nums:
                    continue  # 既完了 trial はスキップ

                # リトライループ
                raw, err = "", None
                for attempt in range(MAX_RETRY + 1):
                    if attempt > 0:
                        time.sleep(5.0 * attempt)
                        print(f"    retry {attempt}/{MAX_RETRY}...")
                    try:
                        raw = llm.chat(system=system_prompt, user=user_prompt)
                        err = None
                        break
                    except Exception as e:
                        raw, err = "", f"{type(e).__name__}: {e}"

                j = extract_judgment(raw, name2id) if raw else None
                trial_judgments.append(j)

                rec = {
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "query_id": qid,
                    "topo_conf": round(topo_conf, 4),
                    "accepted": True,
                    "trial": trial,
                    "raw": raw,
                    "judgment": j,
                    "error": err,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                status = "ERROR" if err else f"j={j}"
                print(f"  {qid} t{trial}: {status}  {raw[:70]!r}")
                time.sleep(SLEEP_BETWEEN_CALLS)

            final = majority_vote(trial_judgments)
            results[qid] = {
                "accepted": True,
                "topo_conf": topo_conf,
                "judgment": final,
                "trial_judgments": trial_judgments,
            }
            print(f"[受理] {qid}  topo={topo_conf:.4f}  final={final}"
                  f"  trials={trial_judgments}")

    # ── 採点 ──────────────────────────────────────────────────
    in_strict_hits = 0
    in_lenient_hits = 0
    in_total = 0
    out_rejected = 0
    out_accepted = 0
    out_total = 0

    # ②の正解24件を追跡（③が壊したか確認用）
    in_broken: list[str] = []   # ②が正解で③が誤マッチしたクエリ
    in_recovered: list[str] = []  # ②が誤マッチで③が回復したクエリ

    # ②の結果（検索 top-1・棄却なし）を取得して比較
    rag_no_llm = CircuitRAG()
    for c in db_raw:
        rag_no_llm.add(extract_hierarchical_features(c))
    db_raw2 = load_json_circuits(DB_PATH)

    cond2_correct: set[str] = set()
    corpus_all = load_json_circuits(CORPUS_PATH)
    for circ in corpus_all:
        qid = circ["id"]
        spec = expected.get(qid, {})
        if spec.get("scope") != "in":
            continue
        qf = extract_hierarchical_features(circ)
        h2 = rag_no_llm.search(qf, top_k=1, alpha=1.0)
        top1_id = h2[0]["features"]["circuit_id"]
        top1_score = h2[0]["score"]
        if top1_score >= theta:
            expect_list = spec.get("expect", []) or []
            strict_id = expect_list[0] if expect_list else None
            if top1_id == strict_id:
                cond2_correct.add(qid)

    print("\n" + "=" * 64)
    print("採点結果")
    print("=" * 64)

    detail_lines: list[str] = []
    for circ in load_json_circuits(CORPUS_PATH):
        qid = circ["id"]
        spec = expected.get(qid, {})
        scope = spec.get("scope")
        expect_list = spec.get("expect", []) or []
        strict_id = expect_list[0] if expect_list else None
        expect_ids = set(expect_list)

        res = results.get(qid, {})
        accepted = res.get("accepted", False)
        judgment = res.get("judgment")
        topo_conf = res.get("topo_conf", 0.0)
        trial_j = res.get("trial_judgments", [])

        if scope == "in":
            in_total += 1
            strict_ok = accepted and (judgment == strict_id)
            lenient_ok = accepted and (judgment in expect_ids)
            if strict_ok:
                in_strict_hits += 1
            if lenient_ok:
                in_lenient_hits += 1

            # ②との比較
            was_cond2_correct = qid in cond2_correct
            if was_cond2_correct and not strict_ok:
                in_broken.append(qid)
            if not was_cond2_correct and strict_ok:
                in_recovered.append(qid)

            status = ("✓ strict" if strict_ok else
                      ("△ lenient" if lenient_ok else "✗ miss"))
            detail_lines.append(
                f"  {status:10s}  {qid:35s}  "
                f"judgment={judgment}  expect={strict_id}  "
                f"topo={topo_conf:.4f}  acc={accepted}  trials={trial_j}"
            )
        elif scope == "out":
            out_total += 1
            if not accepted:
                out_rejected += 1
            else:
                out_accepted += 1
                detail_lines.append(
                    f"  {'誤受理':10s}  {qid:35s}  "
                    f"judgment={judgment}  topo={topo_conf:.4f}"
                )

    print("\n■ in-scope 詳細:")
    for line in sorted(detail_lines):
        print(line)

    lo_s, hi_s = wilson_ci(in_strict_hits, in_total)
    lo_l, hi_l = wilson_ci(in_lenient_hits, in_total)
    lo_r, hi_r = wilson_ci(out_rejected, out_total)

    print("\n■ 集計:")
    print(f"  {fmt_ci(in_strict_hits, in_total,  'in strict  Hit@1')}")
    print(f"  {fmt_ci(in_lenient_hits, in_total, 'in lenient Hit@1')}")
    print(f"  {fmt_ci(out_rejected, out_total,   'out 棄却成功    ')}")

    print("\n■ ②との比較（成功基準照合）:")
    print(f"  ②が正解で③が壊した件数: {len(in_broken)}  {in_broken if in_broken else '（なし）'}")
    print(f"  ②が誤マッチで③が回復:  {len(in_recovered)}  {in_recovered if in_recovered else '（なし）'}")

    print("\n■ 設定記録（凍結）:")
    print(f"  k={args.k}  trials={args.trials}  θ={theta}")
    print(f"  カード枚数: {len(rag.cards)}  provider={args.provider}")

    print(f"\n■ 生応答: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
