"""run_baseline_direct.py — 実験A: LLM直読みベースラインの実行

real_corpus.json の48回路を生ネットリストJSONのままLLMに渡し、回路種別を回答させる。
プロンプトは本ファイル内の固定文字列（凍結物）。生応答は results/*.jsonl に全件追記保存する。

使い方:
    python run_baseline_direct.py --provider mock   --prompt p1 --trials 3   # ドライラン
    python run_baseline_direct.py --provider claude --prompt p1 --trials 3
    python run_baseline_direct.py --provider gemini --prompt p2 --trials 1

前提: TopoRAG リポジトリ直下に本ファイルを置く（llm_client.py / real_corpus.json /
features_db.json を参照する）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SHUFFLE_SEED = 42  # 候補一覧の固定シャッフル（凍結物）

# ── プロンプト（凍結物。実行開始後に変更しない） ─────────────────────────────

P1_SYSTEM = (
    "あなたは電子回路の専門家です。与えられたネットリスト（JSON形式の回路接続情報）を読み、"
    "候補一覧の中から回路種別を1つ選んでください。\n"
    "どの候補にも該当しないと判断した場合は必ず \"unknown\" と答えてください。\n"
    "出力は次のJSONのみとし、他のテキストを一切含めないでください:\n"
    '{"answer_id": "<候補のcircuit_id または unknown>", '
    '"confidence": "high|medium|low", "reason": "<根拠を1文で>"}'
)

P1_USER_TEMPLATE = (
    "## 候補一覧（この中から1つ選ぶか unknown と答える）\n"
    "{candidates}\n\n"
    "## 判定対象のネットリスト\n"
    "```json\n{netlist}\n```\n\n"
    "指定のJSON形式のみで回答してください。"
)

P2_SYSTEM = (
    "あなたは電子回路の専門家です。与えられたネットリスト（JSON形式の回路接続情報）を読み、"
    "この回路が何であるかを答えてください。判別できない場合は「不明」と答えてください。\n"
    "出力は次のJSONのみとし、他のテキストを一切含めないでください:\n"
    '{"answer": "<回路名。不明なら 不明>", "reason": "<根拠を1〜2文で>"}'
)

P2_USER_TEMPLATE = (
    "## 判定対象のネットリスト\n"
    "```json\n{netlist}\n```\n\n"
    "指定のJSON形式のみで回答してください。"
)

# ── データ読み込み ───────────────────────────────────────────────────────────


def load_corpus() -> list[dict]:
    data = json.loads((ROOT / "real_corpus.json").read_text(encoding="utf-8"))
    return data["circuits"] if isinstance(data, dict) else data


def load_candidates() -> list[tuple[str, str]]:
    """features_db.json のトップレベル47回路 (circuit_id, circuit_name) を固定順で返す。"""
    data = json.loads((ROOT / "features_db.json").read_text(encoding="utf-8"))
    items = data["circuits"] if isinstance(data, dict) and "circuits" in data else data
    cands = [(it["circuit_id"], it.get("circuit_name", it["circuit_id"])) for it in items]
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(cands)  # 全クエリ・全モデル共通の固定順（凍結物）
    return cands


def build_client(provider: str):
    if provider == "mock":
        from llm_client import MockLLMClient
        return MockLLMClient()
    if provider in ("claude", "gemini"):
        from llm_client import CLILLMClient
        return CLILLMClient(provider=provider)
    if provider in ("anthropic", "gemini-sdk"):
        from llm_client import LLMClient
        return LLMClient(provider="anthropic" if provider == "anthropic" else "gemini")
    raise ValueError(f"unknown provider: {provider}")


# ── 実行 ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True,
                    choices=["mock", "claude", "gemini", "anthropic", "gemini-sdk"])
    ap.add_argument("--prompt", required=True, choices=["p1", "p2"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--only", default=None, help="デバッグ用: 特定クエリidのみ実行")
    args = ap.parse_args()

    corpus = load_corpus()
    if args.only:
        corpus = [c for c in corpus if c["id"] == args.only]

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"baseline_{args.prompt}_{args.provider}.jsonl"

    if args.prompt == "p1":
        cands = load_candidates()
        cand_text = "\n".join(f"- {cid}: {name}" for cid, name in cands)
        system = P1_SYSTEM
    else:
        cand_text = None
        system = P2_SYSTEM

    client = build_client(args.provider)
    total = len(corpus) * args.trials
    done = 0

    with out_path.open("a", encoding="utf-8") as f:
        for circ in corpus:
            netlist = json.dumps(circ, ensure_ascii=False, indent=1)
            if args.prompt == "p1":
                user = P1_USER_TEMPLATE.format(candidates=cand_text, netlist=netlist)
            else:
                user = P2_USER_TEMPLATE.format(netlist=netlist)

            for trial in range(1, args.trials + 1):
                done += 1
                try:
                    raw = client.chat(system=system, user=user)
                    err = None
                except Exception as e:  # 呼び出し失敗も記録して継続
                    raw, err = "", f"{type(e).__name__}: {e}"
                rec = {
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "provider": args.provider,
                    "prompt": args.prompt,
                    "query_id": circ["id"],
                    "trial": trial,
                    "raw": raw,
                    "error": err,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{done}/{total}] {circ['id']} trial{trial} "
                      f"{'ERROR: ' + err if err else 'ok'}")

    print(f"\n保存先: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
