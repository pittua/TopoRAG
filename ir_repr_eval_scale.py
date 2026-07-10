"""
ir_repr_eval_scale.py — 大規模回路での表現別 LLM 読解可能性テスト

ir_repr_eval.py（小〜中規模）の続き。仮説「生netlistはスケールで破綻し、
決定的な消化が要る」を検証する。

設計（生netlistが楽をできないように意図的に難しくする）:
  - N段エミッタ接地アンプの直列鎖を生成する（部品 6N 個）。
  - ノード名は«不透明»（n0,n1,… をシャッフル）。部品IDも型ごとに通し番号で
    シャッフル。→ 名前から段順は読めず、信号経路をたどらないと
    「入力から3番目の段」を特定できない。これがスケール時の navigation 負荷。
  - 段kのエミッタ抵抗値を k(kΩ) と段ごとに変える → 値の質問が真の探索になる。

3表現は ir_repr_eval の C0_raw / C1_ir / C2_rich を再利用。
採点は値・ノード名の取り違えを避けるため正規表現（境界つき）で行う。

使い方:
  python ir_repr_eval_scale.py --provider claude --sizes 10 20
  python ir_repr_eval_scale.py --provider claude --sizes 10 20 40
  python ir_repr_eval_scale.py --dry-run --sizes 10
"""
from __future__ import annotations

import os
import re
import sys
import json
import random
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from ir_repr_eval import BUILDERS, SYSTEM, build_user, slice_answers

SEP = "─" * 70


def gen_chain(n: int, seed: int = 0) -> tuple[dict, dict]:
    """N段CE鎖を生成。(circuit_dict, ground_truth) を返す。ノード/ID は不透明化。"""
    rng = random.Random(seed)
    comps = []
    stages = []
    prev_nc = None
    for i in range(1, n + 1):
        nb, nc, ne = f"NB{i}", f"NC{i}", f"NE{i}"
        src = "VIN" if i == 1 else prev_nc
        comps += [
            {"id": f"_cpl{i}", "type": "C", "value": "1u", "terminals": {"p": src, "n": nb}},
            {"id": f"_r1_{i}", "type": "R", "value": "47k", "terminals": {"p": "VCC", "n": nb}},
            {"id": f"_r2_{i}", "type": "R", "value": "10k", "terminals": {"p": nb, "n": "GND"}},
            {"id": f"_q{i}", "type": "NPN", "value": None,
             "terminals": {"base": nb, "collector": nc, "emitter": ne}},
            {"id": f"_rc{i}", "type": "R", "value": "4.7k", "terminals": {"p": "VCC", "n": nc}},
            {"id": f"_re{i}", "type": "R", "value": f"{i}k", "terminals": {"p": ne, "n": "GND"}},
        ]
        stages.append({"nb": nb, "nc": nc, "ne": ne, "re": f"{i}k"})
        prev_nc = nc
    out_node = f"NC{n}"

    # ノード不透明化（VIN/VCC/GND は残す）
    internal = sorted({t for c in comps for t in c["terminals"].values()}
                      - {"VIN", "VCC", "GND"})
    labels = [f"n{k}" for k in range(len(internal))]
    rng.shuffle(labels)
    nmap = dict(zip(internal, labels))
    rn = lambda x: nmap.get(x, x)
    for c in comps:
        c["terminals"] = {k: rn(v) for k, v in c["terminals"].items()}

    # 部品順シャッフル＋型ごと通し番号で再ID付与（ID から段順を読めなくする）
    rng.shuffle(comps)
    cnt: dict[str, int] = {}
    pre = {"R": "R", "C": "C", "NPN": "Q"}
    for c in comps:
        p = pre[c["type"]]
        cnt[p] = cnt.get(p, 0) + 1
        c["id"] = f"{p}{cnt[p]}"

    circuit = {
        "id": f"chain_ce_{n}", "name": f"{n}-stage CE chain",
        "ports": {"input": "VIN", "output": rn(out_node), "gnd": "GND"},
        "components": comps,
    }
    gt = {
        "n": n, "n_res": 4 * n, "out": rn(out_node),
        "parity": "非反転" if n % 2 == 0 else "反転",
        "stage3_re": stages[2]["re"],                     # 入力から3番目の Re 値
        "couple34": (rn(stages[2]["nc"]), rn(stages[3]["nb"])),  # 3-4段結合の両端
    }
    return circuit, gt


def questions_for(gt: dict) -> list[dict]:
    """gt から (質問文, 採点正規表現clause群) を作る。clause: 全clause成立で正解。"""
    n = gt["n"]
    a, b = gt["couple34"]
    return [
        {"q": "この回路にトランジスタ（NPN）は何個ありますか。",
         "rx": [[rf"(?<!\d){n}(?!\d)"]]},
        {"q": "抵抗は全部で何個ありますか。",
         "rx": [[rf"(?<!\d){gt['n_res']}(?!\d)"]]},
        {"q": ("入力ポート VIN から、結合コンデンサ→トランジスタ→段間結合コンデンサ…と"
               "信号経路を順にたどってください。入力から数えて3番目のトランジスタの、"
               "エミッタとGNDの間に入っている抵抗の値はいくつですか。"),
         "rx": [[r"(?<!\d)3\s*k"]]},
        {"q": "各増幅段は信号を反転させます。全体として入力に対し最終出力は反転ですか非反転ですか。",
         "rx": [[r"非反転", r"同相", r"正転", r"反転しない"]] if n % 2 == 0
               else [[r"(?<!非)反転"]]},
        {"q": ("入力から数えて3番目のトランジスタのコレクタと、4番目のトランジスタのベースを"
               "結ぶ結合コンデンサがあります。その両端のノード名を2つ答えてください。"),
         "rx": [[rf"\b{re.escape(a)}\b"], [rf"\b{re.escape(b)}\b"]]},
    ]


def grade_rx(answer: str, rx: list[list[str]]) -> bool:
    s = answer
    return all(any(re.search(alt, s, re.IGNORECASE) for alt in clause) for clause in rx)


def main() -> int:
    ap = argparse.ArgumentParser(description="大規模回路の表現別読解テスト")
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "claude"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        c, gt = gen_chain(args.sizes[0], args.seed)
        print(f"# {c['id']}  部品数={len(c['components'])}  out={gt['out']}")
        for name, fn in BUILDERS.items():
            text = fn(c)
            print(f"\n--- {name}  ({len(text)}文字) ---\n{text[:1200]}"
                  + ("\n…(略)" if len(text) > 1200 else ""))
        print("\n--- 質問 ---")
        for i, q in enumerate(questions_for(gt)):
            print(f"Q{i+1}: {q['q']}\n     rx={q['rx']}")
        return 0

    from llm_client import CLILLMClient, LLMClient
    llm = (CLILLMClient(provider=args.provider, timeout=300)
           if args.provider in ("claude", "gemini")
           else LLMClient(provider=args.provider))
    print(f"LLM = {llm}")

    scratch = os.path.join(os.environ.get("TEMP", "."), "ir_repr_eval_scale_outputs")
    os.makedirs(scratch, exist_ok=True)
    grid = {}  # (n, rep) -> (correct, total, marks)

    for n in args.sizes:
        c, gt = gen_chain(n, args.seed)
        qs = questions_for(gt)
        print(f"\n{SEP}\n# {c['id']}  部品数={len(c['components'])}  質問{len(qs)}件")
        for name, fn in BUILDERS.items():
            rep = fn(c)
            user = build_user(rep, qs)
            try:
                out = llm.chat(system=SYSTEM, user=user)
            except Exception as e:
                print(f"  [{name}] LLM 失敗: {e}")
                grid[(n, name)] = (0, len(qs), "ERR")
                continue
            with open(os.path.join(scratch, f"{c['id']}__{name}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(f"=== chars={len(rep)} ===\n=== USER ===\n{user}\n\n=== OUTPUT ===\n{out}")
            sl = slice_answers(out, len(qs))
            marks, cor = [], 0
            for i, q in enumerate(qs):
                ans = sl.get(i + 1, out if not sl else "")
                ok = grade_rx(ans, q["rx"])
                cor += ok
                marks.append("✓" if ok else "✗")
            grid[(n, name)] = (cor, len(qs), "".join(marks))
            print(f"  [{name:8}] {cor}/{len(qs)}  {''.join(marks)}  (表現{len(rep)}文字)")

    print(f"\n{SEP}\n[サマリ] 規模 × 表現")
    print(f"  {'規模(部品)':14} " + " ".join(f"{n:>12}" for n in BUILDERS))
    for n in args.sizes:
        row = [f"{grid[(n,nm)][0]}/{grid[(n,nm)][1]}".rjust(12) for nm in BUILDERS]
        print(f"  N={n:<3}({6*n:>3}部品) " + " ".join(row))
    print(f"\n生出力: {scratch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
