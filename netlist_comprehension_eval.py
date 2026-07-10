"""
netlist_comprehension_eval.py — LLM の「ネットリスト読解/理解」能力測定ハーネス

仕様: TEST_SPEC_netlist_comprehension.md。予備実験 ir_repr_eval.py（表現比較）を
能力測定へ一般化したもの。表現は §5 に従い「生ネットリスト固定」、変数は
«次元(P/T/Q/R/N/H) × バリアント(V0正準/V1不透明/V2撹乱) × モデル»。

測る2能力:
  読解(reading)  … 書いてある事実の抽出・追跡（P/T/Q/H）
  理解(understanding) … 役割・必要性の説明（R/N、ルーブリック+judge採点）

読む vs 思い出す(§2):
  V0 正準   … DB そのまま（記憶でも解ける）
  V1 不透明 … ノード名/部品IDを relabel（名前の手掛かり除去）
  V2 撹乱   … 構造を一点だけ変える identity-break（記憶で答えると誤る）

採点(§7):
  P/T/Q … 正規表現＋境界で回答スライスを照合（決定的）
  H(罠) … abstain 明言のみ正解。値を断定したら捏造(hallucination)計上
  R/N   … rubric_points 充足で 0/1/2。judge(claude) 3回多数決、モデル名は伏せる(§14-3)

使い方:
  python netlist_comprehension_eval.py --dry-run
  python netlist_comprehension_eval.py --models claude --circuits rc_lowpass_001 ce_amp_npn_001
  python netlist_comprehension_eval.py --models claude qwen2.5:7b
  python netlist_comprehension_eval.py --variance --models claude   # 代表サブセット温度0×3

生出力・採点・監査ファイルは scratchpad に保存（監査可能性）。
"""
from __future__ import annotations

import os
import re
import sys
import json
import copy
import math
import random
import argparse
import datetime as _dt
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

import yaml

from ir_repr_eval import repr_raw, slice_answers
from ir_repr_eval_scale import gen_chain

DB_PATH = "sample_netlists.json"
CORPUS_PATH = "comprehension_corpus.yaml"
SEP = "─" * 74
RAILS = {"GND", "VCC", "VDD", "VEE", "VSS", "VIN", "0"}  # relabel で温存（構造的・答えを漏らさない）
DIMS = ["P", "T", "Q", "R", "N", "H"]

SYSTEM = (
    "あなたは回路の専門家です。与えられたネットリストだけに基づいて答えてください。"
    "netlist から決定できない場合は『判定不能』と答え、推測で値や素子を創作しないこと。"
    "各答えは 'A<番号>: <簡潔な答え>' の形式で、1行ずつ答えてください。"
)

# ── V2 撹乱（identity-breaking）ビルダー：V0回路を一点だけ構造変更 ──────────────

def _by_id(c):
    return {x["id"]: x for x in c["components"]}

def v2_rc_lowpass(c):                       # R↔C 入替 → ハイパス
    c = copy.deepcopy(c); m = _by_id(c)
    m["R1"]["terminals"] = {"p": "N2", "n": "GND"}   # R をシャント側へ
    m["C1"]["terminals"] = {"p": "N1", "n": "N2"}    # C を直列側へ
    return c

def v2_ce_amp(c):                           # Re を NE→GND からコレクタ負荷へ移設＋エミッタ直接GND
    c = copy.deepcopy(c); m = _by_id(c)
    m["Q1"]["terminals"]["emitter"] = "GND"          # 縮退を消す
    m["Re"]["terminals"] = {"p": "VCC", "n": "NC"}   # 役割をコレクタ負荷へ
    return c

def v2_diff_pair(c):                         # 片側エミッタを GND 直結 → 共通テール崩壊
    c = copy.deepcopy(c); m = _by_id(c)
    m["Q2"]["terminals"]["emitter"] = "GND"
    return c

def v2_cascode(c):                           # Q2 ベースを分圧でなく入力側へ → カスコードでない
    c = copy.deepcopy(c); m = _by_id(c)
    m["Q2"]["terminals"]["base"] = "NB1"
    return c

def v2_mfb(c):                               # 帰還 Rf を除去 → 帰還無し
    c = copy.deepcopy(c)
    c["components"] = [x for x in c["components"] if x["id"] != "Rf"]
    return c

V2_BUILDERS = {
    "rc_lowpass_001": v2_rc_lowpass,
    "ce_amp_npn_001": v2_ce_amp,
    "diff_pair_npn_001": v2_diff_pair,
    "cascode_npn_001": v2_cascode,
    "mfb_bpf_001": v2_mfb,
}

# ── V1 不透明化：ノード名(レール以外)と部品IDを seed 決定論で relabel ──────────────

def relabel(c, seed):
    c = copy.deepcopy(c)
    rng = random.Random(seed)
    nodes = sorted({t for x in c["components"] for t in x["terminals"].values()} - RAILS)
    labels = [f"n{k}" for k in range(len(nodes))]
    rng.shuffle(labels)
    nmap = dict(zip(nodes, labels))
    rn = lambda v: nmap.get(v, v)
    for x in c["components"]:
        x["terminals"] = {k: rn(v) for k, v in x["terminals"].items()}
    P = c.get("ports", {})
    c["ports"] = {k: rn(v) for k, v in P.items()}
    # 部品IDを型ごと通し番号でシャッフル（IDから役割/順序を読めなくする）
    comps = c["components"][:]
    rng.shuffle(comps)
    cnt, pre = {}, {"R": "R", "C": "C", "L": "L", "NPN": "Q", "PNP": "Q",
                    "NMOS": "M", "PMOS": "M", "OPAMP": "U", "D": "D", "ZENER": "D", "SW": "S"}
    for x in comps:
        p = pre.get(x["type"], "X"); cnt[p] = cnt.get(p, 0) + 1
        x["id"] = f"{p}{cnt[p]}"
    return c


def build_variants(cid, circuit):
    """cid に対し {V0,V1,V2} を返す。chain_ce_20 は生成回路で V1 のみ。"""
    out = {"V0": circuit, "V1": relabel(circuit, seed=hash(cid) & 0xFFFF)}
    if cid in V2_BUILDERS:
        out["V2"] = V2_BUILDERS[cid](circuit)
    return out


# ── 採点 ─────────────────────────────────────────────────────────────────

def _rx_any(patterns, s):
    return any(re.search(p, s, re.IGNORECASE) for p in patterns)

def grade_regex(answer, check):
    """check = AND-of-OR。全 clause で alt のいずれかが一致すれば正解。"""
    return all(_rx_any(clause, answer) for clause in check)

VALUE_TOKEN = re.compile(r"\d\s*(k|kΩ|kohm|Ω|ohm|µ|u|nf|n|pf|p|mh|m|μ|f|h|v|hz|個|段|本|倍)", re.IGNORECASE)

def grade_abstain(answer, accept_abstain):
    """戻り値 (correct, fabricated)。abstain明言=正解。値断定=捏造計上。"""
    if _rx_any(accept_abstain, answer):
        return True, False
    fabricated = bool(VALUE_TOKEN.search(answer) or re.search(r"(?<!\d)\d+(?!\d)", answer))
    return False, fabricated


def score_objective(qd, variant, answer):
    """P/T/Q/H を採点。dict(correct,fabricated,memory_trap) を返す。"""
    spec = dict(qd)
    ov = qd.get(variant.lower()) if variant != "V0" else None  # 'v1'/'v2' override
    if isinstance(ov, dict):
        spec.update({k: v for k, v in ov.items() if k != "skip"})
    res = {"correct": False, "fabricated": False, "memory_trap": False, "skip": False}
    if isinstance(ov, dict) and ov.get("skip"):
        res["skip"] = True
        return res
    grade = spec.get("grade")
    if grade == "abstain":
        ok, fab = grade_abstain(answer, spec["accept_abstain"])
        res["correct"], res["fabricated"] = ok, fab
    elif grade == "regex":
        ok = grade_regex(answer, spec["check"])
        res["correct"] = ok
        # V2 の教科書的誤答（記憶優先ハルシネーション）を別計上（正解でない時のみ）
        traps = spec.get("memory_trap") or []
        if not ok and traps and _rx_any(traps, answer):
            res["memory_trap"] = True
    return res


# ── ルーブリック judge（R/N、claude固定・モデル名は伏せる）─────────────────────

JUDGE_SYSTEM = (
    "あなたは回路の採点者です。与えられた«要点リスト»の充足数と«誤り例»の有無だけで "
    "0/1/2 を付けてください。推測で要点を補わないこと。\n"
    "2点=要点を2つ以上妥当に述べ誤り例を含まない / 1点=要点を1つ述べる、または部分的に妥当 / "
    "0点=要点ゼロ・誤り例を含む・無回答・的外れ。\n"
    "出力は必ず1行目に半角で 'SCORE: 2' のように 0/1/2 のみを書き、2行目に理由を一行で書くこと。"
)

def extract_score(text):
    """judge 出力からスコアを頑健に抽出。'SCORE: 2'/'【採点】2'/'**2**'/'2点' 等に対応。"""
    text = text.translate(str.maketrans("０１２３４５６７８９：", "0123456789:"))  # 全角→半角
    for pat in (r"SCORE\s*[:：]?\s*\**\s*([012])",
                r"採点[^0-9０-９]{0,6}([012])",
                r"\**([012])\**\s*点",
                r"(?:^|\n)\s*\**([012])\**\s*$"):
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return int(m.group(1))
    return None  # 抽出失敗（多数決から除外し、全失敗なら0扱い）

def judge_rubric(judge_llm, qd, variant, answer, n_vote=3):
    spec = dict(qd)
    ov = qd.get(variant.lower()) if variant != "V0" else None
    if isinstance(ov, dict):
        spec.update({k: v for k, v in ov.items() if k != "skip"})
    pts = "\n".join(f"- {p}" for p in spec.get("rubric_points", []))
    forb = "、".join(spec.get("forbidden", []) or []) or "（なし）"
    user = (f"【質問】{spec['q']}\n【要点(rubric_points)】\n{pts}\n"
            f"【誤り例(forbidden)】{forb}\n【受験者の回答】{answer.strip() or '(無回答)'}")
    votes = []
    for _ in range(n_vote):
        try:
            out = judge_llm.chat(system=JUDGE_SYSTEM, user=user)
        except Exception as e:
            out = f"SCORE: error {e}"
        votes.append((extract_score(out), out.strip()))
    scores = [v[0] for v in votes if v[0] is not None]
    final = max(set(scores), key=scores.count) if scores else 0  # 多数決（抽出失敗は除外）
    return final, [v[0] for v in votes], [v[1] for v in votes]


# ── 統計 ─────────────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    lo, hi = max(0.0, c-h), min(1.0, c+h)
    if k == 0:                       # 3の法則（0/n でも上限≠0）
        hi = max(hi, min(1.0, 3.0/n))
    return (p, lo, hi)

def fmt_rate(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} {p*100:4.0f}% [{lo*100:3.0f},{hi*100:3.0f}]"


# ── モデルクライアント ───────────────────────────────────────────────────

def make_client(name, timeout=300):
    if name == "claude":
        from llm_client import CLILLMClient
        return CLILLMClient(provider="claude", timeout=timeout)
    from ollama_client import OllamaClient
    return OllamaClient(model=name, timeout=timeout)

def probe_claude_model():
    import subprocess, shutil
    exe = shutil.which("claude")
    if not exe:
        return "claude-cli(unknown)"
    try:
        r = subprocess.run([exe, "--print", "--output-format", "json"],
                           input=b"reply with just: ok", capture_output=True, timeout=120)
        j = json.loads(r.stdout.decode("utf-8", "replace"))
        mu = j.get("modelUsage") if isinstance(j, dict) else None
        if isinstance(mu, dict) and mu:
            return "+".join(mu.keys())          # 例: claude-opus-4-8
        for key in ("model", "modelId"):
            if isinstance(j, dict) and j.get(key):
                return j[key]
        return "claude-cli(unknown)"
    except Exception:
        try:
            v = subprocess.run([exe, "--version"], capture_output=True, timeout=30)
            return "claude-cli " + v.stdout.decode("utf-8", "replace").strip()
        except Exception:
            return "claude-cli(unknown)"


# ── 実行 ─────────────────────────────────────────────────────────────────

def build_user(rep_text, qlist):
    qs = "\n".join(f"Q{i+1}: {qd['q']}" for i, qd in enumerate(qlist))
    return f"【ネットリスト】\n{rep_text}\n\n【質問】\n{qs}"

def load_circuits():
    return {c["id"]: c for c in json.load(open(DB_PATH, encoding="utf-8"))["circuits"]}

def make_corpus_circuits(corpus, db, cids):
    """各 cid に対し {variant: circuit_dict} を返す。"""
    out = {}
    for cid in cids:
        if cid == "chain_ce_20":
            c, _gt = gen_chain(20, seed=0)
            out[cid] = {"V1": c}     # 生成＝不透明＝V1相当（単一）
        else:
            out[cid] = build_variants(cid, db[cid])
    return out


def main():
    ap = argparse.ArgumentParser(description="LLM ネットリスト読解/理解 能力測定")
    ap.add_argument("--models", nargs="+", default=["claude"],
                    help="claude / qwen2.5:7b / llama3.1:8b ...")
    ap.add_argument("--circuits", nargs="*", default=None, help="既定は全6回路")
    ap.add_argument("--variants", nargs="*", default=["V0", "V1", "V2"])
    ap.add_argument("--no-judge", action="store_true", help="R/N の judge をスキップ（客観のみ）")
    ap.add_argument("--judge-votes", type=int, default=3)
    ap.add_argument("--variance", action="store_true",
                    help="代表サブセット(V1全問×先頭モデル)を3回実行し分散を見る")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = yaml.safe_load(open(CORPUS_PATH, encoding="utf-8"))
    db = load_circuits()
    cids = args.circuits or list(corpus.keys())
    cvars = make_corpus_circuits(corpus, db, cids)

    if args.dry_run:
        for cid in cids:
            print(f"\n{SEP}\n# {cid}  variants={list(cvars[cid])}")
            for v, c in cvars[cid].items():
                print(f"\n--- {cid} {v} ---\n{repr_raw(c)}")
            print("\n--- 質問 ---")
            for i, qd in enumerate(corpus[cid]):
                ov = qd.get("v2")
                tag = "  [V2 skip]" if isinstance(ov, dict) and ov.get("skip") else \
                      ("  [V2 override]" if ov else "")
                print(f"Q{i+1} [{qd['dim']}/{qd['grade']}] {qd['q']}{tag}")
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    scratch = os.path.join(os.environ.get("TEMP", "."), f"comprehension_eval_{stamp}")
    os.makedirs(scratch, exist_ok=True)
    claude_model = probe_claude_model()
    print(f"実行日: {_dt.date.today()}  frontier(claude) model id: {claude_model}")
    print(f"出力先: {scratch}")

    judge_llm = None
    if not args.no_judge:
        judge_llm = make_client("claude")

    # records: list of dict(model,cid,variant,qid,dim,grade,correct,fabricated,memory_trap,skip,score,answer)
    records = []
    audit_rows = []   # R/N の監査用（モデル名・judge点は別管理で伏せる）

    runs = [(m, 1) for m in args.models]
    if args.variance:
        runs = [(args.models[0], r) for r in (1, 2, 3)]

    for model, trial in runs:
        llm = make_client(model)
        print(f"\n{SEP}\n### MODEL = {model}  (trial {trial})  [{llm}]")
        for cid in cids:
            qlist = corpus[cid]
            variants = [v for v in args.variants if v in cvars[cid]]
            if cid == "chain_ce_20":
                variants = ["V1"]
            for variant in variants:
                c = cvars[cid][variant]
                user = build_user(repr_raw(c), qlist)
                try:
                    out = llm.chat(system=SYSTEM, user=user)
                except Exception as e:
                    print(f"  [{cid} {variant}] LLM失敗: {e}")
                    out = ""
                fn = f"{model.replace(':','_')}__{cid}__{variant}__t{trial}.txt"
                with open(os.path.join(scratch, fn), "w", encoding="utf-8") as f:
                    f.write(f"=== MODEL {model} trial {trial} | {cid} {variant} ===\n"
                            f"=== SYSTEM ===\n{SYSTEM}\n=== USER ===\n{user}\n=== OUTPUT ===\n{out}")
                slices = slice_answers(out, len(qlist))
                marks = []
                for i, qd in enumerate(qlist):
                    ans = slices.get(i + 1, out if not slices else "")
                    rec = {"model": model, "trial": trial, "cid": cid, "variant": variant,
                           "qid": qd["id"], "dim": qd["dim"], "grade": qd["grade"],
                           "correct": False, "fabricated": False, "memory_trap": False,
                           "skip": False, "score": None, "answer": ans.strip()}
                    if qd["grade"] == "rubric":
                        ov = qd.get("v2") if variant == "V2" else None
                        if isinstance(ov, dict) and ov.get("skip"):
                            rec["skip"] = True; marks.append("-")
                        elif judge_llm is not None:
                            sc, votes, reasons = judge_rubric(judge_llm, qd, variant, ans,
                                                              args.judge_votes)
                            rec["score"] = sc
                            audit_rows.append({"cid": cid, "variant": variant, "qid": qd["id"],
                                               "dim": qd["dim"], "q": qd["q"],
                                               "rubric_points": (qd.get("v2", {}).get("rubric_points")
                                                                 if variant == "V2" and isinstance(qd.get("v2"), dict)
                                                                 else qd.get("rubric_points")),
                                               "answer": ans.strip(),
                                               "_model": model, "_judge_score": sc,
                                               "_judge_votes": votes})
                            marks.append(str(sc))
                        else:
                            rec["skip"] = True; marks.append("?")
                    else:
                        r = score_objective(qd, variant, ans)
                        rec.update(r)
                        marks.append("-" if r["skip"] else ("✓" if r["correct"] else
                                     ("M" if r["memory_trap"] else ("!" if r["fabricated"] else "✗"))))
                    records.append(rec)
                print(f"  [{cid:18} {variant}] " + " ".join(
                    f"{qd['dim']}{qd['id'][-1]}:{marks[i]}" for i, qd in enumerate(qlist)))

    # ── 保存 ──
    json.dump(records, open(os.path.join(scratch, "records.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    write_audit(scratch, audit_rows)
    report(records, args.models if not args.variance else [args.models[0]],
           cids, claude_model, scratch, variance=args.variance)
    print(f"\n全生出力・records.json・audit_sample.md: {scratch}")
    return 0


def write_audit(scratch, audit_rows):
    """§14-4 人手監査用：R/N回答の層化抽出（次元×モデルで均等、20%以上）。
    監査者にはモデル名・judge点を伏せる版を出力（_付きキーは別JSONに退避）。"""
    if not audit_rows:
        return
    # 層化: (dim, model) ごとに最低 ceil(20%) を抽出
    by_strata = defaultdict(list)
    for r in audit_rows:
        by_strata[(r["dim"], r["_model"])].append(r)
    rng = random.Random(42)
    sample = []
    for strata, rows in by_strata.items():
        rng.shuffle(rows)
        k = max(1, math.ceil(len(rows) * 0.20))
        sample.extend(rows[:k])
    # 監査シート（盲検）
    lines = ["# R/N 人手監査シート（§14-4 / 盲検：モデル名・judge点は非表示）",
             "", "各回答に 0/1/2 を付け、judge との一致率・Cohen's κ を後で算出する。",
             f"抽出: {len(sample)}/{len(audit_rows)} 件（層化20%以上）", ""]
    key = []
    for i, r in enumerate(sample):
        pts = "\n".join(f"  - {p}" for p in (r["rubric_points"] or []))
        lines += [f"## 監査#{i:02d}  [{r['cid']} {r['variant']} {r['qid']} dim={r['dim']}]",
                  f"質問: {r['q']}", "要点:", pts, f"回答: {r['answer'] or '(無回答)'}",
                  "人手スコア(0/1/2): ____", ""]
        key.append({"idx": i, "cid": r["cid"], "variant": r["variant"], "qid": r["qid"],
                    "model": r["_model"], "judge_score": r["_judge_score"],
                    "judge_votes": r["_judge_votes"]})
    open(os.path.join(scratch, "audit_sample.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(key, open(os.path.join(scratch, "audit_key.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


# ── レポート（§12）─────────────────────────────────────────────────────────

def report(records, models, cids, claude_model, scratch, variance=False):
    main_variants = ["V1", "V2", "V3"]  # 主指標は V1-V3（V0は記憶混入で参考）
    print(f"\n{SEP}\n# レポート（TEST_SPEC §12）  claude={claude_model}")

    def cell(rows):
        obj = [r for r in rows if r["grade"] in ("regex", "abstain") and not r["skip"]]
        k = sum(r["correct"] for r in obj); n = len(obj)
        return k, n

    # 1) 能力プロファイル：行=次元(客観 P/T/Q/H)、列=モデル。主指標 V1+（V2の不変問題含む）
    print("\n## 1. 能力プロファイル（客観 P/T/Q/H、主指標=V1-V3）  正答 [95%CI]")
    print(f"{'dim':4} " + " ".join(f"{m:>22}" for m in models))
    for dim in ["P", "T", "Q", "H"]:
        row = []
        for m in models:
            rows = [r for r in records if r["model"] == m and r["dim"] == dim
                    and r["variant"] in main_variants]
            k, n = cell(rows); row.append(fmt_rate(k, n).rjust(22))
        print(f"{dim:4} " + " ".join(row))
    # V0 参考
    print("-- 参考 V0（記憶混入）--")
    for dim in ["P", "T", "Q", "H"]:
        row = []
        for m in models:
            rows = [r for r in records if r["model"] == m and r["dim"] == dim and r["variant"] == "V0"]
            k, n = cell(rows); row.append(fmt_rate(k, n).rjust(22))
        print(f"{dim:4} " + " ".join(row))

    # 2) R/N（開放・judge平均点と2点率）
    print("\n## 2. 役割/必要性（R/N、judge採点・未監査）  平均点(0-2) / 2点率[95%CI]")
    print(f"{'dim':4} " + " ".join(f"{m:>28}" for m in models))
    for dim in ["R", "N"]:
        row = []
        for m in models:
            rows = [r for r in records if r["model"] == m and r["dim"] == dim
                    and r["grade"] == "rubric" and not r["skip"] and r["score"] is not None]
            if rows:
                avg = sum(r["score"] for r in rows) / len(rows)
                k2 = sum(1 for r in rows if r["score"] == 2)
                row.append(f"{avg:.2f} / {fmt_rate(k2, len(rows))}".rjust(28))
            else:
                row.append("—".rjust(28))
        print(f"{dim:4} " + " ".join(row))

    # 3) 読む vs 思い出す：V0→V2 落差 ＋ V2教科書的誤答(memory_trap)
    print("\n## 3. 読む vs 思い出す（客観 V0 vs V2 正答率、+V2教科書的誤答数）")
    for m in models:
        v0 = cell([r for r in records if r["model"] == m and r["variant"] == "V0"])
        v2 = cell([r for r in records if r["model"] == m and r["variant"] == "V2"])
        mt = sum(r["memory_trap"] for r in records if r["model"] == m and r["variant"] == "V2")
        d0 = v0[0]/v0[1]*100 if v0[1] else 0
        d2 = v2[0]/v2[1]*100 if v2[1] else 0
        print(f"  {m:16}  V0={fmt_rate(*v0)}  V2={fmt_rate(*v2)}  落差={d0-d2:+.0f}pt  "
              f"V2教科書的誤答={mt}件")

    # 4) ハルシネーション（H：棄権率・捏造率）
    print("\n## 4. ハルシネーション（H罠、全バリアント）  棄権率 / 捏造率")
    for m in models:
        h = [r for r in records if r["model"] == m and r["dim"] == "H" and not r["skip"]]
        n = len(h)
        ab = sum(r["correct"] for r in h)
        fab = sum(r["fabricated"] for r in h)
        print(f"  {m:16}  棄権={fmt_rate(ab, n)}   捏造={fmt_rate(fab, n)}")

    # 5) 規模依存（chain_ce_20 客観）
    print("\n## 5. 規模依存（大規模 chain_ce_20、客観）")
    for m in models:
        rows = [r for r in records if r["model"] == m and r["cid"] == "chain_ce_20"]
        k, n = cell(rows)
        if n:
            print(f"  {m:16}  {fmt_rate(k, n)}")

    if variance:
        print("\n## 分散（代表サブセット V1 全問・温度0×3、客観正答数/試行）")
        for cid in cids:
            per = []
            for t in (1, 2, 3):
                rows = [r for r in records if r["trial"] == t and r["cid"] == cid
                        and r["variant"] == "V1"]
                k, n = cell(rows); per.append(f"{k}/{n}")
            print(f"  {cid:18} " + "  ".join(per))


if __name__ == "__main__":
    raise SystemExit(main())
