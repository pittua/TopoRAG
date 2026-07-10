"""
ir_repr_eval.py — 回路表現の「LLM 読解可能性」A/B/C 比較ハーネス

仮説:
  「LLM はグラフ／ネットリストを読めない」(CIRCUIT ベンチ等) が真なら、
  回路をどう"表現"して渡すかで、回路についての質問への正答率が変わるはず。
  特に「分からないことをAIに聞くように回路を聞ける」ゴールには、
  値・接続・役割を保持した«消化済み»表現が要る、という主張を実機で検証する。

比較する3表現（同一回路・同一質問・同一 system プロンプト。表現だけ替える）:
  C0 raw   : 生ネットリスト（部品・端子・値・生ノード名のみ。解釈なし）
  C1 ir    : 現行の構造IR（circuit_ir.render_ir, ラベル抑制）。消化済みだが
             値・個別部品・接続を捨てている（＝照合用の指紋）
  C2 rich  : 役割注釈つき接続記述（ポート相対のノード・値・役割を保持。
             生データから決定的に生成。＝推論用の完全記述の試作）

採点:
  各質問に「正解トークン節（AND）× 代替トークン（OR）」を事前登録し、
  LLM 回答の該当スライスに含まれるかで決定的に採点（LLM-as-judge を使わない）。

使い方:
  LLM_PROVIDER=claude python ir_repr_eval.py
  python ir_repr_eval.py --provider claude --circuits ce_amp_npn_001 rc_lowpass_001
  python ir_repr_eval.py --dry-run     # LLM を呼ばず表現とプロンプトだけ表示

生 LLM 出力は scratchpad に保存（監査可能性のため）。
"""
from __future__ import annotations

import os
import re
import sys
import json
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from feature_extractor import extract_hierarchical_features
from circuit_ir import build_ir, render_ir

DB_PATH = "sample_netlists.json"
SEP = "─" * 70

# ── 表現ビルダー ────────────────────────────────────────────────

ACTIVE_TYPES = ("NPN", "PNP", "NMOS", "PMOS")


def _portlabel(n: str, ports: dict) -> str:
    if n == ports.get("input"):
        return f"{n}(IN)"
    if n == ports.get("input2"):
        return f"{n}(IN2)"
    if n == ports.get("output"):
        return f"{n}(OUT)"
    if n == ports.get("gnd"):
        return f"{n}(GND)"
    return n


def repr_raw(c: dict) -> str:
    """C0: 生ネットリスト（解釈なし）。"""
    P = c.get("ports", {})
    lines = [f"ports: input={P.get('input')} output={P.get('output')} gnd={P.get('gnd')}"
             + (f" input2={P['input2']}" if P.get("input2") else ""),
             "components:"]
    for x in c["components"]:
        t = " ".join(f"{k}={v}" for k, v in x["terminals"].items())
        val = f" value={x['value']}" if x.get("value") else ""
        lines.append(f"  {x['id']} {x['type']}{val}  {t}")
    return "\n".join(lines)


def repr_ir(c: dict) -> str:
    """C1: 現行の構造IR（ラベル抑制＝クエリ相当）。"""
    f = extract_hierarchical_features(c)
    return render_ir(build_ir(f), reveal_labels=False)


def repr_rich(c: dict) -> str:
    """C2: 役割注釈つき接続記述。生データから決定的に生成。"""
    P = c.get("ports", {})
    GND = P.get("gnd")
    comps = c["components"]
    roles: dict[str, str] = {}

    # 能動素子相対の役割（コレクタ/ドレイン負荷・エミッタ/ソース縮退・バイアス分圧・入力結合）
    for q in comps:
        if q["type"] not in ACTIVE_TYPES:
            continue
        t = q["terminals"]
        col = t.get("collector") or t.get("drain")
        emi = t.get("emitter") or t.get("source")
        bas = t.get("base") or t.get("gate")
        for r in comps:
            if r["type"] != "R":
                continue
            nodes = set(r["terminals"].values())
            rail = {"VCC", "VDD", "VEE", "VSS"} & nodes
            if col in nodes and rail:
                roles[r["id"]] = "コレクタ/ドレイン負荷"
            elif emi in nodes and GND in nodes:
                roles[r["id"]] = "エミッタ/ソース縮退"
            elif bas in nodes and rail:
                roles[r["id"]] = "バイアス分圧(上)"
            elif bas in nodes and GND in nodes:
                roles[r["id"]] = "バイアス分圧(下)"
        for cap in comps:
            if cap["type"] == "C" and bas in cap["terminals"].values() \
                    and P.get("input") in cap["terminals"].values():
                roles[cap["id"]] = "入力結合(AC)"

    # 受動素子の汎用役割（未注釈のもののみ）
    for x in comps:
        if x["id"] in roles or x["type"] in ACTIVE_TYPES:
            continue
        nodes = set(x["terminals"].values())
        if GND in nodes:
            roles[x["id"]] = "GNDシャント(片端GND)"
        elif {P.get("input"), P.get("output")} & nodes:
            roles[x["id"]] = "信号経路上の直列素子"

    lines = ["回路の接続（ノードは IN/OUT/GND を明示。役割は構造から決定的に付与）:"]
    for x in comps:
        t = " , ".join(f"{k}={_portlabel(v, P)}" for k, v in x["terminals"].items())
        val = f" 値={x['value']}" if x.get("value") else ""
        rl = f"   ◀ {roles[x['id']]}" if x["id"] in roles else ""
        lines.append(f"  {x['id']}({x['type']}){val}: {t}{rl}")
    return "\n".join(lines)


BUILDERS = {"C0_raw": repr_raw, "C1_ir": repr_ir, "C2_rich": repr_rich}


# ── 質問バッテリ（事前登録の採点規則）──────────────────────────
# check: list[clause]、clause: list[alt]。各 clause につき alt のいずれかが
# 回答スライスに(大小無視で)含まれれば合格。全 clause 合格でその質問は正解。

QUESTIONS: dict[str, list[dict]] = {
    "ce_amp_npn_001": [
        {"q": "この回路は反転増幅か非反転増幅か。", "check": [["反転", "inverting", "invert"]]},
        {"q": "出力はどのノードから取り出されるか（トランジスタのコレクタが繋がるノード名）。",
         "check": [["NC", "OUT", "出力"]]},
        {"q": "ベースのバイアス用抵抗2本の値はそれぞれいくらか。",
         "check": [["47k", "47 k", "47kΩ", "47kohm"], ["10k", "10 k", "10kΩ", "10kohm"]]},
        {"q": "コレクタ負荷抵抗の値はいくらか。", "check": [["4.7k", "4.7 k", "4k7", "4.7kΩ"]]},
        {"q": "小信号電圧利得の大きさを概算せよ。",
         "check": [["4.7", "4.6", "4.5", "4.8", "約5", "5倍", "4倍", "×5", "x5", "4〜5", "4-5", "4 ~ 5"]]},
        {"q": "入力とトランジスタのベースの間に挿入されている部品は何か。",
         "check": [["Cin", "コンデンサ", "capacitor", "キャパシタ", "1u", "結合", "カップリング"]]},
    ],
    "rc_lowpass_001": [
        {"q": "この回路はローパスかハイパスか。", "check": [["ローパス", "lowpass", "low-pass", "低域"]]},
        {"q": "カットオフ周波数を概算せよ（Hz）。",
         "check": [["1592", "1591", "1.59", "1.6k", "1.5k", "1600", "1500", "約1.6", "約1.5", "1.6 k", "1.5 k"]]},
        {"q": "GNDへ並列に接続されている（シャント）部品はどれか。",
         "check": [["C1", "コンデンサ", "capacitor", "100n", "キャパシタ"]]},
        {"q": "抵抗とコンデンサの値はそれぞれいくらか。",
         "check": [["1k", "1 k", "1kΩ", "1000"], ["100n", "100 n", "0.1u", "100nf"]]},
    ],
    "zener_regulator_001": [
        {"q": "この回路の機能は何か。",
         "check": [["定電圧", "安定化", "レギュレ", "regulat", "reference", "電圧基準", "クランプ"]]},
        {"q": "ツェナーダイオードのカソードはどのノードに接続されているか。",
         "check": [["N1", "OUT", "出力"]]},
        {"q": "直列抵抗の値はいくらか。", "check": [["1k", "1 k", "1kΩ", "1000"]]},
        {"q": "ツェナーは出力に対して直列か並列(シャント)か。",
         "check": [["並列", "シャント", "shunt", "parallel"]]},
    ],

    # ── 複雑回路（多ホップ推論を要する難問。生netlistが破綻するか検証）──
    "two_stage_ce_001": [
        {"q": "増幅段は何段あるか。", "check": [["2段", "2 段", "二段", "2つ", "two", "2 stage", "2-stage"]]},
        {"q": "1段目と2段目を結合している部品はどれか。",
         "check": [["Cc", "結合コンデンサ", "段間", "カップリング"]]},
        {"q": "全体の電圧利得の大きさを概算せよ。",
         "check": [["22", "20", "21", "23", "24", "4.7^2", "4.7²", "4.7×4.7", "二乗", "約22", "≈22"]]},
        {"q": "入力に対して最終出力は反転か非反転か（2段の符号を考慮せよ）。",
         "check": [["非反転", "non-invert", "noninvert", "同相", "正転", "反転しない"]]},
        {"q": "最終出力ノードはどれか。", "check": [["NC2"]]},
    ],
    "cascode_npn_001": [
        {"q": "このトポロジーの名称は何か。", "check": [["カスコード", "cascode"]]},
        {"q": "上段トランジスタ Q2 のエミッタはどこに接続されているか。",
         "check": [["NC1", "Q1", "コレクタ", "collector", "下段"]]},
        {"q": "上段 Q2 の構成は CE/CB/CC のどれか（接地エミッタ/ベース接地/コレクタ接地）。",
         "check": [["ベース接地", "共通ベース", "common-base", "common base", "CB", "ベース接地増幅"]]},
        {"q": "最終出力ノードはどれか。", "check": [["NC2"]]},
    ],
    "mfb_bpf_001": [
        {"q": "この能動フィルタの種別は何か（LPF/HPF/BPF/BSF）。",
         "check": [["バンドパス", "帯域通過", "band-pass", "bandpass", "BPF", "band pass"]]},
        {"q": "出力から反転入力へ戻る帰還抵抗はどれか。",
         "check": [["Rf", "100k", "帰還抵抗"]]},
        {"q": "OpAmp の非反転入力(in_p)はどこに接続されているか。",
         "check": [["GND", "接地", "グランド", "ground"]]},
        {"q": "出力から戻る帰還経路は何本あるか（多重帰還）。",
         "check": [["2", "二", "2本", "2つ", "two", "multiple", "多重"]]},
    ],
    "diff_pair_npn_001": [
        {"q": "このトポロジーの名称は何か。",
         "check": [["差動", "ロングテール", "long-tail", "long tail", "differential", "エミッタ結合"]]},
        {"q": "Q1 と Q2 はどの端子を共通に接続しているか。",
         "check": [["エミッタ", "emitter", "NTAIL", "tail", "テール"]]},
        {"q": "テール（共通エミッタ→GND）抵抗の値はいくらか。",
         "check": [["10k", "10 k", "10kΩ"]]},
        {"q": "差動出力はどのノードから取り出すか（2つ）。",
         "check": [["NC1"], ["NC2"]]},
    ],
    "bridge_rect_001": [
        {"q": "この回路の機能・名称は何か。",
         "check": [["ブリッジ", "全波", "bridge", "full-wave", "fullwave", "整流"]]},
        {"q": "ダイオードは何個使われているか。",
         "check": [["4", "四", "4個", "4本", "four"]]},
        {"q": "平滑コンデンサの値はいくらか。",
         "check": [["100u", "100µ", "100 u", "100uf", "100 µ"]]},
        {"q": "整流方式は半波か全波か。",
         "check": [["全波", "full-wave", "fullwave", "full wave"]]},
    ],
}

SYSTEM = (
    "あなたは回路の専門家です。与えられた«回路の表現»だけに基づいて質問に答えてください。"
    "表現に書かれていない情報は推測せず「表現から不明」と答えること。"
    "各質問には必ず 'A<番号>: <簡潔な答え>' の形式で、1行ずつ答えてください。"
)


def build_user(rep_text: str, questions: list[dict]) -> str:
    qs = "\n".join(f"Q{i+1}: {d['q']}" for i, d in enumerate(questions))
    return f"【回路の表現】\n{rep_text}\n\n【質問】\n{qs}"


def slice_answers(text: str, n: int) -> dict[int, str]:
    """'A<num>:' マーカーで回答を質問ごとに分割。失敗時は空 dict。"""
    marks = list(re.finditer(r"(?im)^\s*A\s*(\d+)\s*[:：\)]", text))
    if len(marks) < 2:
        return {}
    out: dict[int, str] = {}
    for i, m in enumerate(marks):
        idx = int(m.group(1))
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[idx] = text[start:end]
    return out


def grade(answer_slice: str, check: list[list[str]]) -> bool:
    s = answer_slice.lower()
    return all(any(alt.lower() in s for alt in clause) for clause in check)


def main() -> int:
    ap = argparse.ArgumentParser(description="回路表現の LLM 読解可能性 A/B/C 比較")
    ap.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "claude"),
                    help="claude / gemini / anthropic / ollama")
    ap.add_argument("--model", default="qwen2.5:7b", help="provider=ollama 時のモデル名")
    ap.add_argument("--circuits", nargs="*", default=list(QUESTIONS.keys()))
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--dry-run", action="store_true", help="LLM を呼ばず表現とプロンプトのみ表示")
    args = ap.parse_args()

    cs = {c["id"]: c for c in json.load(open(args.db, encoding="utf-8"))["circuits"]}

    if args.dry_run:
        for cid in args.circuits:
            print(f"\n{SEP}\n# {cid}")
            for name, fn in BUILDERS.items():
                print(f"\n--- {name} ---\n{fn(cs[cid])}")
            print("\n--- 質問 ---")
            for i, d in enumerate(QUESTIONS[cid]):
                print(f"Q{i+1}: {d['q']}  check={d['check']}")
        return 0

    if args.provider == "ollama":
        from ollama_client import OllamaClient
        llm = OllamaClient(model=args.model)
    elif args.provider in ("claude", "gemini"):
        from llm_client import CLILLMClient
        llm = CLILLMClient(provider=args.provider, timeout=240)
    else:
        from llm_client import LLMClient
        llm = LLMClient(provider=args.provider)
    print(f"LLM = {llm}")

    scratch = os.path.join(os.environ.get("TEMP", "."), "ir_repr_eval_outputs")
    os.makedirs(scratch, exist_ok=True)

    # スコア集計: rep -> [正解数, 総数]
    totals = {name: [0, 0] for name in BUILDERS}
    per_cell = {}  # (cid, rep) -> (correct, total, details)

    for cid in args.circuits:
        c = cs[cid]
        questions = QUESTIONS[cid]
        print(f"\n{SEP}\n# {cid}  ({c.get('name')})  質問{len(questions)}件")
        for name, fn in BUILDERS.items():
            rep_text = fn(c)
            user = build_user(rep_text, questions)
            try:
                out = llm.chat(system=SYSTEM, user=user)
            except Exception as e:
                print(f"  [{name}] LLM 呼び出し失敗: {e}")
                per_cell[(cid, name)] = (0, len(questions), "LLM_ERROR")
                totals[name][1] += len(questions)
                continue
            # 監査用に生出力保存
            with open(os.path.join(scratch, f"{cid}__{name}.txt"), "w", encoding="utf-8") as f:
                f.write("=== SYSTEM ===\n" + SYSTEM + "\n\n=== USER ===\n" + user
                        + "\n\n=== OUTPUT ===\n" + out)
            slices = slice_answers(out, len(questions))
            correct, marks = 0, []
            for i, d in enumerate(questions):
                ans = slices.get(i + 1, out if not slices else "")
                ok = grade(ans, d["check"])
                correct += ok
                marks.append("✓" if ok else "✗")
            per_cell[(cid, name)] = (correct, len(questions),
                                     "".join(marks) + ("" if slices else " (format-fail→全文採点)"))
            totals[name][0] += correct
            totals[name][1] += len(questions)
            print(f"  [{name:8}] {correct}/{len(questions)}  {''.join(marks)}")

    # サマリ表
    print(f"\n{SEP}\n[サマリ] 表現別 正答率")
    print(f"  {'回路':24} " + " ".join(f"{n:>10}" for n in BUILDERS))
    for cid in args.circuits:
        row = []
        for name in BUILDERS:
            cor, tot, _ = per_cell.get((cid, name), (0, 0, ""))
            row.append(f"{cor}/{tot}".rjust(10))
        print(f"  {cid:24} " + " ".join(row))
    print(f"  {'─'*24} " + " ".join("─"*10 for _ in BUILDERS))
    tot_row = []
    for name in BUILDERS:
        cor, tot = totals[name]
        pct = (cor / tot * 100) if tot else 0
        tot_row.append(f"{cor}/{tot}({pct:.0f}%)".rjust(10))
    print(f"  {'合計':24} " + " ".join(tot_row))
    print(f"\n生 LLM 出力: {scratch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
