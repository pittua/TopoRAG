"""score_baseline_direct.py — 実験A: P1（閉集合選択）の機械採点

保存済み JSONL（run_baseline_direct.py の出力）のみを入力とし、設計書 §5–6 の定義で採点する。
    python score_baseline_direct.py results/baseline_p1_claude.jsonl

採点規則（EXPERIMENT_A_DESIGN.md より）:
- 各試行: JSON抽出 → answer_id が {DB47 id} ∪ {unknown} 外なら「無効票」
- クエリ集約: 多数決（最頻値）。全票が異なる/有効票0 → 「不一致」＝不正解扱い・別掲
- in28: strict = expect先頭1件と一致 / lenient = expectのいずれかと一致
- out20: 棄却成功 = unknown / 誤受理 = いずれかのDB id
- 全比率に Wilson 95%CI を併記
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fmt(k: int, n: int, label: str) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{label}: {k}/{n} = {k / n * 100:.1f}%  [95%CI {lo * 100:.1f}–{hi * 100:.1f}]"


def extract_answer_id(raw: str) -> str | None:
    """応答からJSONを抽出し answer_id を返す。抽出不能なら None（無効票）。"""
    if not raw:
        return None
    text = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ans = obj.get("answer_id")
    return ans.strip() if isinstance(ans, str) else None


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    jsonl_path = Path(sys.argv[1])

    # 正解定義と有効ラベル集合
    expected = yaml.safe_load((ROOT / "real_expected.yaml").read_text(encoding="utf-8"))
    db = json.loads((ROOT / "features_db.json").read_text(encoding="utf-8"))
    items = db["circuits"] if isinstance(db, dict) and "circuits" in db else db
    valid_ids = {it["circuit_id"] for it in items}

    # 試行を読み込みクエリ単位に集約
    trials: dict[str, list[str | None]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        ans = extract_answer_id(rec.get("raw", ""))
        if ans is not None and ans != "unknown" and ans not in valid_ids:
            ans = None  # 候補外 → 無効票
        trials.setdefault(rec["query_id"], []).append(ans)

    n_in = n_out = 0
    strict_hit = lenient_hit = 0
    reject_ok = false_accept = 0
    all_same = inconsistent = 0
    invalid_votes = total_votes = 0
    detail_rows: list[str] = []

    for qid, votes in sorted(trials.items()):
        exp = expected.get(qid)
        if exp is None:
            print(f"警告: real_expected.yaml に {qid} が無いためスキップ")
            continue
        total_votes += len(votes)
        invalid_votes += sum(v is None for v in votes)
        valid_votes = [v for v in votes if v is not None]

        # 多数決（最頻値）。同数トップや有効票0は「不一致」
        answer = None
        if valid_votes:
            cnt = Counter(valid_votes).most_common()
            if len(cnt) == 1 or cnt[0][1] > cnt[1][1]:
                answer = cnt[0][0]
        if answer is None:
            inconsistent += 1
        if len(set(votes)) == 1 and votes[0] is not None:
            all_same += 1

        if exp["scope"] == "in":
            n_in += 1
            exp_ids = exp.get("expect") or []
            s = answer == (exp_ids[0] if exp_ids else None)
            l = answer in exp_ids
            strict_hit += s
            lenient_hit += l
            mark = "◎" if s else ("○" if l else "×")
        else:
            n_out += 1
            if answer == "unknown":
                reject_ok += 1
                mark = "◎"
            elif answer in valid_ids:
                false_accept += 1
                mark = "×誤受理"
            else:
                mark = "×不一致"
        detail_rows.append(f"  {mark} {qid:35s} votes={votes} -> {answer}")

    print(f"=== {jsonl_path.name} ===")
    print("\n".join(detail_rows))
    print("\n── in-scope（Hit@1）──")
    print(" " + fmt(strict_hit, n_in, "strict "))
    print(" " + fmt(lenient_hit, n_in, "lenient"))
    print("── out-of-scope ──")
    print(" " + fmt(reject_ok, n_out, "棄却成功（unknown回答）"))
    print(" " + fmt(false_accept, n_out, "誤受理（既知idを回答）"))
    print("── 再現性 ──")
    n_all = n_in + n_out
    print(" " + fmt(all_same, n_all, "3試行完全一致"))
    print(" " + fmt(inconsistent, n_all, "多数決不成立（不正解扱い）"))
    print(f" 無効票（JSON不能/候補外）: {invalid_votes}/{total_votes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
