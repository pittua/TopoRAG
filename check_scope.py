"""
check_scope.py — スコープ定義の整合性検証（HANDOFF §7.1）

目的:
  real_expected.yaml の scope ラベルが、DB の中身ではなく
  scope_taxonomy.yaml（DB 非依存の機能分類）と整合しているかを検証する。

  従来 scope は「DB にカテゴリが存在 = in」という DB 相対の定義で、DB を足すたびに
  ラベルが動く循環的なものだった。本スクリプトは scope が family（機能分類）から
  正しく導かれているかを機械的にチェックし、ブラインドラベリング（HANDOFF §4）の
  前提＝安定した scope 定義を担保する。

検査項目:
  1. in-scope の各エントリは family を持ち、その family が taxonomy に存在する。
  2. out-of-scope の各エントリは family: none（または family 未指定）である。
  3. scope と family が矛盾しない（in なのに none、out なのに実 family を禁止）。
  4. DB 相対の根拠混入を検出（note に「DB に追加して in-scope 化」等の循環論法痕跡）。
  5. real_corpus.json の全回路が real_expected.yaml にラベル付けされている。

使い方:
  python check_scope.py
  python check_scope.py --expected real_expected.yaml --taxonomy scope_taxonomy.yaml

終了コード: 0 = 整合 / 1 = 違反あり（CI で利用可）
"""
from __future__ import annotations

import argparse
import re
import sys

import yaml

# Windows の既定コンソール(cp932)で ✓/✗/─ 等が出力できず落ちるのを防ぐ。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# scope を DB 相対に判定している循環論法の痕跡（note 内）を検出する語。
# 「family が DB にあるから in」という根拠は禁止。in 根拠は family の defining_criteria のみ。
_CIRCULAR_PATTERNS = [
    r"DB\s*に\s*追加",
    r"in-?scope\s*化",
    r"DB\s*未収録\s*だから",
    r"DB\s*にある\s*から",
]


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_corpus_ids(path: str) -> list[str]:
    import json
    try:
        with open(path, encoding="utf-8") as f:
            return [c["id"] for c in json.load(f)["circuits"]]
    except FileNotFoundError:
        return []


def check(expected: dict, taxonomy: dict, corpus_ids: list[str]) -> list[str]:
    """違反メッセージのリストを返す（空なら整合）。"""
    errors: list[str] = []
    families = set((taxonomy.get("families") or {}).keys())

    for qid, spec in expected.items():
        if not isinstance(spec, dict):
            errors.append(f"[{qid}] エントリが dict でない")
            continue
        scope = spec.get("scope")
        family = spec.get("family")
        note = spec.get("note", "") or ""

        if scope not in ("in", "out"):
            errors.append(f"[{qid}] scope は in/out のいずれかが必須（現在: {scope!r}）")
            continue

        # 1 & 3: in は taxonomy の family を持つ
        if scope == "in":
            if family in (None, "none"):
                errors.append(
                    f"[{qid}] scope=in には scope_taxonomy.yaml の family が必須"
                    f"（現在: {family!r}）。DB の有無ではなく family で in を根拠づけること")
            elif family not in families:
                errors.append(
                    f"[{qid}] family={family!r} は scope_taxonomy.yaml に存在しない"
                    f"（定義済み: {sorted(families)}）")
        # 2 & 3: out は family: none
        elif scope == "out":
            if family not in (None, "none"):
                errors.append(
                    f"[{qid}] scope=out なのに family={family!r}。"
                    f"どの family にも該当しないから out のはず → family: none にする")

        # 4: 循環論法（DB 相対の根拠）の痕跡を検出
        for pat in _CIRCULAR_PATTERNS:
            if re.search(pat, note):
                # 否定文脈（「〜ではない」）は除外したいが、機械判定は難しいので
                # 「ではない/無関係/事実ではない」を含む行は許容する簡易ヒューリスティック。
                if not re.search(r"ではない|無関係|事実ではない|から in", note):
                    errors.append(
                        f"[{qid}] note に DB 相対の根拠の疑い（'{pat}'）。"
                        f"scope の根拠は family の defining_criteria のみ（HANDOFF §7.1）")
                break

    # 5: コーパス全件にラベルがあるか
    for cid in corpus_ids:
        if cid not in expected:
            errors.append(f"[{cid}] real_corpus.json にあるが real_expected.yaml に未ラベル")

    return errors


def summarize(expected: dict, taxonomy: dict) -> None:
    families = taxonomy.get("families") or {}
    by_family: dict[str, list[str]] = {f: [] for f in families}
    n_in = n_out = 0
    for qid, spec in expected.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("scope") == "in":
            n_in += 1
            fam = spec.get("family")
            by_family.setdefault(fam, []).append(qid)
        elif spec.get("scope") == "out":
            n_out += 1

    print(f"\nin-scope={n_in}  out-of-scope={n_out}  family 定義数={len(families)}")
    print("─" * 60)
    print("[family 別 in-scope クエリ数]（DB 非依存の機能分類）")
    for fam, info in families.items():
        members = by_family.get(fam, [])
        title = info.get("title", "") if isinstance(info, dict) else ""
        print(f"  {fam:18} {title:14} : {len(members)} 件  {members}")
    empty = [f for f in families if not by_family.get(f)]
    if empty:
        print(f"  ※ コーパスに実機例が無い family: {empty}（拡充候補）")


def main() -> int:
    ap = argparse.ArgumentParser(description="スコープ定義の整合性検証（§7.1）")
    ap.add_argument("--expected", default="real_expected.yaml")
    ap.add_argument("--taxonomy", default="scope_taxonomy.yaml")
    ap.add_argument("--corpus", default="real_corpus.json")
    args = ap.parse_args()

    expected = load_yaml(args.expected)
    taxonomy = load_yaml(args.taxonomy)
    corpus_ids = load_corpus_ids(args.corpus)

    print("=" * 60)
    print("スコープ定義整合性チェック（real_expected.yaml × scope_taxonomy.yaml）")
    print("=" * 60)

    errors = check(expected, taxonomy, corpus_ids)
    summarize(expected, taxonomy)

    print("\n" + "=" * 60)
    if errors:
        print(f"✗ {len(errors)} 件の違反:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("✓ 整合: 全 scope ラベルが DB 非依存の family 定義と一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
