---
name: add-real-corpus
description: 実機評価コーパス (real_corpus.json / real_expected.yaml) へ第三者回路を追加する手順。KiCad 変換・ブラインドラベリング原則・check_scope 検証・再評価まで。「実機コーパス拡充」「out-of-scope を増やす」と言われたら使う。
---

# 実機コーパス拡充手順

## 原則（先に読む。違反すると評価が偽装される）

1. **ブラインドラベリング**: scope (in/out) は**スコアを見る前に**決める。「低く出たから out」は循環論法
2. **DB 非依存の scope 判定**: in = scope_taxonomy.yaml のいずれかの family の defining_criteria を満たすこと。「DB に似た回路があるから in」は禁止
3. **独立性**: DB (sample_netlists.json) を見てから評価用回路を選ばない
4. **タグ禁止**: コーパス回路に function_tags を付けない（汎化テストのため全クエリタグ無し）
5. **in/out バランス**: AUC は両クラス数に効く。片方だけ増やさない

## 手順

```powershell
# 1. 変換（.kicad_sch は kicad_real_test/ 等に置く。git 追跡外）
python kicad_sch_to_toporag.py path\to\Circuit.kicad_sch tmp.json

# 2. ここで scope をラベリング（スコア確認前！）
#    in なら family も決める（scope_taxonomy.yaml の 7 family から）

# 3. tmp.json の circuits を real_corpus.json に追記（function_tags は付けない）

# 4. real_expected.yaml にラベル追加
#    qid:
#      scope: in | out
#      family: filter_passive 等（in のみ）
#      expect: [DB回路id, ...]（in のみ。先頭が strict 判定対象）
#      note: 出典と判断根拠

# 5. 整合検査と再評価
python check_scope.py
python validate_real.py
python reject_eval.py
```

## 結果の読み方

- 撤退ゲート (reject_eval.py 末尾): n_out≥20 / AUC 95%CI 下限>0.5 / nested bacc≥0.65 の事前登録基準
- 棄却閾値を再校正したら circuit_rag.py の `CALIBRATED_REJECT_THRESHOLD` とコメント(校正日・n・TNR/TPR)を更新
- easy/hard の分離報告: DB 厳密一致がある in は Hit@1 を水増しするので、near-miss in と分けて解釈

## 新カテゴリを in にしたい場合

先に scope_taxonomy.yaml へ family を追加 → DB へ回路追加（/add-db-circuit）→ コーパス追加。
**DB を足してから後付けで in にしない**（sallen_key の前科）。
