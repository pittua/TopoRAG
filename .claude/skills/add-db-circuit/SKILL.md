---
name: add-db-circuit
description: TopoRAG の DB (sample_netlists.json) に新しい回路を追加する手順。ネットリスト形式・端子命名規則・features_db.json 再生成・回帰確認まで。「DBに回路を追加」「新しい回路種別を登録」と言われたら使う。
---

# DB への回路追加手順

## 1. sample_netlists.json の "circuits" に追記

```json
{
  "id": "snake_case_001",
  "name": "表示用回路名",
  "description": "1行説明（LLMプロンプトに入る）",
  "function_tags": ["filter", "lowpass"],
  "components": [
    {"id": "R1", "type": "R", "terminals": {"p": "IN", "n": "OUT"}},
    {"id": "C1", "type": "C", "terminals": {"p": "OUT", "n": "GND"}}
  ],
  "ports": {"input": "IN", "output": "OUT", "gnd": "GND"}
}
```

### 端子命名規則（特徴抽出が端子名を解釈するため必須）

| 種別 | terminals キー |
|---|---|
| R/C/L/SW（2端子受動） | 任意（p/n 慣例） |
| D/DZ | `anode` / `cathode`（向き特徴 B2 に必要） |
| NPN/PNP | `base` / `collector` / `emitter` |
| NMOS/PMOS | `gate` / `drain` / `source` |
| OPAMP | `in_p` / `in_n` / `out` |

- 差動回路は ports に `input2` を追加可
- カレントミラー＝ダイオード接続、差動対＝共通端子共有は構造から自動検出される

## 2. 特徴量DB再生成（必須・忘れやすい）

```powershell
python feature_extractor.py   # features_db.json を再生成
```

features_db.json は生成物。手編集せず、sample_netlists.json と**必ず同時にコミット**する。

## 3. 回帰確認

```powershell
python -m pytest -q && python evaluate.py
```

- 自己検索 Hit@1 = 100% 維持、摂動 PASS、exit 0 を確認
- 取り違えが出た場合: まず衝突相手と特徴量を比較する。**新次元の追加は最終手段**
  （次元を増やすと全スコアが動き、棄却閾値 CALIBRATED_REJECT_THRESHOLD の再校正が必要になる）

## 4. 注意

- 実機評価コーパス(real_corpus.json)を見てから DB 回路を選ばない（自己検索化＝評価汚染。HANDOFF §4 の独立性原則）
- README の回路一覧表も更新する（陳腐化の前科あり）
