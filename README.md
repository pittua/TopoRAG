# TopoRAG — 回路ネットリスト 類似検索RAGシステム

ネットリスト（JSON形式の回路接続情報）を入力として、
特徴量ベクトルによるコサイン類似度検索と LLM を組み合わせ、
回路トポロジーを判定するシステムです。

複合・多段回路は **ブロック単位 RAG** によって対応します。
LLM への依存を極力排除し、DB 照合結果の読み取りのみを要求します。

---

## 対応回路（features_db.json 収録済み）

### フィルタ回路

| ID | 回路名 | 構成部品 |
|---|---|---|
| rc_lowpass_001 | RCローパスフィルタ | R, C |
| rc_highpass_001 | RCハイパスフィルタ | C, R |
| lc_lowpass_001 | LCローパスフィルタ | L, C |
| lc_highpass_001 | LCハイパスフィルタ | C, L |
| rlc_bandpass_001 | RLCバンドパスフィルタ | L, C, R |
| rlc_notch_001 | RLCノッチフィルタ | R, L, C |
| lc_pi_filter_001 | π型LCフィルタ（ローパス） | C, L, C |
| lc_t_filter_001 | T型LCフィルタ（ローパス） | L, C, L |

### 整流・波形整形

| ID | 回路名 | 構成部品 |
|---|---|---|
| halfwave_rect_001 | 半波整流回路 | D, R |
| positive_clipper_001 | 正クリッパ回路 | R, D |
| negative_clipper_001 | 負クリッパ回路 | R, D |

### 電源・定電圧

| ID | 回路名 | 構成部品 |
|---|---|---|
| zener_regulator_001 | ツェナー定電圧回路 | R, DZ |
| buck_001 | 降圧チョッパ（Buck Converter） | SW, D, L, C |
| boost_001 | 昇圧チョッパ（Boost Converter） | L, SW, D, C |

### 複合・多段回路（階層ブロック対応）

| ID | 回路名 | 構成部品 | ブロック分解 |
|---|---|---|---|
| buck_lc_filter_001 | 降圧チョッパ + LC出力フィルタ | SW, D, L, C, L, C | {SW,D} + {L,C} + {L,C} |
| rc_cascade_lpf_001 | RC 2段カスケードローパスフィルタ | R, C, R, C | {R,C} + {R,C} |

### プリミティブブロック（ブロック単位照合の参照用）

複合回路の各ブロックが DB の全体回路に対応しない場合のフォールバック。
全体回路が DB に存在する場合はそちらが優先される。

| ID | 回路名 | 構成部品 | 対応するブロック |
|---|---|---|---|
| sw_freewheel_001 | スイッチング段（SW＋フリーホイールD） | SW, D | Buck Block0 相当 |
| boost_input_stage_001 | 昇圧入力段（L＋SW） | L, SW | Boost Block0 相当 |
| rectifier_cap_stage_001 | 整流・平滑段（D＋C） | D, C | Boost Block1 相当 |

> 既存の `lc_lowpass_001`（L+C）・`rc_lowpass_001`（R+C）・`rc_highpass_001`（C+R）も
> プリミティブブロック参照として機能します。

---

## ファイル構成

```
TopoRAG/
├── feature_extractor.py   特徴量抽出（グラフ解析）→ features_db.json を生成
├── block_decomposer.py    ブロック分解（T分岐点分割）
├── llm_client.py          LLM呼び出しラッパー（SDK / CLI / Mock）
├── circuit_rag.py         ベクトル化・類似検索・LLM判定のメインスクリプト
├── sample_netlists.json   DB登録済み回路のネットリスト定義（19回路）
├── features_db.json       抽出済み特徴量DB（RAGの検索対象）
├── query_netlists.json    判定したい回路のネットリスト（入力ファイル）
└── vector_db.json         特徴量の数値ベクトル（20次元）
```

---

## システム全体フロー

```
入力ネットリスト（dict）
  │
  ▼
block_decomposer.py（ブロック分解）
  └─ 主直列パス上の T 分岐点（GND 接続節点）を検出
  └─ 単一ブロック → そのまま通過
  └─ 複数ブロック → サブ回路 dict のリストに分割
  │
  ▼
feature_extractor.py（extract_hierarchical_features）
  └─ 回路レベル: A/B1/B2/B3/C 全特徴量 + ports（フラット）
  └─ ブロックレベル: 各ブロックに同じ特徴量を個別抽出
  │
  ▼
circuit_rag.py
  └─ 20次元ベクトル化（回路レベル + 各ブロック）
  └─ 階層対応類似度検索
       単一 vs 単一    → コサイン類似度
       複合 vs 単一    → 各ブロック vs DB のコサイン最大値
       単一 vs 複合    → クエリ vs 各 DB ブロックのコサイン最大値
       複合 vs 複合    → グリーディブロックマッチング
  └─ タグ類似度とのハイブリッドスコア（alpha 重み付け）
  │
  ├─ 単一ブロック回路 ──→ 全体照合プロンプト → LLM に「回路種別を判定せよ」
  │
  └─ 複合回路 ──────────→ ブロック単位RAGプロンプト → LLM に「照合結果を読んでブロック名を組み合わせよ」
  │
  ▼
llm_client.py（LLMClient / CLILLMClient / MockLLMClient）
  │
  ▼
【判定】【ブロック構成 / 根拠】【類似度の解釈】
```

---

## 特徴量ベクトル（20次元）

| 次元 | グループ | 内容 |
|---|---|---|
| 0–4 | A. 部品 | R/C/L/SW/D の有無（各0/1） |
| 5 | A. 部品 | 総部品数（正規化） |
| 6 | B1. 順序 | SW-L 順序（SW前=1 / L前=-1 / なし=0） |
| 7 | B1. 順序 | 先頭直列部品が R（0/1） |
| 8 | B1. 順序 | 先頭直列部品が C（0/1） |
| 9 | B1. 順序 | 先頭直列部品が L（0/1） |
| 10–13 | B2. ダイオード | アノード→GND / カソード→OUT / アノード→OUT / カソード→GND |
| 14 | B3. 直列/並列 | 並列部品の有無（0/1） |
| 15 | B3. 直列/並列 | 直列チェーン長（正規化） |
| 16 | C. ノード | ノード数（正規化） |
| 17 | C. ノード | 高次ノード（次数≥3）の有無（0/1） |
| 18 | C. ノード | ループ数（正規化） |
| 19 | A. 部品 | ツェナーダイオード（DZ）の有無（0/1） |

> **DZ（ツェナー）について**：部品種別 `"DZ"` を使用すると `has_zener=True` として扱われ、
> 構造が同一の負クリッパ回路と次元19で区別されます。

---

## 階層特徴量（ブロック分解）

複数の機能ブロックを持つ回路には `block_decomposer.py` によるブロック分解が適用されます。

### ブロック分割アルゴリズム（T分岐点分割）

```
主直列パス（入力→出力の最短パス、GND経由を除く）
  │
  ▼ 内部節点のうち GND に接続されているものをブロック境界とする
  ├─ Block 0: VIN→N1（境界）  ← SW と フリーホイールダイオード
  ├─ Block 1: N1→N2（境界）  ← L と出力 C（チョッパ本体）
  └─ Block 2: N2→OUT        ← 追加 LC フィルタ段
```

区間終端の GND シャント部品はその区間（前段ブロック）に帰属します。

### features_db.json の階層フォーマット

```json
{
  "circuit_id": "buck_lc_filter_001",
  "circuit_name": "降圧チョッパ + LC出力フィルタ",
  "ports": { "input": "VIN", "output": "N3", "gnd": "GND" },
  "A_component": { ... },
  "B1_order": { ... },
  "is_hierarchical": true,
  "n_blocks": 3,
  "blocks": [
    {
      "circuit_id": "buck_lc_filter_001_b0",
      "ports": { "input": "VIN", "output": "N1", "gnd": "GND" },
      "A_component": { "component_types": ["D", "SW"], ... },
      "B1_order": { "series_type_sequence": ["SW"], ... },
      ...
    },
    { ... },
    { ... }
  ]
}
```

単一ブロック回路は `"is_hierarchical": false, "blocks": []` になります。

### 階層対応類似度検索

| クエリ / DB | 処理方法 |
|---|---|
| 単一 / 単一 | コサイン類似度（従来通り） |
| 複合 / 単一 | 各クエリブロック vs DB回路のコサイン **最大値** |
| 単一 / 複合 | クエリ vs 各 DB ブロックのコサイン **最大値** |
| 複合 / 複合 | グリーディブロックマッチング（余剰ブロックにペナルティ） |

### 複合回路向けプロンプト戦略（ブロック単位 RAG）

複合回路と判定された場合、`build_prompt()` は自動的にブロック単位 RAG に切り替わります。

```
通常（単一ブロック）:
  回路全体 → DB検索（上位K件） → LLMに「回路種別を判定せよ」

複合回路:
  ブロック1 → DB検索（上位2件） → LLMに提示
  ブロック2 → DB検索（上位2件） → LLMに提示
  ブロック3 → DB検索（上位2件） → LLMに提示
  → LLMへの指示: 「照合結果を読んでブロック名を組み合わせよ。推論は不要」
```

LLM が行うのは「ブロック名の結合」のみであり、回路トポロジーの推論は発生しません。

**プロンプト出力例（昇圧チョッパ + LC出力フィルタ）:**

```
ブロック1 (VIN → N1): {L, SW}
  1位 score=0.887  昇圧チョッパ（Boost Converter）
  2位 score=0.800  昇圧入力段（L＋SW）

ブロック2 (N1 → N2): {D, C}
  1位 score=0.887  昇圧チョッパ（Boost Converter）
  2位 score=0.727  整流・平滑段（D＋C）

ブロック3 (N2 → N3): {L, C}
  1位 score=0.910  降圧チョッパ + LC出力フィルタ
  2位 score=0.833  降圧チョッパ（Buck Converter）
```

---

## ハイブリッド検索（トポロジー × 機能タグ）

類似度スコアはトポロジースコアとタグスコアの加重和で計算されます。

```
score = alpha × topo_score + (1 - alpha) × tag_score
```

| 要素 | 計算方法 | 説明 |
|---|---|---|
| `topo_score` | コサイン類似度（20次元ベクトル）または階層マッチング | 回路の構造・接続トポロジーの一致度 |
| `tag_score` | Jaccard 係数（function_tags の集合比較） | 回路の機能・役割の一致度 |
| `alpha` | CLI引数 `--alpha`（デフォルト 0.7） | 0.0〜1.0 でトポロジー重視 / タグ重視を調整 |

> タグが付いた DB 全体回路とプリミティブブロックの優先順位は自動的に決まります。
> DB に対応する全体回路がある場合はタグ一致でそちらが上位になり、
> 未知の組み合わせではプリミティブブロックが浮上します。

### function_tags の書き方

`query_netlists.json` のクエリ回路に `function_tags` を付けると、タグ検索が有効になります。

```json
{
  "id": "my_circuit",
  "name": "テスト回路",
  "description": "回路の簡単な説明（LLMプロンプトに含まれる）",
  "function_tags": ["filter", "lowpass", "rc"],
  "components": [...],
  "ports": {...}
}
```

タグを省略した場合は `alpha=1.0`（トポロジーのみ）と同じ結果になります。

### 推奨タグ一覧

| カテゴリ | タグ例 |
|---|---|
| 機能種別 | `filter` `rectifier` `clipper` `regulator` `converter` |
| 周波数特性 | `lowpass` `highpass` `bandpass` `notch` |
| 受動/能動 | `passive` `active` |
| 電源種別 | `dc_dc` `ac_dc` `switching` `power` |
| 構成部品 | `rc` `lc` `rlc` |
| その他 | `frequency_selective` `step_up` `step_down` `dc_blocking` `noise_reduction` `energy_storage` `smoothing` |

---

## llm_client.py — LLMクライアント一覧

| クラス | 動作方法 | APIキー |
|---|---|---|
| `LLMClient(provider="anthropic")` | Anthropic SDK | 必要（`ANTHROPIC_API_KEY`） |
| `LLMClient(provider="gemini")` | Google GenAI SDK | 必要（`GEMINI_API_KEY`） |
| `CLILLMClient(provider="claude")` | Claude Code CLI をsubprocess呼び出し | **不要**（CLI ログイン済みで動作） |
| `CLILLMClient(provider="gemini")` | Gemini CLI をsubprocess呼び出し | **不要**（CLI ログイン済みで動作） |
| `MockLLMClient()` | ローカルモック（API呼び出しなし） | 不要 |

共通インタフェース：
```python
result = client.chat(system="システムプロンプト", user="ユーザーメッセージ")
```

---

## 使い方

### 前提条件

```bash
pip install numpy networkx anthropic      # Anthropic SDKを使う場合
pip install numpy networkx google-genai   # Gemini SDKを使う場合
pip install numpy networkx                # CLI / Mock のみなら最小構成
```

### 基本的な実行フロー

```
1. query_netlists.json に判定したい回路を記述する
2. circuit_rag.py を実行する
```

### 1. CLI経由でLLM判定を実行（APIキー不要）

**Claude CLI を使う場合**

```bash
# PowerShell
$env:LLM_PROVIDER = "claude"
python circuit_rag.py

# bash / Git Bash
LLM_PROVIDER=claude python circuit_rag.py
```

**Gemini CLI を使う場合**

```bash
# PowerShell
$env:LLM_PROVIDER = "gemini"
python circuit_rag.py

# bash / Git Bash
LLM_PROVIDER=gemini python circuit_rag.py
```

### 2. SDK経由でLLM判定を実行（APIキー必要）

```bash
# Anthropic SDK
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:LLM_PROVIDER = "anthropic"
python circuit_rag.py

# Gemini SDK
$env:GEMINI_API_KEY = "AIza..."
$env:LLM_PROVIDER = "gemini-sdk"
python circuit_rag.py
```

### 3. モックで動作確認（ネットワーク不要）

```bash
python circuit_rag.py
# LLM_PROVIDER 未設定 → 自動的に MockLLMClient が使われる
```

### コマンドラインオプション

```bash
python circuit_rag.py [--query <ファイル>] [--db <ファイル>] [--top-k <件数>] [--alpha <重み>]

オプション:
  -q, --query   判定対象ネットリストのJSONファイル（デフォルト: query_netlists.json）
  --db          特徴量DBファイル（デフォルト: features_db.json）
  -k, --top-k   類似検索で参照する上位件数（デフォルト: 3）
  -a, --alpha   トポロジースコアの重み（0.0〜1.0、デフォルト: 0.7）
                  1.0 = トポロジーのみ / 0.0 = タグのみ
```

別ファイルを指定する例：

```bash
LLM_PROVIDER=claude python circuit_rag.py --query my_circuit.json --top-k 5

# タグ検索を重視する場合（alpha を下げる）
LLM_PROVIDER=claude python circuit_rag.py --alpha 0.4
```

### 4. Pythonコードから直接呼び出す

```python
from circuit_rag import CircuitRAG
from llm_client import CLILLMClient
import json

rag = CircuitRAG(llm=CLILLMClient(provider="claude"))
rag.load_from_file("features_db.json")

with open("query_netlists.json", encoding="utf-8") as f:
    queries = json.load(f)["circuits"]

for circuit in queries:
    print(rag.judge(circuit, top_k=3))
```

---

## DBへの回路追加手順

### 1. sample_netlists.json にネットリストを追記

### JSONサンプルフォーマット

```json
{
  "id":          "一意なID（文字列）",
  "name":        "回路名（表示用）",
  "description": "回路の説明（LLMプロンプトに含まれる）",
  "function_tags": ["filter", "lowpass", "rc"],
  "components": [
    {
      "id":   "部品ID（R1, C1, L1, SW1, D1 など）",
      "type": "部品種別（R / C / L / SW / D / DZ）",
      "terminals": {
        "p": "接続ノード名",
        "n": "接続ノード名"
      }
    }
  ],
  "ports": {
    "input":  "入力ポートのノード名",
    "output": "出力ポートのノード名",
    "gnd":    "GNDノード名"
  }
}
```

- ダイオードの場合は `terminals` のキーを `"anode"` / `"cathode"` とすることで向きの特徴量（B2）が正しく抽出されます。
- `description` と `function_tags` はオプションですが、ハイブリッド検索の精度向上のために付与を推奨します。
- 複合回路のブロック参照用エントリ（プリミティブブロック）も同じ形式で追加できます。

### 2. 特徴量DBを再生成

```bash
python feature_extractor.py
# → features_db.json が更新される（単一ブロック回路は is_hierarchical=false）
```

### 3. circuit_rag.py を実行して確認

```bash
LLM_PROVIDER=claude python circuit_rag.py
```

---

## CLI インストール（未導入の場合）

```bash
# Claude Code CLI
npm install -g @anthropic-ai/claude-code
claude   # ブラウザでログイン認証

# Gemini CLI
npm install -g @google/gemini-cli
gemini   # ブラウザでログイン認証
```
