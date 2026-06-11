# TopoRAG — 回路ネットリスト トポロジー分類器（+LLM 整形）

ネットリスト（JSON形式の回路接続情報）を入力として、
**手設計の 43 次元グラフ特徴ベクトルによる最近傍分類**で回路トポロジーを判定し、
LLM に検索結果の整形（上位候補の読み上げ・名称の組み立て）をさせるシステムです。

複合・多段回路はブロック単位の照合で対応します。
LLM への依存を極力排除し、DB 照合結果の読み取りのみを要求します。

> **正直な位置づけ**：これは学習済み埋め込みやスケール戦略を持つ「RAG」ではなく、
> **47 個の手書きプロトタイプに対する手設計特徴の最近傍分類器 + LLM 整形**です。
> 名前に "RAG" を含むのは経緯的なものです。LLM 段の役割は「検索上位を読んで名前を組み合わせる」
> だけで、その評価指標は「正解名の文字列一致率」という脆いものです。汎化性能は実機 in=28 の
> 検証（CI は依然広い）でのみ測れ、自己検索 Hit@1=100% は汎化ではありません（下記「汎化性能」参照）。

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

### 能動素子回路（BJT / MOSFET / OpAmp）

| ID | 回路名 | 構成部品 | 構成 |
|---|---|---|---|
| ce_amp_npn_001 | エミッタ接地増幅回路（NPN） | NPN, R×4, C | CE / 反転 |
| ce_amp_pnp_001 | エミッタ接地増幅回路（PNP） | PNP, R×4, C | CE / 反転 / p型 |
| emitter_follower_npn_001 | エミッタフォロワ（NPN） | NPN, R×3, C | CC / フォロワ |
| cs_amp_nmos_001 | ソース接地増幅回路（NMOS） | NMOS, R×4, C | CE / 反転 |
| source_follower_nmos_001 | ソースフォロワ（NMOS） | NMOS, R×2, C | CC / フォロワ |
| opamp_inverting_001 | 反転増幅回路（OpAmp） | OPAMP, R×2 | 反転 / 帰還 |
| opamp_noninverting_001 | 非反転増幅回路（OpAmp） | OPAMP, R×2 | 非反転 / 帰還 |
| opamp_buffer_001 | ボルテージフォロワ（OpAmp） | OPAMP | バッファ / 帰還 |
| two_stage_ce_001 | 2段エミッタ接地増幅（NPN） | NPN×2, R×8, C×2 | 多段 / CE |
| ce_follower_001 | エミッタ接地＋エミッタフォロワ（NPN） | NPN×2, R×7, C×2 | 多段 / CE→CC |
| current_mirror_npn_001 | カレントミラー（NPN） | NPN×2, R×2 | ダイオード接続 |
| diff_pair_npn_001 | 差動対（NPN ロングテールペア） | NPN×2, R×3 | 差動 / 結合ペア |

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
├── feature_extractor.py     特徴量抽出（グラフ解析）→ features_db.json を生成
├── block_decomposer.py      ブロック分解（T分岐点分割）
├── llm_client.py            LLM呼び出しラッパー（SDK / CLI / Mock）
├── circuit_simulator.py     ngspice/PySpice によるAC解析（フィルタ特性の定量化・任意）
├── circuit_rag.py           ベクトル化・類似検索・LLM判定のメインスクリプト
├── kicad_sch_to_toporag.py  KiCad .kicad_sch → TopoRAG 形式コンバータ
├── evaluate.py              評価スクリプト（自己検索/alpha/摂動/閾値/LLM/シミュレーション）
├── eval_expected.yaml       シミュレーション期待値（evaluate.py --sim が参照）
├── sample_netlists.json     DB登録済み回路のネットリスト定義（47回路）
├── features_db.json         抽出済み特徴量DB（RAGの検索対象）
└── query_netlists.json      判定したい回路のネットリスト（入力ファイル）
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
  └─ 43次元ベクトル化（回路レベル + 各ブロック）
  └─ 階層対応類似度検索
       単一 vs 単一    → コサイン類似度
       複合 vs 単一    → 各ブロック vs DB のコサイン最大値
       単一 vs 複合    → クエリ vs 各 DB ブロックのコサイン最大値
       複合 vs 複合    → 最適割当（Hungarian）ブロックマッチング
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

## 特徴量ベクトル（43次元）

次元 0–18 は受動トポロジー（R/C/L/SW/D/DZ）。能動素子追加後も**不変**で、受動回路では
次元 19–33 が全て 0 になる（末尾ゼロ拡張のためコサイン類似度が変わらない）。
次元 34–37 はダイオード回路の役割識別で、ダイオードを含まない回路では全て 0 になる（同じく不変）。
次元 38–42 は正規化部品数（R/C/L/D/SW の個数）で、presence(0–4) では区別できない素子数
（半波↔全波整流・buck↔ブリッジ整流など）を補う。

> ⚠ **次元を追加すると全コサイン類似度が動く**ため、棄却閾値などスコア依存の校正値は
> 特徴セットを変えるたびに無効化される（再校正が必要）。下表の次元数は実装
> （`circuit_rag.py` の `vectorize`）と一致していなければならない。

| 次元 | グループ | 内容 |
|---|---|---|
| 0–4 | A. 部品 | R/C/L/SW/D の有無（各0/1） |
| 5 | B1. 順序 | SW-L 順序（SW前=1 / L前=-1 / なし=0） |
| 6 | B1. 順序 | 先頭直列部品が R（0/1） |
| 7 | B1. 順序 | 先頭直列部品が C（0/1） |
| 8 | B1. 順序 | 先頭直列部品が L（0/1） |
| 9–12 | B2. ダイオード | アノード→GND / カソード→OUT / アノード→OUT / カソード→GND |
| 13 | B3. 直列/並列 | 並列部品の有無（0/1） |
| 14 | B3. 直列/並列 | 直列チェーン長（正規化） |
| 15 | C. ノード | ノード数（正規化） |
| 16 | C. ノード | 高次ノード（次数≥3）の有無（0/1） |
| 17 | C. ノード | ループ数（正規化） |
| 18 | A. 部品 | ツェナーダイオード（DZ）の有無（0/1） |
| 19 | D. 能動 | BJT（NPN/PNP）の有無（0/1） |
| 20 | D. 能動 | MOSFET（NMOS/PMOS）の有無（0/1） |
| 21 | D. 能動 | OpAmp の有無（0/1） |
| 22 | D. 能動 | トランジスタ構成 CE（接地エミッタ/ソース＝反転増幅）（0/1） |
| 23 | D. 能動 | トランジスタ構成 CC（コレクタ/ドレイン接地＝フォロワ）（0/1） |
| 24 | D. 能動 | トランジスタ構成 CB（ベース/ゲート接地）（0/1） |
| 25 | D. 能動 | OpAmp 反転アンプ（0/1） |
| 26 | D. 能動 | OpAmp 非反転アンプ（0/1） |
| 27 | D. 能動 | OpAmp ボルテージフォロワ（0/1） |
| 28 | D. 能動 | 帰還の有無（0/1） |
| 29 | D. 能動 | p 型素子（PNP/PMOS）の有無（0/1） |
| 30 | D. 能動 | 能動素子数（正規化、min(n/4,1)） |
| 31 | D. 能動 | ダイオード接続素子の有無（カレントミラー参照側）（0/1） |
| 32 | D. 能動 | 結合ペアの有無（差動/ロングテール）（0/1） |
| 33 | D. 能動 | 差動構成（2入力）（0/1） |
| 34 | B2. ダイオード | 直列ダイオード（両端とも非GND＝整流/昇圧の本線）（0/1） |
| 35 | B2. ダイオード | アノードが入力ポート（真の整流段）（0/1） |
| 36 | B2. ダイオード | シャントダイオード（片端のみGND＝クリッパ/ツェナー）（0/1） |
| 37 | B2. ダイオード | 整流＋出力平滑コンデンサ（直列D かつ 出力→GND の C）（0/1） |
| 38 | A. 部品数 | 抵抗数（正規化、min(n/4,1)） |
| 39 | A. 部品数 | コンデンサ数（正規化、min(n/4,1)） |
| 40 | A. 部品数 | インダクタ数（正規化、min(n/3,1)） |
| 41 | A. 部品数 | ダイオード数（正規化、min(n/4,1)） |
| 42 | A. 部品数 | スイッチ数（正規化、min(n/2,1)） |

> **DZ（ツェナー）について**：部品種別 `"DZ"` を使用すると `has_zener=True` として扱われ、
> 構造が同一の負クリッパ回路と次元18で区別されます。

> **ダイオードの役割（次元 34–37）について**：向き4種（次元9–12）だけでは半波整流・平滑段・
> 昇圧チョッパが同一シグネチャになり衝突するため、直列/シャントの別・アノード位置・出力平滑Cの
> 有無を追加した。これにより半波整流(1,1,0,0)・整流平滑段(1,1,0,1)・昇圧(1,0,0,1)・
> クリッパ/ツェナー(0,0,1,0) が分離する（順に次元34,35,36,37）。

> **能動素子（3端子以上）について**：NPN/PNP/NMOS/PMOS/OPAMP は `feature_extractor.build_graph` が
> デバイスノード `__dev_<id>` として扱い（2端子素子は従来通り辺）、接地構成・帰還・極性を
> 次元 19–29 に符号化します。トランジスタ構成は端子ネットが入力/出力ポートのどちらに近いか
> （受動素子のみを辿り GND は経由しない到達性）で CE/CC/CB を判定します。

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
| 複合 / 複合 | 最適割当（Hungarian, scipy.linear_sum_assignment）ブロックマッチング（余剰ブロックにペナルティ） |

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
| `topo_score` | コサイン類似度（43次元ベクトル, 既定 beta=0.95）と WL カーネルの混合、または階層マッチング | 回路の構造・接続トポロジーの一致度 |
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
pip install PySpice                        # シミュレーション解析を使う場合（任意）
pip install pyyaml                         # evaluate.py --sim を使う場合（任意）
```

### シミュレーション解析（任意）

部品に `value`（部品値）を付けると、`circuit_rag.py` は LLM 判定の前に ngspice で
AC 解析を行い、フィルタ特性・カットオフ周波数などをプロンプトに添えます（詳細は
`docs/SIMULATION_DESIGN.md`）。シミュレーション結果は RAG の類似検索には影響しません。

- 必要なもの: `PySpice` と `ngspice`。
- KiCad は ngspice を**共有ライブラリ**（Windows: `ngspice.dll` / Linux: `libngspice.so` /
  macOS: `libngspice.dylib`）として同梱します。`circuit_simulator.py` は実行ファイル
  （`ngspice.exe` 等）と共有ライブラリの両方を自動検出し、共有ライブラリの場合は PySpice が
  読み込めるよう必要な環境変数（`NGSPICE_LIBRARY_PATH` / `SPICE_LIB_DIR`）を自動設定します。
- ngspice の場所は次の順で解決します: `NGSPICE_PATH`（環境変数）→ `PATH` → KiCad 同梱パス。
- いずれも見つからない / 部品値が無い / SW・ダイオードを含む場合は**シミュレーションをスキップ**し、
  トポロジー情報のみで判定を継続します（実行は止まりません）。

```bash
# ngspice の場所を明示する場合（PATH 上に無いとき）。ファイル/ディレクトリどちらも可。
$env:NGSPICE_PATH = "C:\Users\<user>\AppData\Local\Programs\KiCad\10.0\bin\ngspice.dll"  # PowerShell
export NGSPICE_PATH=/usr/lib/libngspice.so                                               # bash
```

> 検証実績: KiCad 10.0 同梱の ngspice-46 ＋ PySpice 1.5 で、オープンソースの RC ローパス
> （R=100Ω, C=1µF）の AC 解析に成功（lowpass 判定・カットオフ ≈1.6kHz、理論値との誤差約2%）。

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
      "id":   "部品ID（R1, C1, L1, SW1, D1, Q1 など）",
      "type": "部品種別（R / C / L / SW / D / DZ / NPN / PNP / NMOS / PMOS / OPAMP）",
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
- **能動素子（3端子以上）の端子名は役割で指定します**（`build_graph` が端子名で構成を判定するため必須）：
  - BJT（NPN/PNP）: `"base"` / `"collector"` / `"emitter"`
  - MOSFET（NMOS/PMOS）: `"gate"` / `"drain"` / `"source"`
  - OpAmp（OPAMP）: `"in_p"`（非反転）/ `"in_n"`（反転）/ `"out"`
  - 端子数は2に限定されません。2端子素子は辺、3端子以上はデバイスノードとして自動的に扱われます。
- **差動回路など2入力の回路**は `ports` に任意で `"input2"` を追加できます（差動構成の判定に使用）。
- 多素子トポロジーは構造から自動検出されます：**カレントミラー**＝ダイオード接続（制御端子と出力端子が同一ネット）、
  **差動対**＝共通端子（エミッタ/ソース）を非GNDネットで共有する2素子＋2つの独立入力。
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

## KiCad 回路の取り込み（kicad_sch_to_toporag.py）

`.kicad_sch` を `kicad-cli` 経由でネットリスト化し、TopoRAG 形式の JSON に変換します。

```bash
# 単一ファイル
python kicad_sch_to_toporag.py path/to/Circuit.kicad_sch kicad_query.json

# ディレクトリ内の全 .kicad_sch を一括変換
python kicad_sch_to_toporag.py path/to/dir/ kicad_query.json

# 変換した回路を判定
LLM_PROVIDER=claude python circuit_rag.py --query kicad_query.json
```

### 対応部品

| TopoRAG 種別 | KiCad 由来 |
|---|---|
| R / C / L / D / DZ | Sim.Device または part 名・ref 頭文字から解決 |
| NPN / PNP | BJT（ピン役割 `C` / `B` / `E`） |
| NMOS / PMOS | MOSFET（ピン役割 `D` / `G` / `S`、Bulk はスキップ） |
| OPAMP | OpAmp（`Sim.Device=SUBCKT` + part 名 `OPAMP`、役割 `+` / `-` と出力ピン） |

- **電源・電流源（V/I）はスキップ**し、信号源の端子を入力ポート候補にします。
- **電源レール**（`Vdc` / `VCC` / `VDD` / `V+` / `V-` 等）は入出力ポート候補から自動的に除外されます
  （信号入力 `In` / `Vsig` 等と区別）。
- `kicad-cli` の場所は `KICAD_CLI_PATH` → `PATH` → KiCad 標準インストール先の順で解決します。
- 既知の制約: `VDMOS` の極性（p-channel）は未判定で N 型既定。論理ゲート等はスキップ。

### 実機検証（feastorg/KiCad-Simulation-Examples）

実在の KiCad シミュレーション例で、変換 → 分類が一貫して動作することを確認済み：

| 実回路 | 判定結果 | スコア |
|---|---|---|
| RC-Lowpass / RC-Highpass | RCローパス / RCハイパス | 1.00（完全一致、AC解析カットオフも理論と一致） |
| NPN-Amplifier-CE | エミッタ接地増幅回路（NPN） | 0.99 |
| NMOS-Amplifier-CS | ソース接地増幅回路（NMOS） | 0.99 |
| OPamp-inverting / noninverting / voltage-follower | 反転 / 非反転 / ボルテージフォロワ | 1.00 |

---

## 評価（evaluate.py）

検索段（ベクトル検索）と LLM 段を分離して評価します。詳細は `docs/EVALUATION_DESIGN.md`。

```bash
python evaluate.py                              # Section 1,3,4（常時）
python evaluate.py --alpha-sweep                # + Alpha グリッドサーチ
LLM_PROVIDER=claude python evaluate.py --llm    # + LLM 判定精度
python evaluate.py --sim                        # + シミュレーション精度
```

| セクション | 内容 | 指標 |
|---|---|---|
| 1. 自己検索 | DB全回路を自分自身で検索 | Hit@1 / Hit@3 / MRR |
| 2. Alpha スイープ | alpha を 0.0〜1.0 で走査 | トポロジー/タグ重みの最適値 |
| 3. 摂動ロバスト性 | ノード名・部品ID をリネーム | 特徴抽出の名前非依存性 |
| 4. 棄却閾値校正 | Leave-one-out で未知スコアを近似 | 推奨閾値 θ |
| 5. LLM 判定精度 | RAGのみ vs RAG+シミュレーション | 正解名の文字列一致率 |
| 6. シミュレーション精度 | `eval_expected.yaml` と照合 | simulation_type / フィルタ種別 / カットオフ |

摂動失敗・シミュレーション照合失敗は hard failure として終了コード 1 を返します（CI 連携可能）。
回路を追加・削除すると評価対象は自動的に追従します（`sample_netlists.json` をその場で特徴量化）。

> **Section 4 の注意（棄却閾値）**：LOO で得る θ は『素のDB』の自己類似（スコア≈1.0）を
> 基準にした近似で、タグを持たない実機クエリ（トポロジーのみ ≤ alpha）には過大になり全件
> 誤棄却します。実運用の棄却は下記 `reject_eval.py` で実機コーパスに校正してください。

## 実機検証と棄却（validate_real.py / reject_eval.py）

第三者が描いた実機回路（KiCad 同梱 `demos/simulation` 等）で汎化性能と未知回路の棄却を検証します。

```bash
python validate_real.py            # 実機コーパスを DB 検索（Hit@1/Hit@3・構成分類）
python validate_real.py --sim      # 受動回路はシミュレーション種別も表示
python reject_eval.py              # 棄却シグナルの分離性能（AUC・最良閾値）
python ablation.py                 # 特徴グループの寄与分析（ablation）
```

### 汎化性能（実機 in=28・唯一の汎化指標）

第三者が描いた実機回路（タグ無し）で測る。`validate_real.py` の現行値（DB=47 / in-scope 28件）：

| 集計 | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| strict（厳密一致） | **85.7%** [95%CI 68.5–94.3] | 85.7% | 0.879 |
| lenient（近縁許容） | **89.3%** [95%CI 72.8–96.3] | 92.9% [77.4–98.0] | 0.927 |

> ⚠ **自己検索 Hit@1=100% は成果ではない**。特徴量次元（ダイオード役割 34–37・部品数 38–42 等）と
> WL カーネルの `beta=0.95` は、いずれも DB 内・実機コーパス内の既知の取り違えを潰すために後付けで
> 選ばれた値（`circuit_rag.py` のコメントに明記）。同じコーパスで選んだパラメータを同じコーパスの
> 成績として報告しているため、自己検索 100% は「DB に手彫りした特徴が DB を当てられる」という
> 同語反復であり**汎化の証拠ではない**。汎化を示すのは上表（実機 in=28）のみで、CI は依然として広い。

**ablation（`ablation.py`, alpha=1.0 / beta=1.0 コサインのみ）**：ベースラインは自己検索 76.6% /
実機 89.3%。能動次元(19-33)を除くと**実機 Hit@1 が −10.7pt** と最も影響が大きく、ダイオード役割(34-37)・
部品数(38-42)も各 −3.6pt 寄与する。タグは自己検索のみを押し上げ**実機クエリはタグを持たないため
汎化には寄与しません**（自己検索 100% の底上げはタグ依存）。

### 棄却について（実機 in28/out20 で校正・事前登録ゲート合格）

`real_corpus.json` の out-of-scope（未知）回路を **5 → 20 件に拡充**し、`reject_eval.py` の
事前登録ゲート（`docs/HANDOFF_2026-06-09.md` §7.3）を**合格**した：

- top-1 スコアの棄却 **AUC = 0.868**（95%CI [0.757, 0.963]、CI 下限 > 0.5）。
- **nested CV balanced acc = 0.800**（事前登録基準 0.65 以上。楽観バイアスは +0.025 に縮小）。
- ゲート判定 = **「棄却は有意に機能・硬い二値判定を継続」**（n_out=20 ≥ 事前登録 20）。
- 閾値 θ=0.8863 での実績: in 受理 21/28（**TPR 0.75**）/ out 棄却 18/20（**TNR 0.90**）。

> ⚠ **ただし完璧ではない**。分離余裕は **−0.33**（負＝重なりあり）で、`simulation-diode-characteristics`
> (1.0000・ダイオード測定試験ベンチ)・`opamp-freerunning`(0.9517・OpAmp 発振器) の 2 件は DB 収録回路と
> 構造が一致し閾値で切れない（誤受理）。誤棄却 7 件は DB に近傍が無い hard in-scope（多段アンプ・
> コンバータ）。**誤受理を絶対に避けたい用途や境界回路が多い用途では、硬い二値判定ではなく
> top-k＋スコア提示（ランカー）を人が確認する運用**も選択肢（同 §7.3）。

実装は `CircuitRAG.search_with_rejection()`、閾値 `CALIBRATED_REJECT_THRESHOLD`（既定 0.8863）。

- 棄却は**トポロジーのみ(alpha=1.0)の top-1 スコア**で判定（タグ非依存。実機クエリはタグを持たないため）。
- **スイッチング素子の正規化**：DB はコンバータのスイッチを `SW` でモデル化しますが、実機は実
  MOSFET を使います。`feature_extractor.switch_mosfet_ids()` が「L＋D の変換器文脈で drain/source が
  インダクタ節点に接する MOSFET」を SW 相当に正規化するため、実 MOSFET の buck/boost も SW ベース DB に
  一致します（増幅用 MOSFET は不変）。

---

## 既知の限界と設計上の制約

過大評価を避けるため、本システムの構造的な弱点を明記する。

### 1. 特徴量は「パッチの塔」
`switch_mosfet_ids()`・SW-L順序・ダイオード役割・部品数…等の次元は、いずれも**特定の誤分類を1件ずつ
潰すために後付けされた個別ルール**である。新しい回路ファミリを足すたびに新次元が必要になり、構造的に
スケールしない。さらに **次元を1つ足すと全コサイン類似度が動き、校正済みのスコア依存値
（棄却閾値）が無言で無効化される**。閾値がどの特徴セット時点で校正されたかを追跡する仕組みは無い
（学習済みの距離計量ではなく手設計のため）。

### 2. 評価の循環性 — 自己検索 100% は汎化ではない
特徴量次元と WL カーネルの `beta` は、DB 内・実機コーパス内の既知の取り違えを潰すために**同じコーパスで
選ばれている**（テストセットでのチューニング）。その結果の自己検索 Hit@1=100% は同語反復であり、
汎化の証拠ではない。汎化を測れるのは実機 in=28 のみで、その CI は依然広い（上記「汎化性能」）。

### 3. 小標本ゆえ CI が広く、棄却にも構造的な天井がある
実機 in=28 / out=20 に拡充して棄却の事前登録ゲートは合格した（AUC 0.868・nested bacc 0.800）が、
比率の 95%CI は依然 ±0.1〜0.2 あり、Hit@1 の小さな差は有意とは言い切れない。棄却も TNR 0.90 止まりで、
DB と構造が一致する out（測定試験ベンチ・発振器）は閾値で切れない（分離余裕 −0.33）。硬い二値判定の
限界が残るため、境界用途ではランカー運用を併記している。

### 4. 「RAG」ではなく最近傍分類器
実体は 47 個の手書きプロトタイプに対する手設計 43 次元ベクトルの最近傍分類器であり、学習済み埋め込みも
スケール戦略も持たない。LLM 段の役割は検索上位の読み上げ・名称結合のみで、その評価指標
（正解名の文字列一致率）は脆い。

### 改善の方向性（参考）
- 評価基盤を先に直す（out-of-scope を 20件以上に拡充。手順は `docs/HANDOFF_2026-06-09.md`）。
- 硬い棄却を諦め、top-3＋スコア提示のランカーに製品定義を切り替える（撤退ルール §7.3）。
- 手設計次元の塔をやめ、構造カーネル（WL 等）や学習距離計量に寄せる。

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
