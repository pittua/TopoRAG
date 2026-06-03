# TopoRAG シミュレーション統合 設計・仕様書

## 設計方針

トポロジー解析（RAG）による回路種別識別に、ngspice シミュレーションによる定量的な回路特性を統合する。  
シミュレーション結果は RAG の類似検索ベクトルには含めず、**後段の独立したモジュール**として実装する。  
シミュレーション結果は LLM のプロンプトに含め、LLM が構造的根拠と定量特性の両方をもとに判定できるようにする。

```
ネットリスト
  │
  ▼
RAG 類似検索                    ← 既存。変更なし
  │  トポロジー特徴量のみで類似回路を検索
  │
  ▼
シミュレーション解析             ← 今回追加（LLM より先に実行）
  │  ngspice で AC / DC / 過渡解析を実行
  │  → カットオフ周波数、利得特性、フィルタ種別 等
  │
  ▼
LLM 判定                        ← トポロジー特徴量 ＋ シミュレーション結果を両方受け取る
  │  構造的根拠と定量特性を合わせて回路種別を判定
  │
  ▼
統合出力
     識別結果（LLM）＋ シミュレーション要約
```

---

## シミュレーション結果を LLM に渡すことについて

### メリット

| メリット | 内容 |
|---|---|
| 定量的な根拠を含む説明 | 「ローパスフィルタ」という種別だけでなく「カットオフ1,592Hz、DC利得0dB」まで含めた説明をLLMが生成できる |
| 識別精度の向上 | トポロジーが曖昧な回路（RLCバンドパス vs ノッチ等）をシミュレーション結果で補完できる |
| 矛盾の検出 | トポロジーが示す種別とシミュレーション特性が一致しない場合、LLMが矛盾を指摘できる |
| 設計検証への応用 | 種別の識別にとどまらず、仕様（fc の妥当性等）についてもLLMが言及できる |

### リスクと対策

| リスク | 対策 |
|---|---|
| スキップ回路（SW含む）ではシミュレーション結果がNone | プロンプトにスキップ理由を明記し、LLMがトポロジーのみで判定するよう誘導する |
| D/DZを含む回路でシミュレーション精度が下がる場合がある | `simulation_type` フィールドで解析種別を明示し、LLMが信頼度を考慮できるようにする |
| シミュレーション結果が予期しない値になった場合のLLMの混乱 | 整理済みの特徴量（boolean フラグ + dB値 + Hz値）として渡すため、LLMの誤解は起きにくい |

### LLMがシミュレーション結果を理解できる根拠

Claude・GPT-4・Gemini などの LLM は電気工学の教科書・論文・技術文書を大量に学習しており、以下を理解している。

- 「DC利得0dB、高周波利得-62dB」→ ローパス特性
- 「カットオフ周波数1,592Hz」→ R=1kΩ、C=100nFのRC回路に対応
- 「共振あり、ピーク利得+3dB以上」→ RLC回路の共振現象

また、渡す情報はすでに解釈しやすい形（boolean フラグ・dB 値・Hz 値）に整理されており、生の波形データではないため誤解が生じにくい。

---

## シミュレーションバックエンド

### 採用：ngspice + PySpice

| 項目 | 内容 |
|---|---|
| シミュレータ | ngspice（KiCad にバンドル済み） |
| Python インターフェース | PySpice（`pip install PySpice`） |
| 追加インストール | ngspice は KiCad 同梱のため原則不要 |

### 自作 MNA を採用しない理由

自作 MNA が対応できるのは**線形素子（R/L/C）のみ**であり、素子の拡張に追従できない。

| 素子 | 自作 MNA | ngspice |
|---|---|---|
| R / L / C | ✓ 正確 | ✓ 正確 |
| D / DZ | △ 近似（開放扱い） | ✓ 非線形モデル |
| MOSFET / BJT | ✗ 不可 | ✓ 対応 |
| Op-Amp | ✗ 不可 | ✓ 対応 |
| 過渡解析 | ✗ 不可 | ✓ 対応 |

### ngspice のパス解決（環境非依存）

ngspice の場所はハードコードせず、以下の順で探索する。最終的に見つからない場合は
**例外を投げずにシミュレーションをスキップ**し、`simulation_type="skipped_no_ngspice"`
を返してトポロジーのみで判定を継続する（環境構築の失敗で全体が止まらないようにする）。

```python
def resolve_ngspice_path() -> str | None:
    # 1. 環境変数による明示指定（最優先）
    if env := os.environ.get("NGSPICE_PATH"):
        if os.path.exists(env):
            return env
    # 2. PATH 上の ngspice
    if found := shutil.which("ngspice"):
        return found
    # 3. KiCad 同梱の既知パス候補（バージョン非依存にグロブ探索）
    for base in (
        r"C:\Program Files\KiCad",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\KiCad"),
    ):
        for hit in glob.glob(os.path.join(base, "*", "bin", "ngspice.exe")):
            return hit
    return None   # → 呼び出し側で skipped_no_ngspice にフォールバック
```

> 優先順位：`NGSPICE_PATH`（環境変数）→ `PATH` → KiCad 同梱パスのグロブ探索。
> 開発環境固有の絶対パスを設計・コードに埋め込まない。README には
> `NGSPICE_PATH` の設定方法と ngspice 単体インストール手順を記載する。

---

## 変更ファイル一覧

| ファイル | 種別 | 変更内容 |
|---|---|---|
| `circuit_simulator.py` | **新規** | ngspice ラッパー・解析結果の特徴量化 |
| `sample_netlists.json` | **変更** | 各部品に `"value"` フィールドを追加 |
| `query_netlists.json` | **変更** | 各部品に `"value"` フィールドを追加 |
| `kicad_sch_to_toporag.py` | **変更** | KiCad `Value` プロパティから部品値を抽出 |
| `circuit_rag.py` | **変更** | `analyze()` 追加・`build_prompt()` にシミュレーション結果を組み込み |
| `README.md` | **変更** | ngspice インストール手順を追記（実装時） |

`feature_extractor.py` と RAG のベクトル次元は**変更しない**。

---

## 1. `circuit_simulator.py`（新規）

### 役割

ネットリスト（部品値あり）を PySpice 経由で ngspice に渡し、解析結果を回路特性として返す。

### SI 接頭辞パーサ `parse_value()`

ネットリストの文字列値を float に変換する。

```python
parse_value("1k")    # → 1000.0
parse_value("100n")  # → 1e-7
parse_value("10uF")  # → 1e-5
parse_value("4.7m")  # → 0.0047
parse_value(None)    # → None
```

対応接頭辞：`f / p / n / u / μ / m / k / K / M / G`

### 解析種別

| 解析種別 | 対象回路 | ngspice コマンド |
|---|---|---|
| AC 解析 | R/L/C フィルタ全般 | `.ac dec 100 1 10Meg` |
| DC 解析 | 定電圧・整流回路 | `.dc` |
| 過渡解析 | Buck / Boost 等スイッチング回路 | `.tran`（将来拡張） |

### 回路種別ごとの処理方針

整流・クリッパ・ツェナーの本質は**大信号の非線形挙動**であり、`.ac`（動作点まわりの
線形小信号解析）では「ローパスらしい」等の誤った特性が出てLLMにノイズを渡す恐れがある。
このため、ダイオード回路は AC 解析を行わず、**過渡解析（入力正弦波 → 出力波形）**で扱う。
過渡解析が未実装の段階では `skipped_nonlinear` としてスキップし、トポロジーのみで判定する。

| 回路の構成 | `simulation_type` | 解析種別 |
|---|---|---|
| R / L / C のみ | `ac_passive` | AC 解析 |
| D / DZ 含む（SW なし） | `tran_nonlinear`（未実装時は `skipped_nonlinear`） | 過渡解析（大信号挙動）。AC 解析は用いない |
| SW 含む | `skipped_switch` | スキップ（将来：過渡解析で対応） |
| 部品値が欠落 | `skipped_missing_values` | スキップ |
| ngspice 未検出 | `skipped_no_ngspice` | スキップ |

### フィルタ特性の判定

判定に使うしきい値は**マジックナンバーをコードに散らさず、名前付き定数として一箇所に集約**する。
特に RLC の bandpass / notch や「共振 vs バンドパス」は Q 値・部品値次第でしきい値境界に
乗りやすいため、定数は採用部品値（Q=1）での実測により調整・回帰テストで固定する（後述のテスト戦略を参照）。

```python
# --- 判定しきい値（定数として集約。実測で調整する）---
RATIO_PASS_STOP   = 2.0    # 通過/阻止を分ける利得比
RATIO_RESONANCE   = 1.5    # 共振とみなすピーク比
GAIN_FLOOR        = 0.05   # 有意な利得とみなす下限（線形値）
EDGE_MARGIN_DEC   = 0.5    # 端点アーティファクト除外（解析レンジ端からの余裕・dec）
```

DC利得・高周波利得は固定周波数（1Hz / 10MHz）ではなく、**解析レンジ端から
`EDGE_MARGIN_DEC` だけ内側の点**で評価し、端点アーティファクトと fc の低い/高い回路での
誤判定を避ける。解析レンジは採用部品値の fc を内包するよう動的に決める。

```
f_lo = 解析下限を EDGE_MARGIN_DEC デケード内側に寄せた周波数
f_hi = 解析上限を EDGE_MARGIN_DEC デケード内側に寄せた周波数
dc   = |H(f_lo)|      # 低域利得（DC 利得の代理）
hf   = |H(f_hi)|      # 高域利得
pk   = max(|H|)       # ピーク利得
mn   = min(|H|)       # 最小利得

is_lowpass   : dc > hf × RATIO_PASS_STOP   かつ  dc > GAIN_FLOOR
is_highpass  : hf > dc × RATIO_PASS_STOP   かつ  hf > GAIN_FLOOR
is_bandpass  : pk > max(dc, hf) × RATIO_PASS_STOP  かつ  ピークが端点以外
is_bandstop  : mn < min(dc, hf) / RATIO_PASS_STOP  かつ  dc・hf ともに GAIN_FLOOR 超
has_resonance: pk > max(dc, hf) × RATIO_RESONANCE
```

> より厳密には、DC 利得は `.op`（直流動作点）で求めるのが堅牢。AC 解析の下限点で
> 代用する場合は上記の端点マージンを必ず適用する。

カットオフ周波数（−3 dB 点）：

```
ローパス : |H| が dc / √2 を下回る最初の周波数
ハイパス : |H| が hf / √2 を超える最初の周波数
```

### 返り値の構造

`confidence` と `warnings` を必ず含め、しきい値境界ケースや近似の不確かさを
LLM が信頼度として考慮できるようにする（リスク表の「信頼度を考慮」と直結）。

```python
# 解析成功時
{
    "simulation_type": "ac_passive",
    "dc_gain_db":       0.0,
    "hf_gain_db":      -61.94,
    "peak_gain_db":     0.0,
    "peak_freq_hz":     1.0,
    "is_lowpass":       True,
    "is_highpass":      False,
    "is_bandpass":      False,
    "is_bandstop":      False,
    "has_resonance":    False,
    "cutoff_freq_hz":   1592.0,
    "confidence":       "high",      # high / medium / low
    "warnings":         [],          # 例: ["利得比がしきい値境界に近い"]
}

# スキップ時
{
    "simulation_type": "skipped_switch",
    "dc_gain_db":      None,
    "confidence":      None,
    "warnings":        ["スイッチング回路のためスキップ"],
    ...（その他のフィールド None）
}
```

- `confidence` の決め方：利得比がしきい値（`RATIO_*`）から十分離れていれば `high`、
  境界付近なら `medium`、ダイオード近似など本質的に不確かなものは `low`。
- `warnings` には境界ケース・近似・スキップ理由を文字列で積む。

---

## 2. ネットリストフォーマット変更

### 追加フィールド

```json
{
  "id": "R1",
  "type": "R",
  "value": 1000,
  "terminals": {"p": "N1", "n": "N2"}
}
```

- 単位：Ω（R）、F（C）、H（L）
- `value` がない場合はシミュレーションをスキップ（既存動作に影響なし）

### 採用値とカットオフ周波数

| 回路 | 部品値 | fc |
|---|---|---|
| RC ローパス / ハイパス | R = 1 kΩ、C = 100 nF | 1,592 Hz |
| LC ローパス / ハイパス | L = 1 mH、C = 100 nF | 50.3 kHz |
| RLC バンドパス / ノッチ | R = 100 Ω、L = 1 mH、C = 100 nF | f₀ = 50.3 kHz、Q = 1 |
| π / T 型 LC フィルタ | L = 1 mH、C₁ = C₂ = 100 nF | 50.3 kHz |
| RC 2 段カスケード | R₁ = R₂ = 1 kΩ、C₁ = C₂ = 100 nF | 2 次特性 |
| Buck / Boost | L = 10 μH、C = 100 μF | スキップ（SW あり） |
| 整流 / クリッパ / ツェナー | R = 1 kΩ | 過渡解析（大信号）。未実装時はスキップ |

---

## 3. `kicad_sch_to_toporag.py` の変更

KiCad ネットリストの `Value` プロパティを部品値として抽出する。

```python
from circuit_simulator import parse_value   # 追加

# _parse_netlist() 内
raw_value    = props.get("Value", "")
parsed_value = parse_value(raw_value)

components.append({
    "ref":        ref,
    "sim_device": sim_device,
    "pin_role":   pin_role,
    "value":      parsed_value,   # 追加
    "value_raw":  raw_value,      # デバッグ用
})

# convert_kicad_sch() 内
toporag_comps.append({
    "id":        ref,
    "type":      ttype,
    "value":     comp.get("value"),   # 追加
    "terminals": terminals,
})
```

---

## 4. `circuit_rag.py` の変更

### 処理順序

シミュレーションを LLM より先に実行し、結果をプロンプトに含める。

```
analyze() の内部:

  1. CircuitSimulator()   ← 先にシミュレーション実行
  2. self.judge()         ← シミュレーション結果をプロンプトに含めて LLM へ
  3. 統合出力             ← LLM 回答 ＋ シミュレーション要約
```

### `build_prompt()` へのシミュレーション結果の組み込み

LLM に渡すプロンプトにシミュレーション結果を追加するセクションを設ける。

```
## シミュレーション解析結果

  解析種別      : AC解析（パッシブ）
  フィルタ特性  : ローパス
  DC 利得       :  0.00 dB
  高周波利得    : -61.94 dB
  カットオフ    : 1,592 Hz
  共振          : なし
  信頼度        : high
  注意          : （warnings があれば列挙）

※ スキップの場合（スイッチング / 非線形 / 部品値欠落 / ngspice 未検出）:
  解析種別      : スキップ（理由を明記）
  ※ トポロジー情報のみで判定してください。

※ ダイオード回路を過渡解析した場合の注記（confidence=low のとき必須）:
  ※ この結果は大信号の過渡挙動に基づく。AC（小信号）特性とは性質が異なるため、
     フィルタ特性フラグは参考値として扱ってください。
```

> `confidence` と `warnings` は必ずプロンプトに出力し、LLM が定量結果をどの程度信頼すべきかを
> 判断できるようにする。`confidence=low`（ダイオード近似など）では上記の注記を添える。

### 追加メソッド `analyze()`

RAG 検索は **1 回だけ**実行する。`judge()` 内部でも検索していたため、従来案では同じ検索が
二重に走っていた。これを避けるため、`analyze()` で検索を一度行い、その結果を `judge()` に
渡す（`judge()` は hits が渡されたら再検索しない）。

```python
def analyze(self, circuit: dict, top_k: int = 3, alpha: float = 0.7) -> dict:
    """
    シミュレーション解析 → RAG 検索 → LLM 判定 の順に実行し、結果を統合して返す。
    RAG 検索は 1 回だけ実行し、その結果を judge() に渡す（二重検索を避ける）。

    Returns:
        {
            "identification":     str,        # LLM による識別結果テキスト
            "top_hits":           list[dict],  # RAG 検索上位結果
            "simulation":         dict,        # シミュレーション特徴量
            "simulation_summary": str,         # 人が読める要約
        }
    """
    from circuit_simulator import CircuitSimulator

    # Step 1: シミュレーション（LLM より先に実行）
    sim_result = CircuitSimulator(circuit).extract_simulation_features()

    # Step 2: RAG 検索（ここで 1 回だけ実行）
    hits = self.search(
               extract_hierarchical_features(circuit),
               top_k=top_k, alpha=alpha
           )

    # Step 3: LLM 判定（検索結果とシミュレーション結果を渡す。再検索しない）
    identification = self.judge(circuit, hits=hits, sim_result=sim_result,
                                top_k=top_k, alpha=alpha)

    return {
        "identification":     identification,
        "top_hits":           hits,
        "simulation":         sim_result,
        "simulation_summary": _format_sim_summary(sim_result),
    }
```

> `judge()` のシグネチャは `judge(self, circuit, hits=None, sim_result=None, ...)` とし、
> `hits is None` のときのみ内部で `search()` を呼ぶ（既存の単独呼び出しとの後方互換を維持）。

### シミュレーション要約フォーマッタ `_format_sim_summary()`

```
【シミュレーション解析】
  解析種別         : AC解析（パッシブ）
  DC 利得          :  0.00 dB
  高周波利得        : -61.94 dB
  フィルタ特性      : ローパス
  カットオフ周波数   : 1,592 Hz
  共振              : なし
  信頼度            : high
  注意              : （warnings があれば列挙、なければ省略）
```

スキップ時（理由を明記）：

```
【シミュレーション解析】
  解析種別 : スキップ（スイッチング回路 / 非線形 / 部品値欠落 / ngspice 未検出）
```

---

## 実行フローと出力例

```
=======================================================
入力: RC ローパスフィルタ（クエリ）
=======================================================

--- LLM 判定（トポロジー ＋ シミュレーション） ---
【判定】RCローパスフィルタ
【根拠】直列R・シャントCの接続構造がRCローパスと一致する。
        シミュレーションでカットオフ1,592Hz・DC利得0dBを確認。
        トポロジーと動作特性が一致する。
【類似度の解釈】1位スコア0.98で既存DBのRCローパスに強く一致。

--- シミュレーション解析 ---
【シミュレーション解析】
  解析種別         : AC解析（パッシブ）
  DC 利得          :  0.00 dB
  高周波利得        : -61.94 dB
  フィルタ特性      : ローパス
  カットオフ周波数   : 1,592 Hz
  共振              : なし
```

---

## テスト戦略

しきい値（`RATIO_*` / `GAIN_FLOOR`）と基準周波数の妥当性を担保するため、各 DB 回路の
理論値に対する回帰テストを用意する。これがしきい値調整時の安全網となる。

| テスト対象 | 期待値 | 許容誤差 |
|---|---|---|
| カットオフ周波数 | 採用部品値の理論 fc（例: RC = 1,592 Hz、LC = 50.3 kHz） | ±5 % |
| フィルタ特性フラグ | 各回路の期待種別（lowpass / highpass / bandpass / notch） | 完全一致 |
| `has_resonance` | RLC バンドパス / ノッチで True、RC / LC 単段で False | 完全一致 |
| `confidence` | 明瞭な回路で `high`、境界ケースで `medium`/`low` | 区分一致 |
| `parse_value()` | SI 接頭辞の変換（`1k`→1000、`100n`→1e-7 等） | 完全一致 |
| ngspice 未検出時 | `skipped_no_ngspice` を返し例外を投げない | 完全一致 |

- bandpass / notch のように境界に乗りやすいケースは、Q=1 の採用値で実測した値を基準に固定する。
- ngspice に依存するテストは、未検出環境では `skip` 扱いにして CI を壊さない。

---

## 変更しないもの

| 項目 | 理由 |
|---|---|
| `feature_extractor.py` | トポロジー特徴量はシミュレーションと独立して管理する |
| `vectorize()` の次元数 | 類似検索はトポロジーのみで行う |
| `block_decomposer.py` | 変更不要 |
| `llm_client.py` | 変更不要 |

---

## 今後の拡張候補

| 拡張 | 概要 |
|---|---|
| 過渡解析 | `.tran` によるスイッチング回路（Buck / Boost）の時間領域解析 |
| 小信号モデル | スイッチング回路の平均化モデルによる AC 解析 |
