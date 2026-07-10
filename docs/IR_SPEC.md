# 構造IR 仕様（IR_SPEC）

知覚層（feature_extractor / block_decomposer）が出力する特徴 dict を、LLM 合成層が
消費する **決定的な構造IR** に翻訳する契約。実装は `circuit_ir.py`。

方針再定義（2026-06-11）の3層アーキテクチャ
「知覚層(決定的IR) → 構造検索 → LLM合成」の **第1層の出力仕様**をここに固定する。

- 現行版: `IR_VERSION = "1.0"`
- 生成: `build_ir(features) -> dict`（構造化・JSON 直列化可能）
- 描画: `render_ir(ir) -> str`（LLM 可読テキスト。プロンプトに渡る唯一の描画経路）

> ⚠ **前提（欠陥1）への反証あり（2026-06-29）。** 本仕様は「LLMはグラフ/netlistを読めない」を
> 前提に消化済みIRを渡す設計だが、実機検証（`docs/EXP_ir_readability.md`）でこの前提は揺らいだ:
> - frontier級モデルは**生netlist（値+ポート付き）を小中規模〜240部品まで正答率100%**で読み、
>   現行 `render_ir` は同条件で **31%** かつ**誤誘導**（2段アンプを「1段」と誤答＝能動素子の個数を保持しない）。
> - 「消化が生netlistに勝つ領域（大規模/弱モデル）」は**見つからなかった**。
>
> 含意: **Q&A経路には現行IRより忠実netlistを渡すべき。** 本IRは検索/照合用途に限定し、
> オープンQ&Aの第1層としては再設計（または忠実netlistへの置換）を要する。詳細は EXP_ir_readability.md。

## 設計原則

1. **消化済みの構造事実だけを載せる。** 生のネットリスト隣接（どのネットが何に
   繋がるか）は **載せない**。直列列・シャント・ループ・能動構成・ダイオード役割
   といった、知覚層が既に解いた構造事実だけを渡す。LLM にグラフ走査をさせない
   こと自体が欠陥1（LLMはグラフ構造を追えない）への解だからである。
   （※ 2026-06-29 反証: この「欠陥1」は frontier 級では成立せず、生netlist を消化なしで
   読める。さらに事実を捨てる本原則は Q&A では逆効果と判明。上部の ⚠ 注記を参照。）
2. **決定的。** 同じ `features` からは常に同じ IR が出る（リストの順序も固定）。
3. **欠落セクションは `None`。** ダイオード / 能動 / 階層は、その素子・構造を
   含む回路にのみ出現し、無ければ `None`。描画側はセクションを省略する。
4. **版付き。** スキーマを変えるときは `IR_VERSION` を上げる。`tests/test_circuit_ir.py`
   が版とスキーマの回帰ガード。

## スキーマ（IR_VERSION = "1.0"）

| フィールド | 型 | 意味 |
|---|---|---|
| `ir_version` | str | スキーマ版。`"1.0"` |
| `circuit_name` | str \| None | 回路名（DB エントリは保持。クエリは無いことがある） |
| `tags` | list[str] | 機能タグ（DB のラベル。クエリは空。検索の既定 alpha=1.0 では不使用） |
| `description` | str | 説明文（DB メタデータ） |
| `ports` | dict | `input` / `output` / `gnd`（差動は `input2` も） |
| `components` | dict | `types`(sorted distinct), `normalized_counts`(SW化MOSFET・DZ→D 正規化済み) |
| `topology` | dict | 下表 |
| `diode` | dict \| None | ダイオードを含む回路のみ。下表 |
| `active` | dict \| None | 能動素子を含む回路のみ。下表 |
| `hierarchy` | dict \| None | 複合回路のみ。下表 |

### `topology`

| キー | 型 | 意味 |
|---|---|---|
| `first_series` | str \| None | 入力→出力 直列経路の先頭部品型 |
| `series_sequence` | list[str] | 直列部品の型列 |
| `shunt_to_gnd` | list[str] | GND 並列（シャント）部品の型列 |
| `sw_l_order` | str \| None | `SW_before_L` / `L_before_SW`（昇降圧コンバータの弁別） |
| `loop_count` | int | 閉路数（`cycle_basis`） |
| `has_parallel` | bool | 同一ノード対に並列部品があるか |

### `diode`（ダイオード回路のみ）

| キー | 型 | 意味 |
|---|---|---|
| `anode_to_gnd` | bool | アノードが GND |
| `cathode_to_out` | bool | カソードが出力 |
| `series` | bool | 信号経路上の直列ダイオード（両端非GND） |
| `anode_at_input` | bool | アノードが入力ポート（真の整流段） |
| `shunt` | bool | 片端のみ GND（クリッパ/ツェナー型シャント） |
| `rectifier_smoothing` | bool | 直列D ＋ 出力→GND の平滑コンデンサ |
| `has_zener` | bool | ツェナー素子(DZ)を含む |

> `series` / `anode_at_input` / `shunt` / `rectifier_smoothing` は改善(a)（dim34-37）の
> 役割次元。整流 / クリッパ / 平滑の弁別カードの決め手になる。

### `active`（能動回路のみ）

| キー | 型 | 意味 |
|---|---|---|
| `devices` | list[str] | `BJT` / `MOSFET` / `OpAmp` の含有 |
| `n_active` | int | 能動素子数 |
| `transistor_config` | str \| None | `CE`(接地エミッタ/ソース) / `CC`(フォロワ) / `CB`(ベース/ゲート接地) |
| `opamp_config` | str \| None | `inverting` / `non_inverting` / `buffer` / `comparator` |
| `p_type` | bool | p 型素子を含む |
| `has_feedback` | bool | 帰還あり |
| `is_inverting` | bool | 反転構成 |
| `is_follower` | bool | フォロワ構成 |
| `is_differential` | bool | 差動（結合ペア＋2独立入力） |
| `has_coupled_pair` | bool | 共通端子を共有する2素子（差動対/ロングテール） |
| `has_diode_connected` | bool | ダイオード接続（カレントミラー参照側） |

### `hierarchy`（複合回路のみ）

| キー | 型 | 意味 |
|---|---|---|
| `n_blocks` | int | ブロック数 |
| `blocks` | list[dict] | 各 `{types, series_sequence, shunt_to_gnd}` |

## 描画 `render_ir`

`_fmt`（旧プロンプト整形）の後継。構造IR を日本語ラベル付きテキストに落とす。
`circuit_rag._fmt` はこの経路に委譲しており、プロンプトに渡る回路記述はすべて
`render_ir(build_ir(features))` を通る。

## 後続ステップとの関係

- **②知識カード**: `diode` 役割次元・`active` 構成・`topology` 列が、近縁回路の
  「識別の決め手・差分」を書く材料になる。
- **③プロンプト再設計**: system 文を「推論不要」から「カードと IR を照合して裁定」に
  変える際、IR がその照合対象。
- **④3アーム評価**: アームB(+IR)/アームC(+IR+RAG) が渡すのがこの IR。
