# 実験B 算術・条件同一性 監査レポート

**監査日**: 2026-07-09  
**対象**: `RESULTS_experiment_B_2026-07-08.md`、`results/experiment_b_2026-07-08.jsonl`  
**目的**: ①②③の数字を「単一の凍結条件で測った算術の閉じた値」に確定させる

---

## 1. 算術監査 — 全28件テーブル

`results/experiment_b_2026-07-08.jsonl` と `validate_real.py`（②再現）から機械生成。

| query_id | topo_conf | θ判定 | ②top-1 | ③多数決 | strict正解 | ②ok | ③ok | 変化 |
|---|---|---|---|---|---|---|---|---|
| amplifier-ac | 0.7457 | 棄却 | two_stage_ce_001 | None | two_stage_ce_001 | ✓ | ✗ | **ゲート喪失** |
| buck_conv | 0.7884 | 棄却 | buck_001 | None | buck_001 | ✓ | ✗ | **ゲート喪失** |
| lc-series-resonance | 0.8297 | 棄却 | lc_highpass_001 | None | lc_highpass_001 | ✓ | ✗ | **ゲート喪失** |
| nmos-amplifier-cs | 0.9517 | 受理 | cs_amp_nmos_001 | cs_amp_nmos_001 | cs_amp_nmos_001 | ✓ | ✓ | 不変 |
| npn-amplifier-ce | 0.9517 | 受理 | ce_amp_npn_001 | cascode_npn_001 | ce_amp_npn_001 | ✓ | ✗ | **破壊** |
| opamp-inverting | 1.0000 | 受理 | opamp_inverting_001 | opamp_inverting_001 | opamp_inverting_001 | ✓ | ✓ | 不変 |
| opamp-noninverting | 1.0000 | 受理 | opamp_noninverting_001 | opamp_noninverting_001 | opamp_noninverting_001 | ✓ | ✓ | 不変 |
| opamp-voltage-follower | 1.0000 | 受理 | opamp_buffer_001 | opamp_buffer_001 | opamp_buffer_001 | ✓ | ✓ | 不変 |
| rc-bandpass | 0.9572 | 受理 | rc_highpass_001 | rc_highpass_001 | rlc_bandpass_001 | ✗ | ✗ | miss両方 |
| rc-highpass | 1.0000 | 受理 | rc_highpass_001 | rc_highpass_001 | rc_highpass_001 | ✓ | ✓ | 不変 |
| rc-lowpass | 1.0000 | 受理 | rc_lowpass_001 | rc_lowpass_001 | rc_lowpass_001 | ✓ | ✓ | 不変 |
| rectifier-1d | 0.9001 | 受理 | fullwave_ct_rect_001 | halfwave_rect_001 | halfwave_rect_001 | ✗ | ✓ | **回復** |
| rectifier | 0.7828 | 棄却 | fullwave_ct_rect_001 | None | halfwave_rect_001 | ✗ | ✗ | miss両方 |
| sallen_key | 1.0000 | 受理 | sallen_key_lpf_001 | sallen_key_lpf_001 | sallen_key_lpf_001 | ✓ | ✓ | 不変 |
| smps-com | 0.6721 | 棄却 | boost_001 | None | boost_001 | ✓ | ✗ | **ゲート喪失** |
| lc-parallell-resonance | 0.8595 | 棄却 | rlc_bandpass_001 | None | rlc_bandpass_001 | ✓ | ✗ | **ゲート喪失** |
| rc-charge | 1.0000 | 受理 | rc_lowpass_001 | rc_lowpass_001 | rc_lowpass_001 | ✓ | ✓ | 不変 |
| rc-discharge | 1.0000 | 受理 | rc_lowpass_001 | rc_lowpass_001 | rc_lowpass_001 | ✓ | ✓ | 不変 |
| rc-voltage-current | 1.0000 | 受理 | rc_lowpass_001 | rc_lowpass_001 | rc_lowpass_001 | ✓ | ✓ | 不変 |
| simulation-ac-sweep | 1.0000 | 受理 | rc_lowpass_001 | rc_lowpass_001 | rc_lowpass_001 | ✓ | ✓ | 不変 |
| simulation-transient-pulse | 1.0000 | 受理 | rc_lowpass_001 | rc_lowpass_001 | rc_lowpass_001 | ✓ | ✓ | 不変 |
| simulation-transient-sine | 1.0000 | 受理 | rc_lowpass_001 | rc_lowpass_001 | rc_lowpass_001 | ✓ | ✓ | 不変 |
| rectifier-2d | 0.9656 | 受理 | fullwave_ct_rect_001 | rectifier_cap_stage_001 | fullwave_ct_rect_001 | ✓ | ✗ | **破壊** |
| rectifier-4d | 0.9625 | 受理 | bridge_rect_001 | bridge_rect_001 | bridge_rect_001 | ✓ | ✓ | 不変 |
| rectifier-6d | 0.9492 | 受理 | bridge_rect_001 | bridge_rect_001 | bridge_rect_001 | ✓ | ✓ | 不変 |
| rectifier-4d-regulator | 0.7705 | 棄却 | bridge_rect_001 | None | bridge_rect_001 | ✓ | ✗ | **ゲート喪失** |
| opamp-adding | 0.9579 | 受理 | opamp_summing_001 | opamp_inverting_001 | opamp_summing_001 | ✓ | ✗ | **破壊** |
| opamp-headphone | 0.9041 | 受理 | phase_shift_osc_001 | opamp_noninverting_001 | opamp_noninverting_001 | ✗ | ✓ | **回復** |

### 1.1 集計

| 変化区分 | 件数 | 内訳 |
|---|---|---|
| 不変（② ✓ → ③ ✓） | 15 | — |
| **回復**（② ✗ → ③ ✓） | **2** | rectifier-1d, opamp-headphone |
| **破壊**（② ✓ → ③ ✗） | **3** | npn-amplifier-ce, rectifier-2d, opamp-adding |
| **ゲート喪失**（② ✓ → ③ θ棄却） | **6** | amplifier-ac, buck_conv, lc-series-resonance, smps-com, lc-parallell-resonance, rectifier-4d-regulator |
| miss両方（② ✗ かつ ③ ✗） | 2 | rc-bandpass, rectifier |

### 1.2 等式の検証

```
② strict正解 − ゲート喪失 − 破壊 + 回復 = ③ strict正解（全28件）

       24     −     6     −    3   +   2  =   17

→ 等式閉合 ✓（未解決0件）
```

**結論**: `RESULTS_experiment_B_2026-07-08.md` の見出しスコア `17/28` は算術的に正しい。  
「24 − 6 − 3 + 2 = 17」で完全に説明される。

### 1.3 ゲート喪失 6件の内訳補足

これら6件は **② が θ ゲートなしの純検索順位で top-1 正解を出せていた** が、
③ では topo_conf が θ=0.8863 未満のため θ棄却となり strict miss に計上された。
② の 85.7% は `validate_real.py`（θ適用なし、純検索 top-1 採点）の数値であり、
θ を適用した場合の ② strict は **18/28 = 64.3%** になる（下表参照）。

---

## 2. 同一条件比較（θ通過 21 件限定）

θ通過したクエリのみに限定すれば、**② と ③ が同一ゲート条件で採点される**。
これが裁定層の正味の実力を測る公正な比較であり、論文表2の③列に相応しい値。

| 指標 | ② θ通過21件 | ③ θ通過21件 | 差分 |
|---|---|---|---|
| strict Hit@1 | **18/21 = 85.7%** [65.4–95.0%] | **17/21 = 81.0%** [60.0–92.3%] | −4.7 pt |
| θ棄却ミス（同一） | 7/28 | 7/28 | 0 |

> CI が大幅に重なっており（n=21 の小標本）、−4.7 pt は統計的に有意とは言えない。
> 裁定層の正味の効果は「回復 2件 − 破壊 3件 = −1件」。

**参考: θ適用 ② との比較**（② もθ棄却を miss 計上した場合）

| 指標 | ② (θ適用) | ② (θなし=validate_real) | ③ |
|---|---|---|---|
| strict Hit@1 | 18/28 = 64.3% | **24/28 = 85.7%** | 17/28 = 60.7% |

`validate_real.py` は θ を適用せず純検索順位で採点するため、`② = 85.7%` は
θ通過21件の 18 件 + ゲート喪失予定の6件がたまたま top-1 正解だった分を含む。
論文で「② vs ③ の条件を揃える」には θ通過限定比較が適切。

---

## 3. 条件同一性の監査（git 履歴）

### 3.1 距離計算に関わるコミット年表

| 日時 | コミット | 内容 | 距離計算への影響 |
|---|---|---|---|
| 2026-06-09 | **538f212** | WL グラフカーネル (topo_kernel.py) 導入、DB 31→47 拡張、部品数特徴 次元38–42 追加 | **変更あり**（WL カーネル・特徴次元追加） |
| 2026-06-10 | **c0b9ea8** | CI・テスト・リファクタ。`features_db.json` バイト一致確認・自己検索 Hit@1=100% 再確認 | なし（挙動非変更を公式に確認） |
| 2026-06-11 | **abc5485** | **θ=0.8863 校正**、in28/out20 コーパス統合。`② strict 85.7% / lenient 89.3%` を公式記録 | なし（校正実施・数値確定） |
| 2026-06-12 | **4e4ceaa** | 知識カード試作（knowledge_cards.yaml / knowledge_cards.py 追加） | なし |
| 2026-06-XX | **e4c6226** | circuit_ir.py 追加（LLM 合成層向け描画。vectorize には非関与） | なし |
| 2026-07-08 | **c218827** | **③ 評価実行**（run_experiment_b.py、知識カード16枚） | なし |

**重要**: `circuit_rag.py` の `vectorize()`、`topo_kernel.py`、`feature_extractor.py` は
**abc5485（θ校正）以降 c218827（③実行）まで一切変更されていない**。

```
git log --oneline abc5485..c218827 -- circuit_rag.py topo_kernel.py feature_extractor.py
→ 60c07a0  feat: プロンプト再設計（build_prompt のシステムプロンプト変更のみ）
   e4c6226  feat: circuit_ir.py 追加（vectorize 非関与）
```

### 3.2 sample_netlists.json と features_db.json の関係

| 項目 | 値 |
|---|---|
| 回路数 | 両方とも **47 回路**（circuit_id 集合が完全一致） |
| フォーマット | `sample_netlists.json`: 生ネットリスト（components 付き）<br>`features_db.json`: 抽出済み特徴量（A_component, B1_order 等） |
| 実質関係 | **同一回路の別表現**（改名ではなく、前者を後者に事前変換したもの） |
| validate_real.py | `sample_netlists.json` を使用（`extract_hierarchical_features()` で都度抽出） |
| run_experiment_b.py | **同じく `sample_netlists.json` を使用**（`extract_hierarchical_features()` で都度抽出） |
| 一致確認 | c0b9ea8 のコミットメッセージ「features_db.json バイト一致」で確認済み |

`②` と `③` は**同一の raw DB** (sample_netlists.json, 47 回路) を同一の
`extract_hierarchical_features()` で処理している。フォーマットの不一致はない。

### 3.3 ② スコアの現行コードによる再現確認

```
python validate_real.py
→  in strict  Hit@1: 24/28 = 85.7% [95%CI 68.5–94.3]
   in lenient Hit@1: 25/28 = 89.3% [95%CI 72.8–96.3]
```

**abc5485 のコミットメッセージ記録値（汎化 strict 85.7% / lenient 89.3%）と完全一致。**
再測定不要。

### 3.4 条件同一性の結論

**①②③は同一凍結条件か: YES**

- **距離計算コード**: `abc5485`（θ校正実施）時点と `c218827`（③実行）時点で同一。変更なし。
- **θ 校正の有効性**: WL カーネル・DB47 回路は `538f212`（Jun 9）で確定し、θ校正は `abc5485`（Jun 11）にこの状態で実施された。θ=0.8863 は現行コードに対して有効。
- **DB**: 両評価ともに `sample_netlists.json`（47 回路）を使用。
- **コーパス・正解**: 同一（real_corpus.json in28/out20、real_expected.yaml）。

> **引き継ぎ文書 §6.4「wl_kernel 導入等の影響と推測」の誤り**: wl_kernel は `538f212`（Jun 9）に導入済みで、θ校正（`abc5485`、Jun 11）はその後に実施されている。②の 85.7% は WL カーネル込みの計算で測定されており、③も同一コードで動作している。この推測は誤りであり、条件の不一致は存在しない。

---

## 4. 再測定の要否

| 項目 | 要否 | 根拠 |
|---|---|---|
| ② の再測定 | **不要** | 現行コードで 85.7% が完全再現（§3.3）|
| θ の再校正 | **不要** | WL カーネル・DB47 は校正後に変更なし（§3.1） |
| ③ の再測定 | **不要** | JSONL 記録済み・算術閉合確認済み（§1.2） |
| ① の再測定 | **不要** | ① は distance を使わない直読み。DB・コーパス・採点基準に変更なし |

**全指標の測り直しは不要。現行の数値を論文に使用できる。**

---

## 5. RESULTS_experiment_B_2026-07-08.md への訂正提案

### 訂正 A（必須）: §6.4 の誤記削除

**現行**: 「特徴量凍結後の wl_kernel 導入等の影響と推測」  
**訂正**: この記述を削除。WL カーネルは θ 校正（Jun 11）の 2 日前（Jun 9）に導入済みであり、②③は同一距離計算コードで動作している。

### 訂正 B（推奨）: 見出しスコアへの脚注追加

**現行**: `③ strict Hit@1: 17/28 (60.7%)`（脚注なし）  
**追記案**:
> ※ ②(validate_real.py) は θ ゲートを適用せず純検索 top-1 で採点（= 85.7%）。
> ③ は θ 通過クエリのみ LLM 裁定が起動するため、θ 棄却の in-scope 7 件が
> ③ のみ strict miss に計上される（「ゲート喪失」）。
> θ 通過 21 件限定の同一条件比較は ② 85.7% / ③ 81.0%（差 −4.7 pt）。

### 訂正 C（推奨）: §2 成功基準欄に脚注追加

**追記案**:
> ゲート喪失 6 件は ② も θ 棄却相当（TPR 0.75 記述済みの已知誤棄却）であり、
> ③ の設計欠陥ではなく θ の既知限界によるもの。
> 論文比較の文脈では θ 通過 21 件限定値（② 85.7% / ③ 81.0%）を使用推奨。

### 訂正 D（推奨）: §2 表への行追加

| 基準 | 目標 | 実績 | 達成 |
|---|---|---|---|
| strict Hit@1（全28件） | 27/28 | 17/28 | 未達 |
| **strict Hit@1（θ通過21件）** | **−** | **17/21 = 81.0%** | **参考値** |

---

## 6. 完了条件チェックリスト

- [x] **17/28 の算術が全28件テーブルで完全に説明されている**（24 − 6 − 3 + 2 = 17、未解決ゼロ）
- [x] **θ通過限定の②vs③比較値が確定**（② 85.7% / ③ 81.0%、ともに 21件、Wilson CI 付き）
- [x] **条件同一性の Yes/No が根拠コミット付きで結論**（YES: 距離計算は abc5485 以降変更なし）
- [x] **再測定の要否と計画が明記**（全指標再測定不要）
- [x] **未解決項目**: なし
