# 実験B 引き継ぎ — 知識カード執筆と条件③評価（Claude Code 作業指示）

作成: 2026-07-05　|　リポジトリ: TopoRAG（作業ブランチを切って進める）
前提文書: `docs/CARD_SPEC.md`（カード契約）、`docs/IR_SPEC.md`、`docs/EXPERIMENT_A_DESIGN.md`、
実験A結果 `RESULTS_experiment_A_2026-07-05.md`、骨子改訂メモ（本文書§2に要点を再掲）

---

## 1. 目的（なぜこの作業か）

学会原稿（電気学会研究会・締切8/6）の表1「③提案手法」列を埋める。
実験Aで判明した事実:

- 条件②（検索のみ）のstrictミス4件のうち3件は、**正解が検索5〜6位に残っている**
  （= 裁定層で回復可能な位置にある）:
  | クエリ | ②のtop1（誤） | 正解 | 正解の順位 |
  |---|---|---|---|
  | rectifier | fullwave_ct_rect_001 | halfwave_rect_001 | **6位** |
  | Rectifier-1D | fullwave_ct_rect_001 | halfwave_rect_001 | **5位** |
  | OPamp-headphone | phase_shift_osc_001 | opamp_noninverting_001 | **5位** |
- OPamp-headphone は IR が `active.opamp_config=non_inverting / has_feedback=true` を
  **既に正しく抽出している**のに、距離計量が発振器2種を上位に置いた。
  → 「弁別事実はIRにあり、検索が埋めたものをカード×IR照合の裁定が掘り起こす」を実証する。

**成功基準（事前宣言・変更禁止）**: 条件③（k=6、カード裁定）が上記3クエリのstrictを回復し、
②が正解していた24件を壊さない → strict 27/28。壊れた場合も件数と原因をそのまま報告する
（数字の取捨選択・基準の事後変更は禁止）。

## 2. 凍結物（変更禁止）

- `features_db.json`（DB47回路）・特徴量次元・カーネル係数・棄却閾値 θ=0.8863
- 実機コーパス `real_corpus.json` / 正解 `real_expected.yaml`
- 裁定は **θ通過（in判定）後のみ**起動。out判定（棄却）は閾値規則のまま裁定層に触らせない
  （棄却の決定性を壊さない。原稿の主張の根幹）。
- 裁定層に unknown への逃げは**許可しない**（候補k件からの選択に限定。実験Aで直読みが
  示した過棄却の失敗様式を持ち込まないため）。

## 3. 作業1: 知識カードの執筆（`knowledge_cards.yaml` に追記）

### 3.1 執筆手順（CARD_SPEC 執筆原則の運用）

1. **先に実IRを見る。** 対象DB回路と関連実機クエリのIRを必ずダンプしてから書く。
   決め手は観察されたフィールド値にのみ紐づける（教科書知識の机上記述は禁止。
   IRに現れない事実は decisive に書けない）:
   ```python
   from feature_extractor import extract_hierarchical_features
   from circuit_ir import build_ir
   import json
   db = json.load(open('features_db.json'))
   # DB側・実機側（real_corpus.json）双方の対象回路で build_ir(extract_hierarchical_features(c)) を確認
   ```
2. スキーマは `docs/CARD_SPEC.md` に厳密に従う（name / role 1文 / decisive / confused_with / pitfalls）。
   既存2枚（halfwave_rect_001, negative_clipper_001）が書式の見本。
3. `confused_with.with` は実在の circuit_id のみ。テストで解決検査されることに留意。
4. `pitfalls` は取り違えログ由来の事実を書く（実験Aの誤マッチ・②のミスが一次情報）。

### 3.2 カード一覧（優先度順・計15枚目標）

**族1: 整流族（最優先。rectifier / Rectifier-1D の回復を狙う）**
| circuit_id | 状態 | confused_with に必ず含めるペア |
|---|---|---|
| halfwave_rect_001 | 既存 | **fullwave_ct_rect_001 を追記**（②の実誤マッチ）、既存の negative_clipper / rectifier_cap_stage は維持 |
| fullwave_ct_rect_001 | 新規 | halfwave_rect_001, bridge_rect_001 |
| bridge_rect_001 | 新規 | fullwave_ct_rect_001, voltage_doubler_001 |
| rectifier_cap_stage_001 | 新規 | halfwave_rect_001, boost系（IRで sw_l_order を確認して判断） |
| voltage_doubler_001 | 新規 | bridge_rect_001, rectifier_cap_stage_001 |

弁別キー候補（**必ず実IRで確認してから採用**）: ダイオード数（components.normalized_counts.D）、
直列/シャント（diode.series / diode.shunt）、平滑（diode.rectifier_smoothing）、
センタータップ/トランスの現れ方（IRで実際にどう見えるかを要確認。想定と違えば観察に従う）。

**族2: OpAmp線形 vs 発振器（OPamp-headphone の回復を狙う）**
| circuit_id | 状態 | confused_with に必ず含めるペア |
|---|---|---|
| opamp_noninverting_001 | 新規 | **phase_shift_osc_001, wien_bridge_osc_001**（②の実誤マッチ）, opamp_buffer_001 |
| opamp_buffer_001 | 新規 | opamp_noninverting_001 |
| opamp_inverting_001 | 新規 | opamp_summing_001, mfb_bpf_001 |
| phase_shift_osc_001 | 新規 | opamp_noninverting_001, wien_bridge_osc_001 |
| wien_bridge_osc_001 | 新規 | phase_shift_osc_001, opamp_noninverting_001 |
| mfb_bpf_001 | 新規 | opamp_inverting_001 |

弁別キー候補: active.opamp_config（non_inverting / inverting / buffer …実値を確認）、
active.has_feedback、帰還路のRC段構成（topology.series_sequence / shunt_to_gnd / loop_count に
どう現れるか要確認）。発振器と増幅器の差が IR 上のどのフィールド差になるかが本族の核心。
**もし IR 上で弁別可能な差が存在しない場合は、その旨を作業レポートに明記して止まる**
（カードに書けない差は裁定に使えない。特徴量追加は凍結物のため不可。原稿の限界節の材料になる）。

**族3: LC/RLC共振族（防御的整備。①が落とした族）**
| circuit_id | 状態 | confused_with |
|---|---|---|
| rlc_bandpass_001 | 新規 | rlc_notch_001, lc_highpass_001, lc_lowpass_001 |
| rlc_notch_001 | 新規 | rlc_bandpass_001 |
| lc_highpass_001 | 新規 | lc_lowpass_001, rlc_bandpass_001 |
| lc_lowpass_001 | 新規 | lc_highpass_001, lc_pi_filter_001 |

弁別キー候補: 直列/シャント素子の並び（topology.first_series / series_sequence / shunt_to_gnd）、
has_parallel。直列共振と並列共振の差が IR にどう現れるかを実回路で確認して書く。

### 3.3 IRフィールド早見（circuit_ir.py 実装から。値は必ず実物で確認）

- `topology`: first_series / series_sequence / shunt_to_gnd / sw_l_order / loop_count / has_parallel
- `diode`（D含む回路のみ）: anode_to_gnd / cathode_to_out / series / anode_at_input / shunt /
  rectifier_smoothing / has_zener
- `active`（能動素子含む回路のみ）: devices / n_active / transistor_config / opamp_config /
  p_type / has_feedback / is_inverting / is_follower / is_differential / has_coupled_pair /
  has_diode_connected
- `components`: types / normalized_counts、`hierarchy`（階層回路のみ）

## 4. 作業2: 条件③の評価

1. 裁定実装はコミット `60c07a0`（プロンプト再設計＝カード×IR照合）の経路を使用。
   **候補数を k=6 に設定**（回復対象の正解が5〜6位のため。k=3では原理的に回復不能）。
2. LLM クライアントは実験Aと同じ Claude CLI（`llm_client.py` の CLILLMClient。
   `--system-prompt-file` 修正済み版）。試行3回・多数決（実験Aと同一の集約規則）。
3. 対象: 実機コーパスの **in判定（θ=0.8863 通過）クエリのみ**。out側は②の閾値棄却の
   結果をそのまま表に使う（裁定層は関与しない）。
4. 採点: `real_expected.yaml` の strict / lenient（実験A・②と同一基準）。Wilson 95%CI 併記。
5. 出力: `RESULTS_experiment_B_<date>.md` に、
   - ③の Hit@1 strict/lenient（CI付き）
   - 回復対象3クエリの個別結果（回復/非回復と裁定根拠の引用）
   - **②正解24件のうち③が壊した件数と原因**（0件でもその旨明記）
   - カード枚数・k・プロンプトなど設定の凍結記録
   - 生応答 JSONL のファイルパス

## 5. やらないこと（スコープ外）

- DBへの回路追加、特徴量・閾値の変更（§2 凍結）
- 全47回路分のカード網羅（15枚で十分。「取り違え駆動で必要分のみ執筆」が原稿の設計思想）
- 棄却判定への裁定層の関与
- 成功基準の事後変更・不利な結果の非報告

## 6. 完了条件チェックリスト

- [ ] カード15枚（既存2枚の更新含む）が CARD_SPEC 準拠で `knowledge_cards.yaml` に存在
- [ ] 各カードの decisive が実IRダンプと突き合わせ済み（作業ログに対象IRの抜粋を残す)
- [ ] 既存テスト（confused_with の id 解決検査等）と CI が通る
- [ ] k=6・試行3・多数決で条件③評価が完走、結果レポート作成
- [ ] 成功基準（3件回復・24件無破壊）との照合結果が明記されている（未達でも正直に）
