# 実験A 結果レポート — LLM直読みベースライン（条件①）

**設計書**: `docs/EXPERIMENT_A_DESIGN.md`（凍結版 2026-07-05）
**実行日**: 2026-07-05 | **採点**: 機械採点（P1）＋キーワード写像（P2）
**統計**: 全率に Wilson 95%CI を併記

> **モデル**: Claude CLI（`claude --print --system-prompt-file`）
> **Gemini**: 個人向け CLI サービス終了により**未実施**（認証エラー）

---

## 0. コーパスと設定

| 項目 | 設定 |
|---|---|
| コーパス | `real_corpus.json` 48件（in-scope 28 / out-of-scope 20） |
| P1 試行数 | クエリごと 3試行 → 多数決（最頻値）を最終回答 |
| P2 試行数 | 1試行（補助条件） |
| 候補一覧順 | seed=42 固定シャッフル（全クエリ共通） |
| 入力情報 | components の type / terminals / value を含む生ネットリスト JSON |

---

## 1. P1 主条件 — 閉集合選択＋「unknown」

### 1.1 確定スコア（マージ後）

| 指標 | k / n | 率 | Wilson 95%CI |
|---|---|---|---|
| Hit@1 strict（in 28） | 22 / 28 | **78.6%** | 60.5–89.8% |
| Hit@1 lenient（in 28） | 24 / 28 | **85.7%** | 68.5–94.3% |
| 棄却成功（unknown 回答、out 20） | 19 / 20 | **95.0%** | 76.4–99.1% |
| 誤受理（既知 id を回答、out 20） | 1 / 20 | **5.0%** | 0.9–23.6% |
| 3試行完全一致（全 48） | 45 / 48 | **93.8%** | 83.2–97.9% |
| 多数決不成立（全 48） | 0 / 48 | **0.0%** | 0.0–7.4% |
| 無効票 JSON 不能（全 144票） | 3 / 144 | **2.1%** | — |

### 1.2 提案手法②との比較

| 指標 | 提案手法②（検索のみ, θ=0.8863） | Claude 直読み① | 差分（①−②） |
|---|---|---|---|
| strict Hit@1 | 85.7% | 78.6% | −7.1 pt |
| lenient Hit@1 | 89.3% | 85.7% | −3.6 pt |
| out 棄却成功 | 90.0% | **95.0%** | **+5.0 pt** ← 逆転 |
| 誤受理 | — | 5.0% | — |
| 再現性 | **100%**（構成上保証） | 93.8% | −6.2 pt |

### 1.3 in-scope strict ミス詳細（6件）

| クエリ id | Claude 多数決 | 正解（strict 先頭） | lenient? | 所見 |
|---|---|---|---|---|
| amplifier-ac | unknown | two_stage_ce_001 | ✗ | 多段 CE 増幅を棄却 |
| lc-parallell-resonance | rlc_notch_001 | rlc_bandpass_001 | ✗ | 並列共振をノッチと誤同定 |
| lc-series-resonance | rlc_bandpass_001 | lc_highpass_001 | **○** | 近縁の bandpass を選択 |
| rc-bandpass | unknown | rlc_bandpass_001 | ✗ | RC BPF を棄却 |
| rectifier-1d | rectifier_cap_stage_001 | halfwave_rect_001 | **○** | 近縁整流を選択 |
| rectifier-6d | unknown | bridge_rect_001 | ✗ | 6ダイオード整流を棄却 |

### 1.4 out-of-scope 誤受理（1件）

| クエリ id | Claude 回答 | 所見 |
|---|---|---|
| actuator-npn-switch-led | ce_amp_npn_001 | NPN スイッチ/LED 駆動を CE 増幅と混同 |

---

## 2. P2 補助条件 — 自由記述

*P2 は補助条件。本文考察用。表1には載せない。*

### 2.1 in-scope（機械採点）

| 指標 | k / n | 率 |
|---|---|---|
| family 一致（lenient 相当） | 22 / 28 | **78.6%** |

### 2.2 out-of-scope（キーワード写像＋判定ログ）

| 指標 | k / n | 率 | 備考 |
|---|---|---|---|
| 棄却（「不明」回答） | 1 / 20 | **5.0%** | kicad-simulation-commands のみ |
| キーワード誤マップ | 8 / 20 | 40.0% | 既存 family に誤写像 |
| 人手判定必要（None/MULTI） | 11 / 20 | 55.0% | 下表参照 |

### 2.3 out-scope 全件判定ログ

| クエリ id | Claude 回答（冒頭） | 自動写像 | 判定 |
|---|---|---|---|
| kicad-simulation-commands | 不明 | unknown | **棄却** |
| actuator-npn-switch-led | NPN トランジスタスイッチ（LED 駆動） | None | 正同定 |
| actuator-pnp-switch-led | PNP トランジスタ LED スイッチ | None | 正同定 |
| class-d | Class-D アンプ | None | 正同定 |
| opamp-freerunning | 電圧コンパレータ（発振回路？） | None | 正同定（発振器） |
| generic_opamp_bip | 反転増幅（オペアンプ）＋NPN 複合 | MULTI | 人手判定要 |
| hv_converter | コッククロフト・ウォルトン高電圧変換 | smps_converter | 正同定（SMPS 族） |
| laser_driver | レーザーダイオードドライバ | None | 正同定 |
| royer1 | Royer コンバータ DC-DC | smps_converter | 正同定（SMPS 族） |
| rectifier-4d-opamp-npn | ブリッジ整流・精密化回路 | rectifier | 人手判定要 |
| rectifier-4d-regulator-npn | ブリッジ＋NPN レギュレータ電源 | rectifier | 正同定（DB 外変種） |
| rectifier-4d-regulator-pnp | ブリッジ＋PNP シリーズパス電源 | rectifier | 正同定（DB 外変種） |
| rectifier-4d-zener-npn | ブリッジ＋ツェナー NPN レギュレータ | rectifier | 正同定（DB 外変種） |
| transformer-1p-1s | トランス（1次1巻線） | None | 正同定 |
| transformer-1p-2s | センタータップ付きトランス | None | 正同定 |
| rl-voltage-current | RL ハイパスフィルタ | filter_passive | **誤同定**（RL V-I 回路） |
| simulation-diode-characteristics | ダイオード特性測定回路 | None | 正同定 |
| simulation-npn-characteristics | NPN トランジスタ特性測定回路 | None | 正同定 |
| simulation-dc-operational-point | 分圧回路（電圧ディバイダ） | None | 人手判定要 |
| simulation-dc-sweep | 分圧回路（電圧ディバイダ） | None | 人手判定要 |

---

## 3. 解釈

### 3.1 各指標の読み方

**in-scope 認識（strict −7.1pt）**
点推定では提案手法が優位だが、n=28 の小標本では CI が重複し統計的有意差の主張は難しい。失敗パターンは (a) 多段・亜種回路を「unknown」に逃がす過棄却、(b) 近縁候補（rlc_notch vs rlc_bandpass 等）の混同、の 2 種。

**out-scope 棄却（+5.0pt — 予想と逆転）**
Claude の棄却率 95.0% が提案手法 90.0% を上回った。設計書 §8 の予想に反する。
ただし「1件の誤受理」（actuator-npn-switch-led → ce_amp_npn_001）が示すように、棄却は確率的挙動であり決定的保証ではない。
提案手法の棄却は閾値 θ の機械規則であり、**根拠付き・100% 再現可能**という性質は維持される。

**再現性（−6.2pt）**
3試行での多数決不成立は 0 件だが、無効票（JSON 形式不能）が 2.1% 存在した。提案手法は決定論的構成のため再現性は構成上 100%。

**P2 自由記述（out 棄却率 5%）**
閉集合を与えない状況では Claude は「不明」をほぼ返さず、発振器・レーザードライバ・トランス等を正確に命名する。「直読みは回路を記述できるが、検証可能なラベル空間と決定論的棄却には載らない」という論旨の根拠として使用できる。

### 3.2 設計書 §8 撤退条件との照合

> 「直読みが全指標で同等以上なら、主張軸を説明可能性・監査可能性に切り替える」

| 指標 | 判定 |
|---|---|
| in-scope strict | 直読み **劣位** (78.6% < 85.7%) |
| out-scope 棄却 | 直読み **優位** (95.0% > 90.0%) — 逆転 |
| 再現性 | 直読み **劣位** (93.8% < 100%) |

**→ 全指標で優位でないため撤退条件は不成立。主張軸の切り替えは不要。**

out-scope 逆転は論文に誠実に記載する。推奨文例：

> *直読みの out-of-scope 棄却率（95%）は検索手法（90%）を上回るが、1件の誤受理が示すように確率的傾向にとどまる。提案手法は閾値規則による 100% 再現可能な棄却と IR 根拠の監査可能性を提供する点で異なる。*

---

## 4. 今後の対応

| 項目 | 内容 |
|---|---|
| Gemini 代替 | API キー (`GEMINI_API_KEY`) または `LLMClient(provider="gemini")` で再実行 |
| P2 人手判定 | 上表「人手判定要」4件（generic_opamp_bip / rectifier-4d-opamp-npn / simulation-dc-* 2件）を確認 |
| 表1 記入 | 本レポートの P1 数値（条件①）を論文 §3 表1 の条件①欄に転記 |

---

## 5. 実行ログ

| 日時 | 内容 |
|---|---|
| 2026-07-05 | 初回 P1 実行（144件）— `--system-prompt` 引数の CP932 文字化けで全件無効。ファイル削除 |
| 2026-07-05 | `llm_client.py` 修正: `--system-prompt` → `--system-prompt-file`（tempfile UTF-8） |
| 2026-07-05 | `score_baseline_direct.py` 修正: en-dash CP932 エラー修正、複数ファイルマージ対応 |
| 2026-07-05 | `run_baseline_direct.py` 修正: `--only` カンマ区切り複数対応、`--output` オプション追加 |
| 2026-07-05 | P1 再実行（144件）— 8クエリに transient CLI エラー（全件 None 票） |
| 2026-07-05 | `baseline_p1_claude_retry.jsonl` に 8クエリ 24件再実行、P1 マージ採点で確定 |
| 2026-07-05 | P2 実行（48件）完了、キーワード写像採点 |

---

## 6. データファイル

| ファイル | 内容 | 件数 |
|---|---|---|
| `results/baseline_p1_claude.jsonl` | P1 Claude 本番（初回） | 144件 |
| `results/baseline_p1_claude_retry.jsonl` | P1 エラー 8クエリ再試行 | 24件 |
| `results/baseline_p2_claude.jsonl` | P2 Claude 自由記述 | 48件 |
