---
name: run-eval
description: TopoRAG の評価スイート（evaluate / validate_real / reject_eval / ablation）の使い分け・実行コマンド・結果の解釈基準。「評価して」「精度を測って」「回帰確認」と言われたら使う。
---

# 評価スイートの使い分け

| コマンド | 測るもの | いつ使う |
|---|---|---|
| `python -m pytest -q` | ユニットテスト(特徴抽出・分解・検索・DB整合) | 全変更後（最速の回帰ガード） |
| `python evaluate.py` | 自己検索 Hit@1/@3/MRR・摂動不変・LOO閾値 | DB/特徴量を変えた後。hard failure で exit 1 |
| `python check_scope.py` | taxonomy と real_expected.yaml の整合 | コーパス/分類体系を変えた後 |
| `python validate_real.py` | 実機汎化 Hit@1/@3/MRR（strict/lenient 両方） | 検索アルゴリズムを変えた後 |
| `python reject_eval.py` | 棄却 AUC+95%CI・nested CV bacc・撤退ゲート | 棄却関連を変えた後 |
| `python ablation.py` | 特徴グループの寄与（ゼロ化で Hit@1 低下測定） | 新次元の価値を検証するとき |
| `python evaluate.py --llm` | LLM 段の判定精度（要 LLM_PROVIDER） | LLM プロンプト変更後のみ |

CI (.github/workflows/ci.yml) は pytest + check_scope + evaluate.py を Python 3.10/3.12 で実行する。
ローカルでの同等確認: `pytest -q && python check_scope.py && python evaluate.py`

## 解釈の規律（過去の失敗から事前登録済み）

- **点推定だけで判断しない**: 必ず 95%CI（Wilson / ブートストラップ）併記。CI が重なる「改善」は改善ではない
- **naive と nested を区別**: 同一データで閾値選択した bacc は楽観的。汎化主張は nested CV 値で行う
- **自己検索 100% は成果ではない**（特徴量を DB に対して手彫りした同語反復）。汎化の主張は実機コーパスの数値のみで行う
- **実機の必要水準**: 検索段の役割は「良い候補出し」なので Hit@3 / MRR を主指標にしてよい（top-1 断定は LLM 裁定層の仕事という設計方針）
- 撤退ゲート判定が RETREAT_TO_RANKER なら、二値棄却をやめてランカー（top-3+スコア提示）へ切替える（HANDOFF §7.3）

## 数値を変更したら

README の実測値表・circuit_rag.py の定数コメント（校正日・n・根拠）・docs/HANDOFF を同期する。
3箇所の数値が食い違った前科があるため、評価値を引用する場所は「実行ログからコピー」を徹底する。
