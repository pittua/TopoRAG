---
name: toporag-map
description: TopoRAG のファイル構成・データフロー・主要定数の要約リファレンス。コードや README を読み直す前にまずこれを参照する（トークン節約用）。構造の質問・改修の起点・影響範囲の確認に使う。
---

# TopoRAG プロジェクトマップ

目的: LLM に回路認識能力を与える（分類器は手段）。知覚層(決定的IR) → 構造検索 → LLM 合成の3層構想。

## データフロー

```
ネットリスト(JSON) → block_decomposer(T分岐分割, 受動のみ)
                   → feature_extractor(43次元特徴 + WLヒストグラム)
                   → circuit_rag.search(コサイン×beta + WL×(1-beta))
                   → LLM整形(llm_client) / 棄却(search_with_rejection)
```

## モジュール（1行要約）

| ファイル | 役割 |
|---|---|
| feature_extractor.py | グラフ化(build_graph: 2端子=辺/3端子以上=__dev_ノード)・A/B1/B2/B3/C/D特徴抽出。__main__ で features_db.json 生成 |
| topo_kernel.py | WL サブツリーカーネル(3反復)。wl_histogram / wl_kernel |
| block_decomposer.py | 主直列パス上のGND接続節点でブロック分割。能動素子回路は分割しない |
| circuit_rag.py | vectorize(43次元)・search(階層対応+Hungarian割当)・search_with_rejection・プロンプト生成 |
| llm_client.py | LLMClient(SDK) / CLILLMClient(subprocess) / MockLLMClient |
| circuit_simulator.py | PySpice/ngspice AC解析(受動+部品値ありのみ。能動/SW/Dはスキップ) |
| kicad_sch_to_toporag.py | .kicad_sch → TopoRAG形式変換(kicad-cli経由) |
| evaluate.py | 自己検索/alpha/摂動/LOO閾値/LLM/sim 評価。hard failure で exit 1(CIが実行) |
| validate_real.py | 実機コーパスの Hit@1/@3/MRR(strict/lenient) |
| reject_eval.py | 棄却シグナル評価(AUC+CI・nested CV・撤退ゲート §7.3) |
| ablation.py | 特徴グループ寄与分析 |
| check_scope.py | scope_taxonomy.yaml と real_expected.yaml の整合検査(CIが実行) |

## データファイル

- sample_netlists.json … DB の原本(回路47件〜)。**編集はこちら**
- features_db.json … 生成物。`python feature_extractor.py` で再生成。**手編集禁止・原本と同時コミット**
- real_corpus.json / real_expected.yaml … 実機評価コーパス(タグ無し)とラベル(scope/family/expect)
- scope_taxonomy.yaml … DB非依存の in-scope 定義(7 family)
- query_netlists.json … circuit_rag.py 実行時の入力

## 主要定数（circuit_rag.py）

- `DEFAULT_ALPHA = 1.0` … トポロジーのみ(タグ不使用が既定)
- `DEFAULT_BETA = 0.95` … コサイン0.95 + WL 0.05
- `CALIBRATED_REJECT_THRESHOLD = 0.8863` … 実機 in15/out20 で校正(TNR 0.90/TPR 0.75)

## 注意点

- 循環import は解消済み（c0b9ea8）。経路探索には爆発ガードあり
- ベクトル次元を増やすと全コサイン類似度が動き棄却閾値の再校正が必要
- 端子命名は意味を持つ: anode/cathode, base/collector/emitter, gate/drain/source, in_p/in_n/out
- 開発方針: main 直接コミット（個人開発）。検証は `pytest -q && python check_scope.py && python evaluate.py`（CI と同じ）
