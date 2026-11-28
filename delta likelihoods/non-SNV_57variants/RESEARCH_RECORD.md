# XLAS 非SNV再スコアリング 研究記録（2026-11-28, A100実行）

## 背景・目的

2026年9月4-5日のXLAS臨床重症度解析（400症例）では、非SNV（deletion/insertion/duplication/delins）にも
SNVと同じwindow構築方法（`build_window`、VCFアンカー中心の固定参照窓から単純置換）を適用していたため、
mutant配列の長さがWT window（8,192bp）と一致していなかった（11月27日の監査レポートで判明）。

Evo2原著 Nature Supplementary Information A.3.12 では、indelスコアリング時に周辺の実ゲノム配列を追加・削除し、
WT/mutant両方の入力配列長を統一する旨が記載されている（著者実装の詳細は非公開）。この方法論上の相違を是正するため、
非SNV 57ユニーク変異（65症例）のみを対象に再スコアリングを実施した。**解析結果の改善を目的としたものではない。**

## 方法論の変更点

- 従来: `mt = ref_window[:rel] + alt + ref_window[rel+len(ref):]`（長さ = 8192 + (len(alt)-len(ref))、WTと不一致）
- 今回: WTの参照窓（VCFアンカー中心、8192bp）は変更せず、mutant側のみ周辺フランクを左右均等に加減して8192bpに統一。
  奇数差の余り1塩基は**右側**で調整（原著に詳細な指定がないため、本研究で定めた統一規則であり、著者実装と同一と断定しない）。
- WT window、入力方向（ゲノム+鎖）、尤度算出方法（mean log-likelihood、BOSなし、batch_size=1、
  average_reverse_complement=False）、モデル（evo2_7b、1Mコンテキストconfig）は一切変更していない。
- SNV 335症例は対象外、変更なし。

## 実行環境

- モデル: `evo2_7b`（HuggingFaceキャッシュrevision `bda0089f92582d5baabf0f22d9fc85f3588f6b58`、
  完全な重みファイルがスナップショット内に存在することを確認、SHA256暗号学的検証は不要なケース）
- ライブラリ: evo2==0.6.0, vtx==1.1.0（詳細は`pip_freeze.txt`）
- GPU: NVIDIA A100 80GB PCIe（`nvidia_smi.txt`）
- torch: 2.7.1+cu128
- HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1（ダウンロード不可の状態で実行）
- HAS_TE=False（bf16フォールバック、従来と同一）

## 実行結果サマリー

| 項目 | 結果 |
|---|---|
| モデルロード時間 | 80.5秒 |
| 1配列あたり処理時間 | 約0.81-0.83秒（8192bp） |
| 総実行時間 | 約193秒（3分強、35分のソフト上限に対し大幅に余裕あり） |
| ①WT reference全57件チェック | **PASS 57/57**（旧ref_llとの最大絶対差 4.92×10⁻⁸、許容差1e-5以内） |
| ②旧方式mutant対照3変異（NSV017: -21bp欠失、NSV026: -1bp欠失、NSV015: +18bp挿入） | **PASS 3/3**（var_ll・ΔLLとも許容差1e-5以内、全て差ゼロ〜1e-8オーダー） |
| ③新方式による57変異再スコアリング | **COMPLETE 57/57** |
| `--allow-numeric-drift` | 不使用（指示通り） |
| WARN/FAIL | **0件** |

全てのQC（①②）が許容差内でPASSしたため、③の新方式スコアリングに進み、57件全てが正常に完了した。
途中停止・時間切れ・異常終了は一切発生しなかった。

## 保存物

`extracted/rescore_run01/` 配下（原本 `extracted/rescore_20261128_final/` は未使用のまま保持）:

- `runtime.json` — 実行環境・キャッシュ・ライブラリ・ハッシュ値の記録
- `pip_freeze.txt`, `nvidia_smi.txt`
- `reference_check.csv` — 57件のWT reference再現性チェック結果
- `old_mutant_controls.csv` — 旧方式mutant対照3変異のチェック結果
- `scores_PARTIAL.csv` / `scores_COMPLETE.csv` — 新方式による57変異の新旧スコア対応表
- `rescore_run01.log` （同ディレクトリの親、`extracted/`直下）— 実行の標準出力ログ全文
- `QC.csv`, `windows.csv`, `prepared.json`, `input_sequences.fasta` — CPU側で事前準備済みのQC・window記録（持ち込み）

新旧スコアの評価・解釈・統計解析への反映は、ご指示の通りMac側で実施する。本セッションでは変異の除外や
解析条件の判断は一切行っていない。

## 原著一次資料による検証（2026-11-28 追記）

Evo2原著 Nature Supplementary Information（DOI: 10.1038/s41586-026-10176-5）A.3.12 "Zero-shot variant
scoring" を直接確認し、以下の記載を確認した（原文引用）：

> "To score a variant with Evo 2, we take a genomic window of length 8,192 around the variant and
> calculate the log-likelihood of the variant sequence normalized by the log-likelihood of the reference
> sequence at the same position. For non-SNV variants, we also normalized to the reference sequence,
> and for DNA models to keep the change in likelihood invariant to length we maintain the total length
> of sequence scored to be the same regardless of the length of the variant by centering on the insertion
> or deletion and adding or removing context nucleotides on the edges of the window."

A.3.2（原核生物変異実験）にも同趣旨の記載がある：

> "For the deletion variants, the 8,192 bp window was extended into the neighboring sequences to match
> the length of the wildtype sequence (8,192 bp) to avoid any biases in the likelihoods that arise from
> differences in sequence lengths."

**結論**: 今回の再スコアリングで実装した「変異を中心に8,192bp windowを取り、window端で文脈塩基を加減して
WT・mutant双方を常に8,192bpに統一する」という方法は、原著A.3.12・A.3.2の記載と一致している。原著に
明記されていないのは「奇数の長さ差の余り1塩基を左右どちらに割り振るか」という実装の細部のみであり、
本研究ではこれを「右側」と定めた（上記の通り）。この点を除き、本手法は原著のzero-shot variant scoring
プロトコルに忠実である。

## 今後のEvo2解析における注意事項（研究記録として明記）

**非SNVのzero-shot variant scoringでは、Evo2原著 Nature Supplementary Information A.3.12 に従い、
WTとmutantの入力配列長を統一すること。** 本プロジェクトの初期実装（2026年9月4-5日、400症例の`手作業確認済み_Evo2score付与.xlsx`
および`XLAS/xlas_cox_evo2_7b_delta.csv`等）はこのルールに従っておらず、非SNVのmutant windowはindelサイズ分だけ
WTと異なる長さでスコアリングされていた。今回（2026-11-28）、57ユニーク非SNV変異についてのみ、長さを統一した
新方式での再スコアリングを実施し、その結果（`scores_COMPLETE.csv`）を新旧スコアの比較・研究記録として保存した。
