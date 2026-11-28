# XLAS 非SNV再スコアリング(left-alignment/parsimony正規化後、56変異) 研究記録

実行日: 2026-11-28（run01と同日の追加実行）

## 事前決定事項（計算結果を見る前の記録）

今回の変更理由は、**変異表記の標準化(left-alignment / parsimonyによる正規化)**であり、解析結果の改善を
目的としたものではない。ユーザーの指示により、以下を計算前の決定事項として明記する。

> **QC(①全56変異のREF一致・WT/mutant長8,192bp確認、②正規化前後で表記が一致する19変異についてrun01との
> 一致)を満たした場合、今回の正規化後のスコアを正式な主解析に採用する。**

この決定は計算結果を確認する前に下されたものであり、結果を見てから採用可否を判断したものではない。

## 背景

run01（2026-11-28、同日先行実行）で、非SNV 57ユニーク変異について「WT/mutant長を8,192bpに統一する」
方法論修正を実施した。その後、left-alignment/parsimonyによるゲノム変異表記の正規化を行った結果、
57件中1組（NSV002とNSV051）が同一の正規化キー（chrX:108584525, REF=TGTAA, ALT=T）に統合され、
56ユニーク変異（65症例、患者数は変わらず）となった。

本実行では、正規化後のPOS・REF・ALTをExcelに記載された通りそのまま使用し、独自の座標変更・再正規化は
一切行っていない。

## 計算条件（run01と同一）

| 項目 | 値 |
|---|---|
| モデル | evo2_7b |
| HFキャッシュrevision | bda0089f92582d5baabf0f22d9fc85f3588f6b58（run01と同一、cache_check()で検証） |
| 参照配列 | col4a5_padded_sequence.pkl（chrX:108,435,549-108,701,743）、sha256 `908c7f13...` がrun01と完全一致することを検証 |
| 入力方向 | ゲノム+鎖 |
| Window構築 | 変異を中心に8,192bp、WT/mutantとも常に8,192bpに統一（奇数差の余りは右側）— run01の`build()`関数と同一ロジック |
| 尤度算出 | mean、batch_size=1、prepend_bos=False、average_reverse_complement=False（シグネチャをコードで検証） |
| 環境 | evo2==0.6.0, vtx==1.1.0, HAS_TE=False（bf16フォールバック）、HF_HUB_OFFLINE=1 |
| GPU | NVIDIA A100 80GB PCIe |

## 実行結果

- 事前QC（CPUのみ、`prepare`サブコマンド）: **56/56件 REF一致、WT=MT=8,192bp 確認済み**（`QC.csv`）
- net_length_changeがExcel記載値と完全一致することも検証済み
- GPU計算: モデルロード80.4秒、1配列あたり約0.83秒、**総実行時間 約187秒**（35分の上限に対し大幅に余裕あり）
- **COMPLETE 56/56**、WARN/FAILなし

## 正規化前後で表記が一致する19変異の一致確認

正規化で表記が「不変」だった19変異について、run01のref_ll・var_ll・ΔLLとの絶対差を確認した
（NSV051はNSVN002と比較、指示通り）。

**19/19 全てPASS（許容差1e-5以下）。実際の最大絶対差は1e-16オーダーで、浮動小数点の丸め誤差の範囲内。**

詳細は `consistency_check_19.csv` を参照。

## QC判定

全てのQC基準（REF一致、window長統一、19変異の一致確認）を満たしたため、**本決定事項に基づき、今回の
正規化後スコア（`scores_COMPLETE.csv`、56変異）を正式な主解析として採用する。**

## 保存物（`左寄せ/prepared_56/`）

- `scores_COMPLETE.csv` — 56変異のID、座標、REF、ALT、ref_ll、var_ll、Evo2score(ΔLL)、window情報、配列sha256
- `QC.csv`, `windows.csv` — 事前CPU検証結果
- `consistency_check_19.csv` — 19変異のrun01との一致確認結果
- `runtime.json`, `pip_freeze.txt`, `nvidia_smi.txt` — 実行環境記録
- `run_log.log` — 実行ログ全文
- `run_nonSNV56_normalized_rescore.py` — 使用したPythonコード
- `input_sequences.fasta`, `prepared.json` — 構築済み入力配列・メタデータ

run01の結果（`../extracted/rescore_run01/`）は上書きしていない。SNV335症例・その他の解析は変更していない。
臨床データへの反映はMac側で実施する。
