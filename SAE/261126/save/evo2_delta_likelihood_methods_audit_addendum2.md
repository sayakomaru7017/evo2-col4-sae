# Evo2 Δlog-likelihood 監査 追加確認2（2026-11-27）

前々回（`evo2_delta_likelihood_methods_audit.md`）・前回（`..._addendum.md`）への続報。
全てCPUのみ・既存ファイル/既存チェックポイントの確認。新規GPU計算・再スコアリングは行っていない。

---

## 1. 訂正: Evo2原著 Methods 4.3.12（Zero-shot variant scoring）における非SNV扱い

ご指摘の通り、Evo2原著preprint（bioRxiv, doi: 10.1101/2025.02.18.638918）Methods **4.3.12 Zero-shot variant scoring** に、indelスコアリングでは**window端の周辺塩基を追加・削除し、referenceとmutantの入力配列長を同一に揃える**旨の記載があるとのことです（前回の監査では本文を取得できず「未確認」としていましたが、ユーザーにより本文の該当箇所が確認されました）。

**これは本プロジェクトの②XLAS実装とは異なります。** 本プロジェクトの`build_windows`/`build_window`関数（`run_xlas_cox_evo2_delta.py`ほか）は、indelサイズの分だけmutant windowの長さがreference windowと異なったまま（長さ調整なし）スコアリングしています（前回監査の5.节、具体例の表を参照）。

**訂正後の結論**: 本プロジェクトの②非SNVスコアリング手法は、Evo2原著が明示的に記載している公式手法（window端でのトリミング／パディングによる長さ統一）を**踏襲していません**。これは論文化にあたり重要な相違点であり、Methods/Limitationsで明示的に記載する必要があります。具体的には、原著の手法では「reference windowとmutant windowは常に同一の8,192bpで比較される」のに対し、本プロジェクトでは「mutant windowの長さはindelサイズに応じて8,192±数十bp変動する」という差があります。

（原文の正確な引用表現・ページ内の具体的な実装詳細は、ユーザーが確認されたMethods 4.3.12の原文をそのままご提供いただければ、より精緻な引用として監査レポート本体に反映いたします。）

---

## 2. 262kチェックポイントのCSV化

`score_checkpoint_262k_pb.pkl`を使い、`run_alport_evo2_262k_pb.py`と全く同じウィンドウ構築ロジック（CPUのみ、再スコアリングなし）で変異情報と対応づけました。

### 件数確認の結果: **チェックポイントは未完了でした**

| 項目 | 件数 |
|---|---|
| 対象変異（1,996 P/B ClinVar SNV） | 1,996（全件REF一致、flag 0件） |
| ユニークreference window数 | 1,798 |
| 必要なスコア数（ref + var） | 1,798 + 1,996 = **3,794** |
| チェックポイントに実際にあるスコア数 | **2,750** |
| 内訳 | reference window: 1,798/1,798（**完了**）、mutant window: 952/1,996（**未完了**） |

つまり262k版は、**全ての参照配列のスコアリングは完了していたものの、変異配列は1,996件中952件（約48%）のみ**でスコアリングが止まった状態でした（`next_idx=2750`で中断）。新規GPU計算はご指示通り行っていないため、**この952件分のみ**Δlog-likelihoodを計算し、残り1,044件は`evo2_delta_score`列を空欄、`computation_status`列に`ref_only_var_missing`と明記したCSVを作成しました。

- 出力: `delta likelihoods/alport_evo2_262k_pb_PARTIAL.csv`（ファイル名に"PARTIAL"を明記）
- 列: 元の変異情報列 + `ref_ll`, `var_ll`, `evo2_delta_score`, `flag`, `computation_status`（`complete`/`ref_only_var_missing`/`ref_mismatch_excluded`/`not_attempted`）
- `computation_status`内訳: `complete`=952, `ref_only_var_missing`=1,044

**本ファイルは不完全なデータセットです。** 262k版での全1,996件のΔlog-likelihoodが必要な場合は、残り1,044件のmutant window分の新規スコアリングが必要になります（本監査の方針により今回は実施していません）。

---

## 3. Paper_patient_ID が392ユニークとなった理由

400行・8組の重複ID（16行）を個別に確認した結果、**2種類の異なる原因が混在**していることが判明しました。

### (A) 論文間での偶然のID重複（5組・10行） — 実質的な問題ではない

| Paper_patient_ID | 論文②での変異 | 論文⑥での変異 | 同一患者か |
|---|---|---|---|
| A126 | c.609+875G>T (chrX:108576847) | c.G142T (chrX:108559064) | **別患者**（別論文がたまたま同じ"A"+番号の命名規則を使用） |
| A329 | c.1424-367C>T | c.1779+3G>T | 別患者 |
| A365 | c.3604+1G>A | c.A4928G | 別患者 |
| A422 | c.548dupG | c.T4605A | 別患者 |
| A424 | c.385-1G>A | c.G4279A | 別患者 |

いずれも論文②と論文⑥それぞれが独立に"A"+数字という患者ID命名規則を使っているための**偶然の文字列衝突**で、変異座標・REF/ALTが完全に異なります。**実際には別々の患者であり、データの重複登録ではありません。**

### (B) 同一論文内での同一患者・同一変異の重複（3組・6行） — 要確認

| Paper_patient_ID | 論文 | 変異（2行とも同一） | Analysis_ID |
|---|---|---|---|
| 8 Mother's cousin | ⑨ | c.3419G>T (chrX:108665552) | ⑨-12, ⑨-13 |
| 9 Cousin | ⑨ | c.3427G>A (chrX:108665560) | ⑨-15, ⑨-16 |
| 10 Mother's cousin | ⑨ | c.3427G>A (chrX:108665560) | ⑨-20, ⑨-21 |

この3組は全て**同一の論文⑨由来**で、**同一患者ラベル・同一変異・同一座標が2行ずつ重複して登録**されています。論文⑨の原著データ転記段階で、同じ症例が（例えば別の表・別の家系図記載から）2回カウントされた可能性があります。**これは単なるID命名規則の偶然の一致ではなく、真の重複エントリである可能性が高い**ため、解析対象から除外すべきかどうかをご確認いただくことを推奨します（除外の判断は臨床的な原著確認が必要なため、本監査では行っていません）。

**結論**: 400行→392ユニークIDの差（8件）のうち、**5件は無害な論文間ID衝突、3件（6行）は論文⑨内での真の重複候補**です。後者についてはAnalysis_ID ⑨-12/⑨-13、⑨-15/⑨-16、⑨-20/⑨-21の計6行が、統計解析上カウントが二重になっていないかご確認ください。

---

## 4. 保存したファイル

`SAE/261126/save/` に以下を追加保存しました。

- `evo2_delta_likelihood_methods_audit_addendum2.md`（本ファイル）
- `delta likelihoods/alport_evo2_262k_pb_PARTIAL.csv`（同ディレクトリに保存、未完了である旨を明記）
- `evo2_scoring.py`, `evo2_models.py`（スコアリング関数本体のコピー、再現性確保のため）
- `col4a5_padded_sequence.pkl`（COL4A5 reference配列、Ensembl由来）
- `col4a34_padded_sequence.pkl`（COL4A3/4 reference配列、Ensembl由来）

（`run_alport_evo2_262k_pb.py`・`score_checkpoint_262k_pb.pkl`は前回のaddendumで保存済みです。）
