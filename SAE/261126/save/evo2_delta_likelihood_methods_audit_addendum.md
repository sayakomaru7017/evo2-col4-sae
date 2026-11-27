# Evo2 Δlog-likelihood 監査 追加確認（2026-11-27）

前回レポート（`evo2_delta_likelihood_methods_audit.md`）への追補。全てCPUのみ・既存ファイルの確認。
新規GPU計算・再スコアリング・既存ファイルの上書きは一切行っていない。

---

## ① 非SNVスコアリング: Evo2原著・公式GitHubとの比較、および400例の統計

### Evo2公式資料における非SNV記載の有無

Web上で確認可能な一次資料（bioRxivアブストラクト、公式GitHubリポジトリREADME、BRCA1 zero-shot VEPノートブック）を確認した。

- **BRCA1公式ノートブック**（`notebooks/brca1/brca1_zero_shot_vep.ipynb`、ローカルにも存在）: "we can predict whether a particular **single nucleotide variant (SNV)**" / "3,893 *BRCA1* **SNVs**" と明記。扱っているのは**SNVのみ**。
- mutant配列の作り方: `var_seq = ref_seq[:snv_pos] + alt + ref_seq[snv_pos+1:]` の後に **`assert len(var_seq) == len(ref_seq)`** を実行しており、これは「SNVなので長さが変わらないことを検証している」だけで、indelへの対応は一切想定されていない。
- 公式GitHub READMEのVEP関連記述は「Reference vs alternative allele normalization」という項目名のみで、具体的な正規化手法の説明はない。
- **bioRxiv論文本文（Methods/Supplementary Methods）は取得できなかった**（HTTP 429エラー、Nature版PDFも未取得）。したがって論文本文に非SNVの扱いが明記されているかどうかは**未確認**。

**結論**: 確認できた公式資料の範囲では、Evo2のzero-shot VEPはSNV限定で説明されており、indelのスコアリング方法（特に参照/変異配列の長さを揃えるか否か）についての公式な記載は**見つかりませんでした**。本プロジェクト（②XLAS解析）における「長さを揃えずmean log-likelihoodで比較する」という実装は、公式に明記された手法を踏襲したものではなく、**独自の実装判断**であることが改めて確認されました。

### 最終400例の統計（`手作業確認済み_Evo2score付与.xlsx`, "S作成all症例"シート）

| 項目 | 値 |
|---|---|
| 総行数 | 400 |
| SNV行数 | 335 |
| 非SNV行数 | 65 |
| ユニーク変異数（chr,pos,REF,ALT） | 273（SNV 216 / 非SNV 57） |
| ユニークPaper_patient_ID数 | 392（※同一症例が枝番付きID（例: "-1"）で複数行計上されている場合があり、厳密な「患者数」の上限値） |
| len(ALT)−len(REF) 分布（ユニーク変異） | 0(SNV):216, −1:27, −2:2, −3:2, −4:6, −5:2, −9:1, −11:1, −13:1, −14:1, −18:1, −21:1, +1:7, +2:1, +4:1, +5:2, +18:1 |
| Reference window長 | 全273件とも **8,192bp固定**（min=max=8192、境界クリップによる短縮は0件） |
| Mutant window長 | min=8,171bp（21bp欠失） 〜 max=8,210bp（18bp挿入） |
| 8,192bp未満のreference window | **0件**（COL4A5遺伝子領域から十分離れた位置の変異がないため、全例でwindow境界クリップは発生していない） |

---

## ② evo2_7b と evo2_7b_262k の違い、checkpoint revision

### 既存の262k再計算ファイル

`delta likelihoods/run_alport_evo2_262k_pb.py` の実行チェックポイント `score_checkpoint_262k_pb.pkl` は存在し、**2,750件のスコア（計算完了、next_idx=2,750で停止）**が記録されています。しかし、この結果をCSVにまとめた最終出力ファイル（`alport_evo2_262k_pb.csv`）は**ディスク上に存在しませんでした**。チェックポイントの生スコアは残っているため、ご希望であれば（新規GPU計算なしで）CSV化だけ行うことは可能です。VUS用の `run_alport_evo2_262k_vus.py` に対応するチェックポイントは見当たりませんでした。

### checkpoint/学習上の違い（`evo2/utils.py`の`CONFIG_MAP`、実ファイル`evo2/configs/*.yml`より）

| | evo2_7b（①②delta-likelihoodで使用） | evo2_7b_262k（SAE feature抽出で使用） |
|---|---|---|
| Configファイル | `configs/evo2-7b-1m.yml` | `configs/evo2-7b-262k.yml` |
| max_seqlen | **1,048,576**（約1M） | **262,144**（約262k） |
| rotary_emb_scaling_factor | **128** | **32** |
| inner_mlp_size | **11,264** | **11,008** |

`inner_mlp_size`（MLP内部次元）が異なる = **単なる推論時のコンテキスト長設定の違いではなく、パラメータ数が異なる別のcheckpoint（別の継続学習を経た別の重み）です。** つまりevo2_7bとevo2_7b_262kは「同じ重みをより長い文脈で使っているだけ」ではなく、**真に異なる学習済みモデル**です。

**重要な留意点**: ご依頼の「SAE featureの変異による活性変化とΔlog-likelihoodとの関連を検証する」際、SAE feature側はevo2_7b_262k、delta-likelihood側はevo2_7bという**異なるcheckpoint**の出力を比較することになります。両者の差分が本当に生物学的なシグナルの違いを反映しているのか、それとも2つの異なるモデル間の系統的な違い（学習データ・学習量・文脈長の違いに由来する一般的なスコアの系統誤差）を一部反映しているのか、原理的に区別できない点にご留意ください。

### Checkpointのrevision/hash（HuggingFaceキャッシュより実測）

| モデル/SAE | HuggingFace snapshotディレクトリ名（= revision/commit hash） |
|---|---|
| `arcinstitute/evo2_7b` | `bda0089f92582d5baabf0f22d9fc85f3588f6b58` |
| `arcinstitute/evo2_7b_262k` | `27c4f055e8b729a27ab5c3e6c8b9c35367a76fba` |
| `Goodfire/Evo-2-Layer-26-Mixed`（SAE） | `a02b08a876b112d1c5da172e57a59e2bc76b1d70` |

（これらは現在のローカルキャッシュのスナップショット名であり、HuggingFace Hubの一意なcommit/revisionハッシュと一致します。過去の実行時点でも同一キャッシュが参照されていたことは、環境自体が2026-06-29作成のまま変更されていないこと（④参照）から、強く支持されます。）

---

## ③ Reference genomeの一致確認（NCBI GRCh38.p14 vs Ensembl REST）

以前使用していたローカルのNCBI GRCh38.p14 gzip FASTA（`/home/azureuser/evo/alport/GRCh38.p14_chrX.fna.gz`）は**既にディスク上から削除されており、直接の再比較はできませんでした**。

代わりに、NCBI E-utilities（efetch, accession **NC_000023.11**, GRCh38.p14 Primary Assembly であることをレスポンスヘッダで確認）から、`col4a5_padded_sequence.pkl`と全く同じゲノム範囲（chrX:108,435,549-108,701,743、266,195bp）を独立に取得し、直接比較しました。

**結果: 266,195bp全域で完全に一致（1塩基の相違もなし）。**

Ensembl REST API由来の配列とNCBI GRCh38.p14の配列は、少なくともCOL4A5領域において完全に同一であることが確認できました。解析対象変異への影響は**ありません**。

---

## ④ 最終400例の再現性

### 提供ファイル

以下を `SAE/261126/save/` に保存し、GitHubにpushしました。

- `run_teshigyou_kakunin_evo2_score_AS_ACTUALLY_EXECUTED.py`（下記の重要な注記を参照）
- `teshigyou_kakunin_score_checkpoint.pkl`（実行時の生スコア、258 ref + 273 var = 531件）
- `手作業確認済み_Evo2score付与.xlsx`（最終結果ファイル）
- `ncbi_grch38_chrX_col4a5_region.pkl`（③の検証に使用したNCBI取得配列）

### 重要な注記: 実行済みコードとgit管理コードの不一致

監査の過程で発見した点です。**現在GitHubにコミットされている`XLAS/run_teshigyou_kakunin_evo2_score.py`（commit `81d9bf4`）は、実際にこの400例の結果を生成したコードではありません。** このファイルは`git log`で確認する限りコミットは1回のみで、その1回のコミットには既に「変異の正規化（左整列）」機能と、別のチェックポイントファイル名（`..._v2.pkl`）を使うよう編集された**後のバージョン**が含まれています。このv2版は提案はしたものの、ユーザーの明示的な指示（「再計算しなくていいです」）によって**一度も実行されておらず**、`_v2`付きのチェックポイントや結果ファイルはディスク上に存在しません。

実際にこの400例を生成した「編集前のオリジナル版」はgit履歴には残っていないため、本セッション自身の会話記録から再構成し、`run_teshigyou_kakunin_evo2_score_AS_ACTUALLY_EXECUTED.py`として保存しました。

### Excel ↔ Checkpoint 全件一致検証

上記の再構成スクリプトのロジックで258件のref window・273件のvar windowを再構築し、`teshigyou_kakunin_score_checkpoint.pkl`の生スコア（531件）をマッピングした結果、**400行全てについて、Excel記載のEvo2score列と、チェックポイントから再構成したΔlog-likelihoodが、浮動小数点の丸め誤差（最大差 5.2×10⁻¹⁸）の範囲で完全に一致**しました（不一致0件）。この一致は、再構成したスクリプトのロジックが実際に実行されたコードと機能的に同一であることの強い証拠です。

### 当時のライブラリバージョン・計算精度の同一性について

**直接のログによる証明はできません**でしたが、以下の状況証拠から、当時（2026年9月4-5日の実行時）と現在（本監査時点）で環境が変更されていないことが強く示唆されます。

- `evo2/models.py`, `evo2/scoring.py` の最終更新日時: **2026-06-29**
- Python仮想環境（`.venv`）およびパッケージ（`evo2==0.6.0`, `vtx==1.1.0`）のインストール記録日時: **2026-06-29**
- これらはいずれも、400例の実行日（9月4-5日）より前であり、かつ現在（11月27日）までタイムスタンプが更新されていません。つまり、**同一のPython環境・ライブラリバージョンが6月29日以降一貫して使われ続けている**と考えられます。

ただし、これは「環境のファイルが書き換えられていない」ことの間接証拠であり、実行時に明示的なバージョンログ（`pip freeze`出力など）が保存されていたわけではないため、**厳密な証明ではなく推定である旨を明記します**。計算精度（bf16フォールバック）についても同様に、同一ライブラリ・同一モデルロード経路であることから「当時も今回確認したのと同じbf16フォールバックが発生していたはず」と推定していますが、当時の実行ログ自体（警告メッセージを含む標準出力）の保存は確認できておらず、**推定である旨を明記します**。

---

## 未確認事項（追加分）

1. Evo2論文本文（Methods/Supplementary Methods）における非SNV scoringの記載有無（bioRxiv full-text取得失敗、Nature版PDF未取得）
2. `score_checkpoint_262k_pb.pkl`から最終CSVへの変換が行われなかった理由（中断か、単に後続処理が未実行か）
3. 2026年9月4-5日実行時点の`pip freeze`等の明示的な環境スナップショット記録
