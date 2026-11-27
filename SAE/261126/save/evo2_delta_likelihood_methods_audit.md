# Evo2 zero-shot Δlog-likelihood 算出方法の監査レポート

作成日: 2026-11-27
対象: ①ClinVar COL4A3/4/5 病的性識別解析（1,996 SNV）、②XLAS臨床重症度解析（SNV + 非SNV）
方針: 推測を排し、実際のコード・ライブラリ実装・実行ログ・既存出力ファイルのみに基づく。
確認できなかった項目は「未確認」と明記する。新規GPU計算・再スコアリング・既存コード/結果の変更は一切行っていない。

対象コード（実ファイル、リポジトリ内）:
- ①: `XLAS/run_alport_delta.py`
- ②: `XLAS/run_xlas_cox_evo2_delta.py`, `XLAS/run_xlas_565_new_variants_delta.py`,
     `XLAS/run_extra_variants_delta.py`, `XLAS/run_teshigyou_kakunin_evo2_score.py`
- ライブラリ本体: `evo2/models.py`, `evo2/scoring.py`

---

## 1. 使用したコード

### ① ClinVar COL4A3/4/5（1,996 SNV）

- スクリプト: `XLAS/run_alport_delta.py`（全文はリポジトリ参照。要旨を2.以降に記載）
- 入力: `/home/azureuser/evo/alport/B or P_COL4A345_all_clinvar_result.xlsx`
  （シート "P group(2159)" + "B group(462)"）、
  `/home/azureuser/evo/alport/GRCh38.p14_chr2.fna.gz`,
  `/home/azureuser/evo/alport/GRCh38.p14_chrX.fna.gz`
- 出力: `/home/azureuser/evo/alport/alport_evo2_delta.csv`
  （プロジェクト内では `XLAS/臨床重症度/alport_evo2_delta.csv` 等として複製利用）
- 実行コマンド: `python XLAS/run_alport_delta.py`（docstringに明記、実行ログの保存は未確認＝スクリプト内にログファイル出力の記述なし）
- 補助関数: `evo2.models.Evo2.score_sequences` → `evo2.scoring.score_sequences` / `_score_sequences` / `logits_to_logprobs` / `prepare_batch`

### ② XLAS臨床重症度解析

複数のスクリプトが時系列で存在し、いずれも同一の中核ロジック（`build_windows`/`build_window`、Evo2呼び出し）を再利用・複製している。

| スクリプト | 入力 | 出力 | 対象 |
|---|---|---|---|
| `XLAS/run_xlas_cox_evo2_delta.py` | `組入症例＿XLAS_strict_analysis_population_v2_原著再監査_GRCh38_最終.xlsx`（"Cox解析"シート） | `XLAS/xlas_cox_evo2_7b_delta.csv` | 436変異 |
| `XLAS/run_xlas_565_new_variants_delta.py` | `XLAS_COL4A5_565例_GRCh38再検証_Evo2用.xlsx` + 旧スコア流用 | `XLAS/xlas_565_full_evo2_7b_delta.csv` | 565行（新規+旧流用） |
| `XLAS/run_teshigyou_kakunin_evo2_score.py` | `臨床重症度/手作業確認済み.xlsx` | `臨床重症度/手作業確認済み_Evo2score付与.xlsx` | 400行（273ユニーク変異、全て新規計算） |

補助関数・参照データ:
- `XLAS/col4a5_padded_sequence.pkl`（Ensembl REST APIで取得したCOL4A5領域のGRCh38 +鎖配列、`run_col4a5_evo2_vep.py`内で生成。`assert gene_meta["strand"] == "+"`を含む）
- 実行ログ: チェックポイントpickle（例: `XLAS/xlas_cox_score_checkpoint.pkl`, `XLAS/teshigyou_kakunin_score_checkpoint.pkl`）に生のref_ll/var_llが保存されている。標準出力の完全なログファイルとしての保存は確認できたもの（本セッションで実行した`run_teshigyou_kakunin_evo2_score.py`等）を除き、古い実行については**未確認**。

---

## 2. Evo2モデルおよび実行環境

| 項目 | ①（ClinVar） | ②（XLAS） |
|---|---|---|
| モデル名 | `evo2_7b`（`Evo2("evo2_7b")`、run_alport_delta.py:106） | `evo2_7b`（全スクリプトで`MODEL_NAME="evo2_7b"`と明示） |
| Checkpoint | HuggingFace `arcinstitute/evo2_7b`（コード上は名前指定のみ、具体的なcommit hash/リビジョンは`evo2.models.Evo2.__init__`が`CONFIG_MAP`経由で解決。**具体的なHF revision番号は未確認**） |同左|
| Evo2ライブラリ | pip show結果: `evo2 0.6.0` | 同左（同一環境） |
| vortex (StripedHyena実装) | `vtx 1.1.0` | 同左 |
| Git commit (evo2リポジトリ) | `evo2/models.py`, `evo2/scoring.py`の最終変更コミット: `53f1959`（"vortex opt-in triton kernels + gene completion eval (#225)"） | 同左（同一チェックアウト） |
| Reference genome | NCBI **GRCh38.p14**（ローカルgzip FASTA、chr2/chrX全長を直接ダウンロード） | Ensembl REST API経由で取得したGRCh38配列（Ensemblの具体的なリリース番号・patchレベルは**未確認**。NCBI GRCh38.p14と塩基レベルで完全に同一かは**未検証**） |
| 計算精度 | Transformer EngineがインストールされていないためFP8は不使用、**bf16へのフォールバック**（`evo2/models.py`のロード時警告: "Falling back to bf16 projections"で確認。SAE側の計算ログでも同一警告を確認済み） | 同左（同一ライブラリ・同一モデルロード経路） |
| use_kernels | `Evo2.__init__`のデフォルト`use_kernels=False`（①②とも明示的な上書きなし） | 同左 |
| Batch size | `model.score_sequences(seqs, batch_size=1)`（明示） | `model.score_sequences(chunk)`（batch_size未指定 → `Evo2.score_sequences`のデフォルト値`batch_size=1`が適用、結果的に①と同一） |

**①②で同一の設定か**: はい、モデル・精度・batch size・ライブラリバージョンは同一です。唯一の相違点は**reference genomeの取得経路**（①: NCBI GRCh38.p14ローカルFASTA、②: Ensembl REST API）で、塩基配列が完全に一致するかどうかは本監査では未検証です（通常は一致しますが、patchレベルの差異が理論上あり得ます）。

**補足（評価されていないが実在する別経路）**: `delta likelihoods/run_alport_evo2_262k_pb.py` および `run_alport_evo2_262k_vus.py` というスクリプトが別途存在し、同じClinVar P/B変異セットを**evo2_7b_262k**で再計算しています（8192bp window・mean・delta=var-refは同一）。ただしこれは①の主解析（`alport_evo2_delta.csv`）とは別の補助的な検証用スクリプトであり、本報告の①②の主要な結果には使われていません（論文中でこの262k再計算結果を引用する場合は別途明記が必要です）。

---

## 3. 入力配列の作成方法

| 項目 | ①（ClinVar） | ②（XLAS） |
|---|---|---|
| Window size | 8,192 bp | 8,192 bp（全スクリプトで`WINDOW_SIZE=8192`） |
| 変異位置の配置 | window中央（`half = WINDOW//2`, `start = pos0-half`）。配列端に近い場合は`max(0,...)`/`min(len(seq),...)`でクリップされ、変異位置が中央からずれることがある | 同様（`s=max(0,local-WINDOW//2)`, `e=min(len(ref_master),local+WINDOW//2)`） |
| 上流・下流配列長 | 原則各4,096bp（端でクリップされた場合は非対称） | 同左 |
| Reference配列の作り方 | GRCh38.p14染色体全長FASTAから直接スライス | Ensembl取得済みの「padding付きマスター配列」（遺伝子境界で配列が尽きないよう余分に取得した固定配列）からスライス |
| Mutant配列の作り方 | `ref_w[:snv_pos] + alt + ref_w[snv_pos+1:]`（VCFアンカー規約、REF/ALTの置換） | `ref_window[:rel] + alt + ref_window[rel+len(ref):]`（同じVCFアンカー規約。SNVに限らずREF/ALTの長さが異なっても同一ロジックを適用。詳細は5.） |
| ゲノム座標の扱い | **0-based**（ClinVarの"Canonical SPDI"由来、`pos0`列名も0-basedを明示）。SPDI形式を直接使用 | **1-based**（Excelシートの`GRCh38_pos_VCF`列、VCF慣習に準拠） |
| REFとreference genomeの一致確認 | `if ref_w[snv_pos] != ref: return None, None, "ref_mismatch..."` で検査し、不一致行は除外（`bad`カウンタで集計、ログ出力） | `assert genome_ref == ref, f"ref mismatch..."`（不一致があれば**プログラムが即座にクラッシュ**する設計。①のような「除外して続行」ではない） |
| Forward/Reverse complement | **行っていない**。常に参照ゲノムのプラス鎖をそのままスコアリング（`Evo2.score_sequences`のデフォルト`average_reverse_complement=False`を上書きせず使用） | 同左（②はCOL4A5（+鎖）のみを扱う全解析のため、この論点はそもそも生じない） |
| COL4A4等マイナス鎖遺伝子の扱い | **revcompしていない**。chr2の生のプラス鎖配列をそのままスコアリングしており、COL4A4（マイナス鎖遺伝子）の変異も遺伝子の転写方向ではなく、ゲノムのプラス鎖方向のまま評価されている | ②はCOL4A5（Xq22.3、+鎖）のみが対象のため、マイナス鎖遺伝子の扱いという論点は**該当なし**（COL4A4はXLAS＝X連鎖性なので対象外。COL4A4を含むのはAlport症候群の常染色体劣性型であり、①のClinVar解析にのみ含まれる） |

**重要**: SAE feature抽出パイプライン（`SAE/run_col4_sae.py`）はマイナス鎖遺伝子を遺伝子方向にreverse complementしてから入力していますが、**delta-likelihoodスコアリング（①②とも）はこの処理を一切行っていません**。これはEvo2原著の零点ショットVEP手法（BRCA1 notebookを踏襲、run_alport_delta.py冒頭のdocstringに明記）と同じ考え方で、常に参照ゲノムのプラス鎖座標系でスコアリングする、という設計です。SAE解析とdelta-likelihood解析で鎖の扱いの流儀が異なる点は、論文中で明示的に区別して記載することを推奨します。

---

## 4. Log-likelihoodの算出方法

ソース: `evo2/scoring.py`（`_score_sequences`, `logits_to_logprobs`, `prepare_batch`）

```python
def logits_to_logprobs(logits, input_ids):
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    input_ids = input_ids[:, 1:]
    logprobs = torch.gather(softmax_logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    return logprobs
```

- **各塩基の条件付き確率の取得方法**: 位置iのlogit（モデルが位置iまでの文脈から計算した次トークン分布）から`log_softmax`を取り、位置i+1の実際のトークンIDで`gather`することで、標準的な自己回帰（次トークン予測）対数尤度 `log P(token_{i+1} | token_1..token_i)` を得ている。
- **合計か平均か**: `reduce_method='mean'`（`Evo2.score_sequences`のデフォルト、①②とも明示的に上書きしていない）。`reduce_func = np.mean`。
- **平均の分母**: **L-1**（Lは入力配列長）。理由: `logits_to_logprobs`で`[:, :-1]`により末尾1トークン分の次トークン予測機会が失われ、系列全体でL-1個の対数尤度しか得られない。最初の塩基（token 1）には先行文脈がないため、そもそも予測確率が存在せず評価対象に含まれない。
- **対数の底**: 自然対数（`torch.log_softmax`、底e）。
- **最初の塩基の扱い**: 評価されない（上記の通り、先行文脈がないため）。
- **BOS tokenの有無**: `prepend_bos=False`がデフォルトであり、①②とも明示的に`True`を指定していない。**BOS tokenは使用されていない**。
- **Forward/Reverse complement平均の有無**: `Evo2.score_sequences`の`average_reverse_complement`パラメータ（デフォルト`False`）。①②とも指定なし → **平均していない。常にforward（入力した通りの鎖）のみでスコアリング**。
- **変異位置のみかwindow全体か**: window全体（8,192bp相当）の平均対数尤度。変異位置1点のみの条件付き確率ではない。
- **Δlog-likelihoodの数式**:

```
ref_ll = mean_{i=2}^{L_ref}  log P(x_i | x_1..x_{i-1})   （参照配列、長さL_ref）
var_ll = mean_{i=2}^{L_var}  log P(y_i | y_1..y_{i-1})   （変異配列、長さL_var）
Δlog-likelihood = var_ll − ref_ll
```

  （①②とも共通して`delta = var_scores - ref_scores[...]`という引き算のみ。正規化係数や重み付けは一切ない。）

`score_sequences`関数定義とデフォルト引数（実際に使用されたバージョン、`evo2/scoring.py`より全文）:

```python
def score_sequences(
        seqs: List[str],
        model: StripedHyena,
        tokenizer: object,
        batch_size: int = None,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        device: str = 'cuda:0',
) -> List[float]:
```

`Evo2.score_sequences`（`evo2/models.py`、実際に①②から呼ばれるラッパー）:

```python
def score_sequences(
        self,
        seqs: List[str],
        batch_size: int = 1,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        average_reverse_complement: bool = False,
) -> List[float]:
```

---

## 5. 非SNVのスコアリング方法（最重要）

ソース: `build_windows`（`run_xlas_cox_evo2_delta.py`）/ `build_window`（`run_teshigyou_kakunin_evo2_score.py`ほか、同一ロジック）

```python
def build_windows(ref_master, master_start, pos, ref, alt):
    local = pos - master_start
    s = max(0, local - WINDOW_SIZE // 2)
    e = min(len(ref_master), local + WINDOW_SIZE // 2)
    ref_window = ref_master[s:e]
    rel = local - s
    genome_ref = ref_window[rel:rel + len(ref)]
    assert genome_ref == ref
    var_window = ref_window[:rel] + alt + ref_window[rel + len(ref):]
    return ref_window, var_window
```

- **SNVと非SNVに同一の関数を使用したか**: **はい、完全に同一の関数・同一のコードパスです。** REF/ALTの長さに関する分岐は一切なく、deletion・insertion・duplication・delins・multi-nucleotide substitutionすべてがこの1つの関数で処理されています。
- **Reference配列とmutant配列の入力長は同一か**: **いいえ、同一ではありません。** `ref_window`はwindow境界（またはs/eのクリップ）で決まる固定長（通常8,192bp）です。`var_window`は`ref_window`から`len(ref)`塩基を取り除いて`alt`（`len(alt)`塩基）を挿入するだけなので、**`var_window`の長さ = `ref_window`の長さ + (len(alt) − len(ref))** となり、indelのサイズ分だけ参照配列と変異配列の長さが異なります。
  - 例: 8,192bpのreferenceで1塩基欠失（REF 5bp→ALT 1bp、正味4bp欠失）の場合、mutant windowは**8,188bp**（8,192−4）のままスコアリングされます。
- **配列長の調整（パディング等）を行ったか**: **行っていません。** 上流・下流どちらの方向にも、欠失/挿入分を補うための追加配列の付与や、超過分の切り詰めは一切行われていません。長さの違いはそのままモデルに入力されます。
- **Insertionで余分な塩基を切り詰めたか**: 行っていません（上記の通り、長さの差はそのまま）。
- **配列長の異なるReferenceとmutantの比較における正規化方法**: `reduce_method='mean'`による位置あたり平均対数尤度を用いることで、長さの違いをある程度緩和していますが、これは「系列ごとの平均を比較する」という一般的な正規化であり、**indelに特化した追加の正規化（例えば長さ差分を考慮した補正）は行われていません**。また4.で述べた通り、両者の分母はそれぞれ独立に(L_ref−1)と(L_var−1)であり、同一の分母で比較しているわけではない点に注意が必要です。
- **VCF形式のREF/ALTに含まれるアンカー塩基の扱い**: 標準的なVCFアンカー規約（`pos`はREFの先頭塩基の位置、ALTはREF全体を置換）に従って正しく実装されています（`assert genome_ref == ref`による検証付き）。
- **変異サイズによる除外条件**: コードレベルでの自動的なサイズ上限/下限フィルタは**見当たりません**。処理対象は上流のExcelシートで`Evo2_eligible == "Yes"`とマークされた行に限定されますが、このフラグを立てる基準（サイズ上限の有無を含む）はスプレッドシート作成段階の判断であり、スクリプトのコードからは追跡できないため**未確認**です。

### 具体例（既存記録より、①②の実データから各1例）

いずれも②（XLAS COL4A5解析）の実データです。window = 8,192bp中心、`col4a5_padded_sequence.pkl`使用、座標は1-based・VCFアンカー規約。

| 変異型 | Analysis_ID | 座標(GRCh38) | REF | ALT | Ref window長 | Mutant window長 | 変異周辺配列(upstream15bp[REF]downstream15bp) | ref_ll | var_ll | Δlog-likelihood |
|---|---|---|---|---|---|---|---|---|---|---|
| SNV | ②-A7 | chrX:108,577,950 | A | T | 8192 | 8192 | `TTGTCTTCTCTTCTT`**[A]**`GGGCCCTCCTGGTCC` | -0.972737 | -0.976595 | -0.003858 |
| Deletion | ②-A371 | chrX:108,584,527 | TAAGT | T | 8192 | 8188 | `GACCTCCTGGACTTG`**[TAAGT]**`TTTTTTTTTTTAGTC` | -1.074194 | -1.076474 | -0.002280 |
| Duplication | ②-A121 | chrX:108,573,656 | T | TT | 8192 | 8193 | `CTCCTGGAATACAAG`**[T]**`AAGTATCCAGTGATT` | -1.087225 | -1.088156 | -0.000931 |
| Delins/MNV | ⑥-A359 | chrX:108,595,559 | CC | TAT | 8192 | 8193 | `ACTGGTATTTCAGGG`**[CC]**`TCCAGGTCAACCTGG` | -0.977524 | -0.986276 | -0.008752 |
| Insertion（非duplication型） | ⑥-A623 | chrX:108,568,785 | A | ATCCGG | 8192 | 8197 | `CCACGATGGGGCCCC`**[A→ATCCGG]**`GGACCTCAAGGTATT` | **未確認**（個別ref_ll/var_llの記録が現存するチェックポイントに含まれていない） | 同上 | -0.007261（CSVに記録済み） |

（Insertion例のみ、現存する`xlas_565_new_score_checkpoint.pkl`に該当変異の生スコアが含まれていなかったため、ref_ll/var_llの個別値は再構成できませんでした。Δlog-likelihoodの値自体は保存済みCSVから確認できています。）

---

## 6. 2つの解析の整合性

| 項目 | ① ClinVar | ② XLAS | 一致 |
|---|---|---|---|
| モデル | evo2_7b | evo2_7b | ✅ |
| Window size | 8192bp | 8192bp | ✅ |
| reduce_method | mean（デフォルト） | mean（デフォルト） | ✅ |
| prepend_bos | False（デフォルト） | False（デフォルト） | ✅ |
| average_reverse_complement | False（デフォルト） | False（デフォルト） | ✅ |
| batch_size | 1（明示） | 1（デフォルト適用） | ✅ |
| Δ算出式 | var_ll − ref_ll | var_ll − ref_ll | ✅ |
| 変異の種類 | SNVのみ | SNV + 非SNV（同一関数で処理） | ①はSNV限定という点で**範囲が異なる**（手法自体は非SNVにも同一ロジックで拡張可能だが、①では実際にSNVしか扱っていない） |
| Reference genome取得元 | NCBI GRCh38.p14（ローカルFASTA） | Ensembl REST API | **経路が異なる**（配列一致は未検証） |
| 座標系 | 0-based（SPDI由来） | 1-based（VCF由来） | **表記が異なる**（実装内部ではどちらも最終的に0-basedインデックスに変換してから配列操作しており、計算結果への影響はないと考えられるが、座標変換ロジックの独立検証は本監査の範囲外） |
| REF不一致時の挙動 | 該当行を除外して続行 | assertでクラッシュ（＝事前に全件一致が保証された状態でのみ実行が完了する） | 挙動は異なるが、**最終的に採用されるデータは両方ともREFが参照ゲノムと一致する変異のみ**という点で実質的に同じ基準 |

**結論**: スコアリングの中核（モデル・window・reduce方法・BOS・RC平均・Δ算出式）は①②で完全に一致しています。異なるのはreference genomeの取得経路（NCBI直接 vs Ensembl REST）と、①がSNVのみを対象としている点です。論文のMethodsでは「同一のzero-shot Δlog-likelihoodプロトコルを両コホートに適用した」と記載した上で、この2点の違いを脚注等で明記することを推奨します。

### Evo2原著・公式GitHubの手法との相違点

- 原著のzero-shot VEP（BRCA1ノートブック等）は、本プロジェクトの①②と同じく「window中心に変異を配置し、mean reduce、Δ = var−ref」という基本設計を採用しており、**大枠は一致**しています（`run_alport_delta.py`のdocstringが明示的にBRCA1ノートブックを参照）。
- 公式リポジトリの`evo2/scoring.py`には`score_sequences_rc`（forward/reverse complement平均）という関数が用意されていますが、①②とも**これを使用せず**、常にforwardのみでスコアリングしています。原著の一部の解析（特にBRCA1ノートブック）では`average_reverse_complement=True`を使うケースがあるとされていますが、本プロジェクトでは一貫してFalseです。この点は論文で明示的に述べるべき相違点です。
- 非SNV（indel）の扱いについて、Evo2公式リポジトリ自体はindel特有の正規化方法を规定していません（zero-shot VEPの主要な公開例はSNV中心）。本プロジェクトの「参照/変異で入力長を揃えない」という扱いは、公式のサンプルコードを逸脱するものではありませんが、公式が明示的に推奨する方法が存在するわけでもないため、**独自の実装判断**として記載すべきです。

---

## 7. 総括: 未確認・要追加確認事項一覧

1. HuggingFace上のevo2_7bチェックポイントの具体的なrevision/commit hash
2. ①②の全実行について、標準出力ログファイルとしての完全な保存（本セッションで実行したもの以外は未確認）
3. NCBI GRCh38.p14（①）とEnsembl REST API提供配列（②）が、対象領域において塩基配列完全一致するかどうかの直接検証
4. Excelシート上の「Evo2_eligible」フラグ算出基準（特に変異サイズによる除外の有無）の一次資料
5. Insertion例（⑥-A623）の個別ref_ll/var_ll生データ（チェックポイントに現存せず）

以上。
