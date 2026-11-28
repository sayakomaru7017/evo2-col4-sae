XLAS non-SNV 再スコアリング FINAL reviewed R2（2026-11-28）
対象57ユニーク変異（65症例）。旧ZIP/v2は使用しない。
★ CPU準備は全件完了。A100でprepareやpkl変換を行う必要はない。新規ライブラリ・モデルのダウンロード禁止。
★ 全57変異について元pklのREF一致、WT/MT 8192bp、旧コードのWT・旧MTとの配列完全一致を検証済み。GPUによるスコア再現性は未検証。

【再計算の背景・目的】
2026年9月4–5日（11月27日の監査報告に記載された実行日。元の実行ログによる独立確認は未了）のXLAS臨床重症度解析では、非SNVにもSNVと同じwindow構築方法を適用し、mutant配列長がWT window（8192 bp）と一致していなかった。原著Supplementary Information A.3.12では周辺の実ゲノム配列で長さを調整して両者の全長を揃える。方法論上の相違を是正するため、非SNV57ユニーク変異（65症例）のみを再計算する。解析結果の改善を目的としない。
旧ref_ll・var_ll・Evo2scoreは比較と研究記録のため保存し、上書きしない。WT windowは変更していない。旧方式mutantの対照は旧値の再現性確認にのみ使用する。新方式のvar_ll・ΔLLが旧値と異なることは想定どおりであり、旧値に合わせるために条件を変更してはならない。SNV335症例は変更しない。新旧スコアの解釈、変異除外、臨床解析への反映はMac側で行う。

【解析方法】
Nature Supplementary Information A.3.12: WT/MTの長さを8192bpに統一し、周辺の実ゲノム配列で長さを調整。今回の規則：従来のVCFアンカー中心のWT窓を維持し、MTの左右で均等に調整、奇数差の余りは右側。REF全体をALTに置換（delins含む）。原著が明記していない細部の実装規則であり、著者のClinVar評価実装そのものと同一とは断定しない。
従来のevo2_7b (1M checkpoint)、ゲノム+鎖、平均LL、BOSなし、batch_size=1、use_kernels=False、delta=mutant-ref。SNVの再計算不要。
今回アップロードされた再構成版旧コードに合わせ、from evo2.models import Evo2 / model=Evo2(MODEL) / model.score_sequences([seq]) とした。旧コードは最大100配列を渡すがデフォルトbatch_size=1で、今回は逐次1配列ずつ呼び出す。旧コード同様に外部torch.inference_mode()なし。コンストラクタは旧経路に戻し、キャッシュの事前確認とオフライン指定でダウンロードを防止。

【A100：起動後に行う作業】
1. 起動時刻を記録し、ZIPを展開してそのディレクトリへcd。従来のPython仮想環境を使用する。インストール、アップデート、再ダウンロード、新規解析は禁止。
   既存の準備済み rescore_20261128_final/ を使用。prepared.json, windows.csv, QC.csv, input_sequences.fasta が入っている。
2. 既存環境の確認とオフライン指定（Python起動前に必須）:
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   python -c 'from huggingface_hub import constants; print(constants.HF_HUB_CACHE)'
   cat ~/.cache/huggingface/hub/models--arcinstitute--evo2_7b/refs/main
   HF_HOME/HF_HUB_CACHEが異なれば読み替える。
   期待revision: bda0089f92582d5baabf0f22d9fc85f3588f6b58
   スクリプトは実キャッシュのrefs/main、snapshot/config.json、完成重みの存在とサイズを検証。通常の公式ローダーが優先するhub親ディレクトリのevo2_7b.pt、またはsnapshot内のevo2_7b.ptが必要。どちらもなければ停止（ダウンロード・マージ禁止）。両方が存在すればSHA256照合（同じ実ファイルなら省略）、完全なshardが残っていれば結合重みと照合する。結合済み重みしかない場合はrevisionとの暗号学的同一性が証明できないため、旧スコアの数値再現も確認し、記録する。
   公式コンストラクタにはrevision引数がないため、検証済みキャッシュmainをオフラインで使う。旧ローダーと異なるlocal_path経路は使わない。インストール済みのローダーが予想したキャッシュ経路を使うことを起動時に確認する。キャッシュ、パッケージ、設定に不一致があれば勝手に条件変更せず停止。
3. GPU実行（デフォルト厳格）。旧実行の配列あたりの処理時間は手元の監査資料から確認できないため、35分以内の完了は保証しない。モデルロード後の実測時間をログから確認し、残り時間を監視する。VM残時間に合わせて --minutes を短縮できる。以下の時間はPython起動時点からのソフト上限で、モデルロード等も含む。
   準備済み原本を保持し、毎回未使用のコピーを作って実行する。次の例ではrun01が未存在であることを確認する。
   test ! -e rescore_run01 && cp -a rescore_20261128_final rescore_run01
   上のコマンドが成功した場合のみ、以下を実行する。
   set -o pipefail
   python -u 02_XLAS_nonSNV_rescore.py score --out rescore_run01 --minutes 35 2>&1 | tee rescore_run01.log
   ユーザー確認により従来からA100のみ使用。今回の実行は --allow-numeric-drift を付けない（旧ref_ll・旧var_ll・旧ΔLLの絶対差1e-5以下をPASS）。GPU機種が同じでも実行環境の完全同一性は未証明。許容差を超えた場合は停止・報告。将来このオプションを使用する場合も、ユーザーが実行前に明示承認した場合に限る（上限1e-4）。失敗後の独自の緩和は禁止。閾値は本研究のQC運用規則で、Evo2原著の閾値ではない。
   全57変異のWT windowのref_ll照合を先に完了し、次に旧方式のmutant対照3変異を計算する（signed net changeの最小・中央値・最大、同値はvariant_id順）：NSV017（-21bp）、NSV026（-1bp）、NSV015（+18bp）。IDはコードで固定・検証する。旧ref_ll、旧var_ll、旧ΔLLの絶対差すべてに同一の上記閾値を適用する。既定は1e-5。いずれかがFAILなら記録して停止・報告し、新方式mutantを1件も計算しない。事前に承認された緩和条件に限りWARNで続行する。全照合完了後にのみ本計算に進む。
   照合済みの全57のrefを再利用し、同じ実行内のnew_var_ll-new_ref_llをnew_Evo2scoreとする。GPU入力は最大117配列（57 WT + 57 MT + 旧MT対照3）。1変異ごとにPARTIALを保存、全57完了時のみCOMPLETEを作る。途中停止時はPARTIALを保持。再実行がユーザーから承認された場合は、失敗時の出力・ログを保持し、未実行の準備済み原本 rescore_20261128_final/ を別名（例：rescore_run02）にコピーし、そのコピーを --out に指定する。runtime.json 等が残る実行済みディレクトリは再利用しない。
   保存物：runtime.json、pip_freeze.txt、nvidia_smi.txt、reference_check.csv、old_mutant_controls.csv、scores_COMPLETE.csv、実行ログ。スクリプト/loaderのSHA256、HF_HUB_OFFLINE、キャッシュの実体、パッケージとGPUも記録する。
4. COMPLETEが57行あることを確認し、出力ディレクトリとログを保存してユーザーに報告する。失敗・途中停止時も保存済み出力とログを報告する。VMの停止・割り当て解除はユーザー本人がAzure Portalで手動実施する。AIはVMの停止操作を行わない。1時間以内を目標に終了処理の時間を確保する。--minutesはモデルロード中や個々のforwardを強制中断しない。VM停止機能ではない。自動停止設定は不要とする。

【Mac側：GPU計算後】
python3 02_XLAS_nonSNV_rescore.py post-qc --scores rescore_run01/scores_COMPLETE.csv --out post_score_QC.csv
外部レビューの差の目安：|net_change|<=5なら0.0002、>=9なら0.0015、6–8は目安なし。超過はREVIEWとするのみで、自動除外・再計算判定にはしない。chr/pos/REF/ALTで65症例に更新し、335 SNVの既存スコアと全400症例を保持する。

【CPU検証記録・再現性（既に完了、A100では不要）】
今回取得したcol4a5_padded_sequence.pklは、配列（266195bp）と1-based開始座標108435549のタプル。chrX:108435549–108701743、GRCh38、+鎖。全配列ACGT。既存UCSC overlap (108439746–108697547、257802bp) と完全一致。以前のNSV016/029/048の未完成はUCSC区間が短いためのフランク不足で、今回すべて解消済み。
旧コードのbuild_window関数のみを抽出して、全57変異のWTと旧MTを直接比較し、57/57ずつ完全一致。旧MTは8171–8210bp、新MTとWTは全件8192bp。REF/ALTの最大長22bp（原著ClinVar評価の64bp以下）。模擬10正常・2異常テスト合格。CPU_VERIFICATION.jsonに記録。
旧コードは監査で会話記録から再構成された版であり、監査では既存checkpointの生スコアの対応付けにより400行のΔLL一致（最大絶対差5.2e-18）を確認した。これはGPU再計算による再現ではない。元の実行済みファイルとバイト単位で同一という意味ではない。旧コードのwindow構築を比較したものであり、旧スコア自体のGPU再現性はA100実行前には確認できない。
任意の再検証（標準ライブラリのみ、Evo2不要）：
   python3 verify_cpu.py
CPU準備を別ディレクトリで再実行する場合：
   python3 prepare_mac.py --reference col4a5_padded_sequence.fasta --out rescore_recheck
参照配列・旧コード本体・変異一覧・構築済み配列は同梱済み。NCBI代替取得は不要。
公式ローダー参照：https://github.com/ArcInstitute/evo2/blob/main/evo2/models.py
公式モデルファイル：https://huggingface.co/arcinstitute/evo2_7b/tree/main
原著 DOI: 10.1038/s41586-026-10176-5 Supplementary Information A.3.12–13
