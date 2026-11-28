"""
RECONSTRUCTED, as faithfully as possible from this session's own prior tool-call
record, of the version of XLAS/run_teshigyou_kakunin_evo2_score.py that was
ACTUALLY EXECUTED to produce:
  - XLAS/teshigyou_kakunin_score_checkpoint.pkl
  - XLAS/臨床重症度/手作業確認済み_Evo2score付与.xlsx

IMPORTANT PROVENANCE NOTE (added 2026-11-27 audit):
The file currently committed to git (XLAS/run_teshigyou_kakunin_evo2_score.py,
commit 81d9bf4) is a LATER, EDITED version that adds a `normalize()` function
(left-align/trim, vt-normalize style) and targets a DIFFERENT checkpoint file
(`teshigyou_kakunin_score_checkpoint_v2.pkl`, suffix "_v2"). That v2 recompute
was proposed but, per explicit user instruction ("再計算しなくていいです"),
was NEVER RUN -- no "_v2" checkpoint or output file exists on disk.
git log -- XLAS/run_teshigyou_kakunin_evo2_score.py shows only ONE commit,
and that commit already contains the edited/normalize version, so the
ORIGINAL pre-edit script is NOT recoverable from git history. It is
reconstructed here from this session's own conversation transcript instead.

Verification performed during the 2026-11-27 audit: rebuilding the (ref,var)
windows exactly as in this reconstructed script, and mapping the existing
`teshigyou_kakunin_score_checkpoint.pkl` scores onto them, reproduces the
Evo2score column of 手作業確認済み_Evo2score付与.xlsx for ALL 400 rows to
floating-point precision (max |diff| = 5.2e-18). This confirms the
reconstruction below is logically identical to the code that actually
produced the existing output, independent of whether this exact text is
byte-for-byte what was run.
"""
import os, pickle, time
import numpy as np
import pandas as pd
import torch
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX_IN = os.path.join(HERE, "臨床重症度", "手作業確認済み.xlsx")
XLSX_OUT = os.path.join(HERE, "臨床重症度", "手作業確認済み_Evo2score付与.xlsx")
PAD_PKL = os.path.join(HERE, "col4a5_padded_sequence.pkl")
CKPT = os.path.join(HERE, "teshigyou_kakunin_score_checkpoint.pkl")

WINDOW_SIZE = 8192
MODEL_NAME = "evo2_7b"
CHUNK = 100
SHEET = "S作成all症例"


def build_window(ref_master, master_start, pos, ref, alt):
    local = pos - master_start
    s = max(0, local - WINDOW_SIZE // 2)
    e = min(len(ref_master), local + WINDOW_SIZE // 2)
    ref_window = ref_master[s:e]
    rel = local - s
    genome_ref = ref_window[rel:rel + len(ref)]
    assert genome_ref == ref, f"ref mismatch at {pos}: file={ref} genome={genome_ref}"
    var_window = ref_window[:rel] + alt + ref_window[rel + len(ref):]
    return ref_window, var_window


def main():
    wb = openpyxl.load_workbook(XLSX_IN, data_only=True)
    ws = wb[SHEET]
    headers = [c.value for c in ws[1]]
    data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
    df = pd.DataFrame(data, columns=headers)
    print(f"Loaded {len(df)} rows")

    ref_master, master_start = pickle.load(open(PAD_PKL, "rb"))

    df["key"] = list(zip(df["GRCh38_chr"], df["GRCh38_pos_VCF"].astype(int), df["REF"], df["ALT"]))
    unique_keys = df.drop_duplicates(subset="key")["key"].tolist()
    print(f"{len(unique_keys)} unique variants to score (from {len(df)} rows)")

    ref_windows, var_windows = [], []
    ref_seq_to_index, ref_indexes = {}, []
    for (chrom, pos, ref, alt) in unique_keys:
        ref_w, var_w = build_window(ref_master, master_start, pos, ref, alt)
        if ref_w not in ref_seq_to_index:
            ref_seq_to_index[ref_w] = len(ref_windows)
            ref_windows.append(ref_w)
        ref_indexes.append(ref_seq_to_index[ref_w])
        var_windows.append(var_w)
    ref_indexes = np.array(ref_indexes)
    print(f"{len(ref_windows)} unique ref windows, {len(var_windows)} var windows")

    all_seqs = ref_windows + var_windows
    n_total = len(all_seqs)

    if os.path.exists(CKPT):
        scores_done, next_idx = pickle.load(open(CKPT, "rb"))
        print(f"Resuming from checkpoint: {next_idx}/{n_total}")
    else:
        scores_done, next_idx = [], 0

    if next_idx < n_total:
        print(f"Loading {MODEL_NAME} ...")
        t0 = time.time()
        from evo2.models import Evo2
        model = Evo2(MODEL_NAME)
        print(f"  ready in {time.time()-t0:.1f}s")

        while next_idx < n_total:
            chunk = all_seqs[next_idx: next_idx + CHUNK]
            t0 = time.time()
            chunk_scores = model.score_sequences(chunk)
            dt = time.time() - t0
            scores_done.extend(chunk_scores)
            next_idx += len(chunk)
            pickle.dump((scores_done, next_idx), open(CKPT, "wb"))
            print(f"  scored {next_idx}/{n_total}  (+{len(chunk)} in {dt:.1f}s)")

        del model
        torch.cuda.empty_cache()

    all_scores = np.array(scores_done)
    ref_scores = all_scores[:len(ref_windows)]
    var_scores = all_scores[len(ref_windows):]
    deltas = var_scores - ref_scores[ref_indexes]
    key_to_score = dict(zip(unique_keys, deltas))

    df["Evo2score"] = df["key"].map(key_to_score)
    df = df.drop(columns=["key"])
    print(f"\nScored rows: {df['Evo2score'].notna().sum()} / {len(df)}")

    wb_out = openpyxl.Workbook()
    wb_out.remove(wb_out.active)
    ws_out = wb_out.create_sheet(SHEET)
    ws_out.append(headers)
    for _, row in df.iterrows():
        ws_out.append([row[h] if h in df.columns else None for h in headers])
    wb_out.save(XLSX_OUT)
    print(f"Saved: {XLSX_OUT}")


if __name__ == "__main__":
    main()
