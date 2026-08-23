"""
Evo2 zero-shot delta-likelihood scoring (evo2_7b) for the XLAS COL4A5 Cox
survival cohort (Cox解析 sheet, Evo2_eligible == "Yes", 436 variants).

Same method as run_col4a5_evo2_vep.py: 8192bp window centered on each
variant (from a padded, gene-boundary-safe reference sequence), REF/ALT
substituted VCF-style (anchor-base convention: position = first base of
REF; ALT replaces the REF span), scored with Evo2.score_sequences()
(mean log-likelihood), delta = var - ref.

Resilient to interruption: scores all unique windows in chunks,
checkpointing to a pickle after each chunk.

Input : 組入症例＿XLAS_strict_analysis_population_v2_原著再監査_GRCh38_最終.xlsx,
        sheet "Cox解析"
Output: XLAS/xlas_cox_evo2_7b_delta.csv (all Cox解析 columns + evo2_delta_score)

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python XLAS/run_xlas_cox_evo2_delta.py
"""
import os, pickle, sys, time
import numpy as np
import pandas as pd
import torch
import openpyxl

HERE    = os.path.dirname(os.path.abspath(__file__))
XLSX    = os.path.join(HERE, "組入症例＿XLAS_strict_analysis_population_v2_原著再監査_GRCh38_最終.xlsx")
OUT_CSV = os.path.join(HERE, "xlas_cox_evo2_7b_delta.csv")
PAD_PKL = os.path.join(HERE, "col4a5_padded_sequence.pkl")
CKPT    = os.path.join(HERE, "xlas_cox_score_checkpoint.pkl")

WINDOW_SIZE = 8192
MODEL_NAME  = "evo2_7b"
CHUNK       = 250


def load_sheet(sheet_name):
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]
    data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
    return pd.DataFrame(data, columns=headers)


def build_windows(ref_master, master_start, pos, ref, alt):
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
    print("Loading Cox解析 sheet ...")
    df = load_sheet("Cox解析")
    df["GRCh38_pos_int"] = pd.to_numeric(df["GRCh38_pos"], errors="coerce")
    elig = df[df["Evo2_eligible"] == "Yes"].copy()
    print(f"  {len(df)} total rows, {len(elig)} Evo2_eligible")

    ref_master, master_start = pickle.load(open(PAD_PKL, "rb"))
    print(f"  Reference: chrX:{master_start:,}-{master_start+len(ref_master)-1:,} ({len(ref_master):,} bp)")

    print("\nBuilding windows ...")
    ref_windows, var_windows = [], []
    ref_seq_to_index, ref_indexes, scorable_idx = {}, [], []
    for i, row in elig.iterrows():
        pos = int(row["GRCh38_pos_int"])
        ref_w, var_w = build_windows(ref_master, master_start, pos, row["REF"], row["ALT"])
        if ref_w not in ref_seq_to_index:
            ref_seq_to_index[ref_w] = len(ref_windows)
            ref_windows.append(ref_w)
        ref_indexes.append(ref_seq_to_index[ref_w])
        var_windows.append(var_w)
        scorable_idx.append(i)
    ref_indexes = np.array(ref_indexes)
    print(f"  {len(ref_windows)} unique ref windows, {len(var_windows)} var windows")

    all_seqs = ref_windows + var_windows
    n_total = len(all_seqs)

    if os.path.exists(CKPT):
        scores_done, next_idx = pickle.load(open(CKPT, "rb"))
        print(f"\nResuming from checkpoint: {next_idx}/{n_total} already scored")
    else:
        scores_done, next_idx = [], 0

    if next_idx < n_total:
        print(f"\nLoading {MODEL_NAME} ...")
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
            print(f"  scored {next_idx}/{n_total}  (+{len(chunk)} in {dt:.1f}s, "
                  f"{dt/len(chunk):.2f}s/seq)")

        del model
        torch.cuda.empty_cache()

    all_scores = np.array(scores_done)
    ref_scores = all_scores[:len(ref_windows)]
    var_scores = all_scores[len(ref_windows):]
    delta_scores = var_scores - ref_scores[ref_indexes]

    df["evo2_delta_score"] = np.nan
    df.loc[scorable_idx, "evo2_delta_score"] = delta_scores
    df.drop(columns=["GRCh38_pos_int"], inplace=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Scored: {len(scorable_idx)} / {len(df)} rows")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
