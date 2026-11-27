"""
Recompute Evo2 delta-likelihood scores for the Alport COL4A3/4/5 P/B ClinVar
SNV set using evo2_7b_262k (long-context), matching the original evo2_7b
pipeline (run_alport_delta.py) otherwise: 8192bp window, mean log-likelihood,
delta = var - ref.

Reuses (chrom, pos, ref, alt) already parsed in
"pre エクセルP_B_alport_evo2_delta.xlsx" (their SPDI-derived positions), so no
raw ClinVar xlsx or local FASTA is needed -- windows are built from
Ensembl-fetched, padded per-chromosome reference sequences (padding avoids
gene-boundary clipping).

Resilient to interruption: scores all unique windows in chunks, checkpointing
to a pickle after each chunk so a restart resumes instead of rescoring.

Output: alport_evo2_262k_pb.csv

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python "delta likelihoods/run_alport_evo2_262k_pb.py"
"""
import os, pickle, sys, time
import numpy as np
import pandas as pd
import torch

HERE     = os.path.dirname(os.path.abspath(__file__))
PB_XLSX  = os.path.join(HERE, "pre エクセルP_B_alport_evo2_delta.xlsx")
OUT_CSV  = os.path.join(HERE, "alport_evo2_262k_pb.csv")
CKPT     = os.path.join(HERE, "score_checkpoint_262k_pb.pkl")
SAE_DIR  = os.path.join(os.path.dirname(HERE), "SAE")
XLAS_DIR = os.path.join(os.path.dirname(HERE), "XLAS")

WINDOW_SIZE = 8192
MODEL_NAME  = "evo2_7b_262k"
CHUNK       = 250

sys.path.insert(0, SAE_DIR)
import run_col4_sae as sae_base


def load_padded_chrom(chrom, needed_min, needed_max, cache_path):
    if os.path.exists(cache_path):
        seq, start = pickle.load(open(cache_path, "rb"))
        end = start + len(seq) - 1
        if start <= needed_min - 100 and end >= needed_max + 100:
            print(f"  Loaded cached chr{chrom} padded seq: {start:,}-{end:,} ({len(seq):,} bp)")
            return seq, start
    pad = WINDOW_SIZE // 2 + 100
    start, end = needed_min - pad, needed_max + pad
    print(f"  Fetching chr{chrom}:{start:,}-{end:,} from Ensembl ...")
    seq = sae_base.fetch_ensembl(chrom, start, end)
    pickle.dump((seq, start), open(cache_path, "wb"))
    print(f"  Saved to {cache_path}")
    return seq, start


def main():
    print("Loading P/B variant table ...")
    df = pd.read_excel(PB_XLSX)
    df["chrom"] = df["GRCh38Chromosome"].astype(str)
    print(f"  {len(df)} rows  (P={ (df.label=='P').sum() }, B={ (df.label=='B').sum() })")

    print("\nLoading/fetching padded reference sequences ...")
    master = {}
    for chrom in df["chrom"].unique():
        sub = df[df["chrom"] == chrom]
        cache_path = os.path.join(
            XLAS_DIR if chrom == "X" else HERE,
            "col4a5_padded_sequence.pkl" if chrom == "X" else "col4a34_padded_sequence.pkl",
        )
        master[chrom] = load_padded_chrom(chrom, sub["GRCh38Location"].min(),
                                           sub["GRCh38Location"].max(), cache_path)

    print("\nBuilding windows ...")
    ref_windows, var_windows = [], []
    ref_seq_to_index, ref_indexes, scorable_idx, flags = {}, [], [], {}
    for i, row in df.iterrows():
        seq, start = master[row["chrom"]]
        local = int(row["GRCh38Location"]) - start
        s = max(0, local - WINDOW_SIZE // 2)
        e = min(len(seq), local + WINDOW_SIZE // 2)
        ref_w = seq[s:e]
        rel = local - s
        if ref_w[rel] != row["ref"]:
            flags[i] = f"ref mismatch: file={row['ref']} genome={ref_w[rel]}"
            continue
        var_w = ref_w[:rel] + row["alt"] + ref_w[rel + 1:]
        if ref_w not in ref_seq_to_index:
            ref_seq_to_index[ref_w] = len(ref_windows)
            ref_windows.append(ref_w)
        ref_indexes.append(ref_seq_to_index[ref_w])
        var_windows.append(var_w)
        scorable_idx.append(i)
    ref_indexes = np.array(ref_indexes)
    print(f"  {len(scorable_idx)} scorable, {len(flags)} flagged")
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
    df["flag"] = ""
    df.loc[scorable_idx, "evo2_delta_score"] = delta_scores
    for i, msg in flags.items():
        df.at[i, "flag"] = msg

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
