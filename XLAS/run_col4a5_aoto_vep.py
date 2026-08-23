"""
Evo2 zero-shot delta-likelihood scoring for the 14 Aoto experimental
COL4A5 splicing variants (col4a5_aoto_variants.csv), using BOTH
evo2_7b (standard) and evo2_7b_262k (long-context) for comparison.

Same method as run_col4a5_evo2_vep.py: 8192bp window centered on each
variant, padded reference sequence (no gene-boundary clipping), WT vs
variant windows scored with Evo2.score_sequences() (mean log-lik),
delta = var - ref.

Output: XLAS/col4a5_aoto_delta_scores.csv
  columns include evo2_delta_score_7b and evo2_delta_score_7b_262k

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python XLAS/run_col4a5_aoto_vep.py
"""
import os, pickle, sys, time
import numpy as np
import pandas as pd
import torch

HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_IN  = os.path.join(HERE, "col4a5_aoto_variants.csv")
OUT_CSV = os.path.join(HERE, "col4a5_aoto_delta_scores.csv")
PAD_PKL = os.path.join(HERE, "col4a5_padded_sequence.pkl")
SAE_DIR = os.path.join(os.path.dirname(HERE), "SAE")

WINDOW_SIZE = 8192
MODELS = ["evo2_7b", "evo2_7b_262k"]

sys.path.insert(0, SAE_DIR)
import run_col4_sae as sae_base

df = pd.read_csv(CSV_IN)
print(f"Loaded {len(df)} variants")

# ── Reference sequence: reuse the padded COL4A5 cache built for the
#    coordinate-list run (chrX, +/-~4.3kb beyond gene boundaries). ────────
if os.path.exists(PAD_PKL):
    print(f"Loading cached padded sequence from {PAD_PKL}")
    ref_master, master_start = pickle.load(open(PAD_PKL, "rb"))
else:
    seqs = pickle.load(open(os.path.join(SAE_DIR, "col4_sequences.pkl"), "rb"))
    gene_meta = seqs["COL4A5"]
    PAD = WINDOW_SIZE // 2 + 100
    master_start = gene_meta["start"] - PAD
    master_end = gene_meta["end"] + PAD
    print(f"Fetching padded COL4A5 region chrX:{master_start:,}-{master_end:,} ...")
    ref_master = sae_base.fetch_ensembl("X", master_start, master_end)
    pickle.dump((ref_master, master_start), open(PAD_PKL, "wb"))

master_len = len(ref_master)
print(f"Master COL4A5 sequence: chrX:{master_start:,}-{master_start+master_len-1:,}  ({master_len:,} bp)")


def local_index(genomic_pos: int) -> int:
    return genomic_pos - master_start


def build_windows(row):
    pos = int(row["pos"])
    local = local_index(pos)
    s = max(0, local - WINDOW_SIZE // 2)
    e = min(master_len, local + WINDOW_SIZE // 2)
    ref_window = ref_master[s:e]
    rel = local - s
    ref_base, alt_base = row["ref"], row["alt"]
    assert ref_window[rel] == ref_base, (
        f"ref mismatch at {pos}: file says {ref_base}, GRCh38 has {ref_window[rel]}"
    )
    var_window = ref_window[:rel] + alt_base + ref_window[rel + 1:]
    return ref_window, var_window


def main():
    # Build windows once (shared across both models)
    ref_windows, var_windows = [], []
    ref_seq_to_index, ref_indexes = {}, []
    for _, row in df.iterrows():
        ref_w, var_w = build_windows(row)
        if ref_w not in ref_seq_to_index:
            ref_seq_to_index[ref_w] = len(ref_windows)
            ref_windows.append(ref_w)
        ref_indexes.append(ref_seq_to_index[ref_w])
        var_windows.append(var_w)
    ref_indexes = np.array(ref_indexes)
    print(f"Built windows: {len(ref_windows)} unique reference windows, "
          f"{len(var_windows)} variant windows\n")

    from evo2.models import Evo2

    for model_name in MODELS:
        print(f"=== Loading {model_name} ===")
        t0 = time.time()
        model = Evo2(model_name)
        print(f"  ready in {time.time()-t0:.1f}s")

        t0 = time.time()
        ref_scores = model.score_sequences(ref_windows)
        print(f"  ref scored in {time.time()-t0:.1f}s")
        t0 = time.time()
        var_scores = model.score_sequences(var_windows)
        print(f"  var scored in {time.time()-t0:.1f}s")

        delta = np.array(var_scores) - np.array(ref_scores)[ref_indexes]
        df[f"evo2_delta_score_{model_name}"] = delta

        del model
        torch.cuda.empty_cache()
        print(f"  Peak GPU memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(df.to_string())


if __name__ == "__main__":
    main()
