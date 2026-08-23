"""
Evo2 zero-shot delta-likelihood scoring (evo2_7b) for 4 extra COL4A3/4/5
variants, same method as run_col4a5_evo2_vep.py / run_alport_evo2_262k_pb.py:
8192bp window, mean log-likelihood, delta = var - ref. Handles minus-strand
COL4A4 by taking the complement of the given (already +strand) ref/alt.

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python XLAS/run_extra_variants_delta.py
"""
import os, pickle, sys
import numpy as np
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "extra_variants_delta_scores.csv")
SAE_DIR = os.path.join(os.path.dirname(HERE), "SAE")

WINDOW_SIZE = 8192
MODEL_NAME  = "evo2_7b"

sys.path.insert(0, SAE_DIR)
import run_col4_sae as sae_base

VARIANTS = [
    dict(gene="COL4A5", protein="p.Gly953Val", cDNA="c.2858G>T", chrom="X", pos=108622766, ref="G", alt="T"),
    dict(gene="COL4A3", protein="p.Gly43Arg",  cDNA="c.127G>C",  chrom="2", pos=227238007, ref="G", alt="C"),
    dict(gene="COL4A4", protein="p.Gly545Ala", cDNA="c.1634G>C", chrom="2", pos=227082177, ref="C", alt="G"),
    dict(gene="COL4A4", protein="p.Gly999Glu", cDNA="c.2996G>A", chrom="2", pos=227051131, ref="C", alt="T"),
]
df = pd.DataFrame(VARIANTS)


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
    print("Loading/fetching padded reference sequences ...")
    master = {}
    for chrom in df["chrom"].unique():
        sub = df[df["chrom"] == chrom]
        cache_path = os.path.join(
            HERE if chrom == "X" else os.path.join(os.path.dirname(HERE), "delta likelihoods"),
            "col4a5_padded_sequence.pkl" if chrom == "X" else "col4a34_padded_sequence.pkl",
        )
        master[chrom] = load_padded_chrom(chrom, sub["pos"].min(), sub["pos"].max(), cache_path)

    print("\nBuilding windows ...")
    ref_windows, var_windows = [], []
    for _, row in df.iterrows():
        seq, start = master[row["chrom"]]
        local = int(row["pos"]) - start
        s = max(0, local - WINDOW_SIZE // 2)
        e = min(len(seq), local + WINDOW_SIZE // 2)
        ref_w = seq[s:e]
        rel = local - s
        assert ref_w[rel] == row["ref"], (
            f"ref mismatch {row['gene']} {row['protein']}: file={row['ref']} genome={ref_w[rel]}"
        )
        var_w = ref_w[:rel] + row["alt"] + ref_w[rel + 1:]
        ref_windows.append(ref_w)
        var_windows.append(var_w)
        print(f"  {row['gene']} {row['protein']}: OK  len(ref)={len(ref_w)}")

    print(f"\nLoading {MODEL_NAME} ...")
    from evo2.models import Evo2
    model = Evo2(MODEL_NAME)

    print(f"\nScoring {len(ref_windows)} ref windows ...")
    ref_scores = np.array(model.score_sequences(ref_windows))
    print(f"Scoring {len(var_windows)} var windows ...")
    var_scores = np.array(model.score_sequences(var_windows))

    df["ref_ll"] = ref_scores
    df["var_ll"] = var_scores
    df["evo2_delta_score_evo2_7b"] = var_scores - ref_scores

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(df.to_string())


if __name__ == "__main__":
    main()
