"""
Find SAE features that:
  (A) fire within ALL 6 specified genomic ranges, AND
  (B) do NOT fire outside those ranges (within the same gene's sequence)

Genomic ranges (GRCh38, 1-based inclusive):
  COL4A1  chr13:110,150,366-110,162,359   strand=-
  COL4A2  chr13:110,506,477-110,512,188   strand=+
  COL4A3  chr2 :227,307,790-227,311,864   strand=+
  COL4A4  chr2 :227,007,328-227,010,442   strand=-
  COL4A5  chrX :108,687,565-108,696,375   strand=+
  COL4A6  chrX :108,157,003-108,160,592   strand=-
"""

import os
import numpy as np
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "features")

# Gene metadata: Ensembl GRCh38 gene boundaries + user-specified analysis range
GENES = {
    "COL4A1": {
        "gene_start": 110_148_963, "gene_end": 110_307_202, "strand": "-",
        "range_start": 110_150_366, "range_end": 110_162_359,
    },
    "COL4A2": {
        "gene_start": 110_305_812, "gene_end": 110_513_209, "strand": "+",
        "range_start": 110_506_477, "range_end": 110_512_188,
    },
    "COL4A3": {
        "gene_start": 227_164_624, "gene_end": 227_314_792, "strand": "+",
        "range_start": 227_307_790, "range_end": 227_311_864,
    },
    "COL4A4": {
        "gene_start": 226_967_360, "gene_end": 227_164_488, "strand": "-",
        "range_start": 227_007_328, "range_end": 227_010_442,
    },
    "COL4A5": {
        "gene_start": 108_439_745, "gene_end": 108_697_547, "strand": "+",
        "range_start": 108_687_565, "range_end": 108_696_375,
    },
    "COL4A6": {
        "gene_start": 108_155_607, "gene_end": 108_439_497, "strand": "-",
        "range_start": 108_157_003, "range_end": 108_160_592,
    },
}

GENE_ORDER = ["COL4A1", "COL4A2", "COL4A3", "COL4A4", "COL4A5", "COL4A6"]


def genomic_to_array(gene_start, gene_end, strand, g_start, g_end):
    """Return (arr_lo, arr_hi) as a half-open [lo, hi) feature-array slice."""
    if strand == "+":
        return g_start - gene_start, g_end - gene_start + 1
    else:  # "-": position 0 = genomic gene_end, position i = gene_end - i
        lo = gene_end - g_end
        hi = gene_end - g_start + 1
        return lo, hi


def load_sparse(gene):
    path = os.path.join(OUT_DIR, f"{gene}_sae_features.npz")
    d = np.load(path)
    return d["rows"].astype(np.int32), d["cols"].astype(np.int32), d["shape"]


def main():
    exclusive_sets = {}
    inside_sets    = {}

    for gene in GENE_ORDER:
        g = GENES[gene]
        arr_lo, arr_hi = genomic_to_array(
            g["gene_start"], g["gene_end"], g["strand"],
            g["range_start"], g["range_end"],
        )
        rows, cols, shape = load_sparse(gene)
        L, F = shape

        mask_in  = (rows >= arr_lo) & (rows < arr_hi)
        mask_out = ~mask_in

        inside_feats  = set(cols[mask_in].tolist())
        outside_feats = set(cols[mask_out].tolist())
        exclusive     = inside_feats - outside_feats

        n_in  = int(mask_in.sum())
        n_out = int(mask_out.sum())
        range_len = arr_hi - arr_lo

        print(f"\n{gene}  array[{arr_lo:,}:{arr_hi:,}]  ({range_len:,} bp of {L:,})")
        print(f"  activations inside : {n_in:,}  → {len(inside_feats):,} distinct features")
        print(f"  activations outside: {n_out:,}  → {len(outside_feats):,} distinct features")
        print(f"  exclusive (in∩¬out) : {len(exclusive):,} features")

        inside_sets[gene]    = inside_feats
        exclusive_sets[gene] = exclusive

    # Intersection across all genes
    common_inside    = set.intersection(*inside_sets.values())
    common_exclusive = set.intersection(*exclusive_sets.values())

    print("\n" + "="*60)
    print(f"Features active in ALL 6 ranges              : {len(common_inside):,}")
    print(f"Features active in ALL 6 AND nowhere outside : {len(common_exclusive):,}")

    if common_exclusive:
        feat_ids = sorted(common_exclusive)
        print(f"\nFeature IDs (SAE feature index, 0-based):")
        for fid in feat_ids:
            print(f"  {fid}")

        # Per-gene activation statistics for these features
        print("\nPer-gene mean activation (inside range) for exclusive features:")
        rows_dict  = {}
        cols_dict  = {}
        vals_dict  = {}
        shape_dict = {}
        for gene in GENE_ORDER:
            d = np.load(os.path.join(OUT_DIR, f"{gene}_sae_features.npz"))
            rows_dict[gene]  = d["rows"].astype(np.int32)
            cols_dict[gene]  = d["cols"].astype(np.int32)
            vals_dict[gene]  = d["vals"].astype(np.float32)
            shape_dict[gene] = d["shape"]

        records = []
        for fid in feat_ids:
            row = {"feature": fid}
            for gene in GENE_ORDER:
                g = GENES[gene]
                arr_lo, arr_hi = genomic_to_array(
                    g["gene_start"], g["gene_end"], g["strand"],
                    g["range_start"], g["range_end"],
                )
                rows = rows_dict[gene]
                cols = cols_dict[gene]
                vals = vals_dict[gene]
                mask = (cols == fid) & (rows >= arr_lo) & (rows < arr_hi)
                row[f"{gene}_n_pos"]   = int(mask.sum())
                row[f"{gene}_mean_act"] = float(vals[mask].mean()) if mask.any() else 0.0
            records.append(row)

        df = pd.DataFrame(records)
        print(df.to_string(index=False))

        out_csv = os.path.join(OUT_DIR, "exclusive_features.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nSaved to {out_csv}")

    else:
        print("\nNo features satisfy both conditions.")
        print("Showing features active in ALL 6 ranges (relaxed: may fire outside too):")
        if common_inside:
            print(f"  {sorted(common_inside)[:50]}{'...' if len(common_inside)>50 else ''}")

        # Fallback: count how many genes' outside-sets each common_inside feature avoids
        # i.e., rank by "in how many genes does this feature NOT fire outside the range"
        if common_inside:
            scores = {}
            for fid in common_inside:
                score = sum(fid not in exclusive_sets[gene] and fid in inside_sets[gene]
                            for gene in GENE_ORDER)
                # Actually count how many genes have it as exclusive
                excl_count = sum(fid in exclusive_sets[gene] for gene in GENE_ORDER)
                scores[fid] = excl_count
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            print("\nTop features ranked by exclusivity (# genes where feature fires only inside range):")
            for fid, sc in ranked[:20]:
                print(f"  feature {fid:5d}  exclusive_in={sc}/6 genes")

            out_csv = os.path.join(OUT_DIR, "common_inside_features.csv")
            pd.DataFrame([{"feature": fid, "exclusive_in_n_genes": sc}
                          for fid, sc in ranked]).to_csv(out_csv, index=False)
            print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
