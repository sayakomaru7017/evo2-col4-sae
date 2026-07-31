"""
Extract ONLY features f11735, f9170, f5026 (full gene length) from the
wild-type and all mutant SAE feature files for COL4A3/COL4A4/COL4A5.

Saves two things per condition:
  1. A filtered .npz in the SAME schema as the original *_sae_features.npz
     (rows/cols/vals/shape + all original metadata), but with rows/cols/vals
     restricted to just these 3 features -- a drop-in replacement for the
     original files, just much smaller.
       WT:     SAE/features/<GENE>_f3_sae_features.npz
       mutant: SAE/features/<GENE>_<aa_tag>_mutant_f3_sae_features.npz
  2. A tidy CSV per gene combining WT + all mutants for quick inspection.
       SAE/features/<GENE>_target_features_wt_vs_mutants.csv
"""
import glob, os, re
import numpy as np
import pandas as pd

OUT_DIR  = "SAE/features"
FEATURES = [11735, 9170, 5026]
GENES    = ["COL4A3", "COL4A4", "COL4A5", "LAMB2", "NID1", "NPHS1", "APOL1", "NPHS2",
            "HTT", "FMR1", "DMPK", "FXN", "CNBP", "C9orf72", "LORICRIN", "ELN", "TPM1"]


def save_filtered_npz(npz, out_path):
    """Same schema as the source npz, rows/cols/vals restricted to FEATURES."""
    rows, cols, vals = npz["rows"], npz["cols"], npz["vals"]
    mask = np.isin(cols, FEATURES)
    kwargs = {k: npz[k] for k in npz.files if k not in ("rows", "cols", "vals")}
    kwargs["rows"] = rows[mask]
    kwargs["cols"] = cols[mask]
    kwargs["vals"] = vals[mask]
    np.savez_compressed(out_path, **kwargs)
    return int(mask.sum())


def dense_for(npz, feat_id):
    rows, cols, vals = npz["rows"], npz["cols"], npz["vals"]
    d = np.zeros(int(npz["shape"][0]), dtype=np.float32)
    mask = cols == feat_id
    d[rows[mask]] = vals[mask]
    return d


def rows_from_npz(npz, gene, condition, protein_change, gene_start, strand):
    out = []
    for feat_id in FEATURES:
        d = dense_for(npz, feat_id)
        nz = np.nonzero(d)[0]
        for off in nz:
            genomic = gene_start + int(off) if strand == "+" else int(npz["end"]) - int(off)
            out.append((gene, condition, protein_change, feat_id,
                         int(off), genomic, float(d[off])))
    return out


def main():
    for gene in GENES:
        wt_path = f"{OUT_DIR}/{gene}_sae_features.npz"
        if not os.path.exists(wt_path):
            print(f"skip {gene}: missing {wt_path}")
            continue
        wt = np.load(wt_path)
        gene_start, strand = int(wt["start"]), str(wt["strand"])

        records = rows_from_npz(wt, gene, "WT", "", gene_start, strand)
        wt_out = f"{OUT_DIR}/{gene}_f3_sae_features.npz"
        nnz = save_filtered_npz(wt, wt_out)
        print(f"{gene}: WT -> {os.path.basename(wt_out)}  (nnz={nnz})")

        mut_files = sorted(glob.glob(f"{OUT_DIR}/{gene}_*_mutant_sae_features.npz"))
        print(f"{gene}: {len(mut_files)} mutant files found")
        for mf in mut_files:
            aa_tag = re.match(rf"{gene}_(.+)_mutant_sae_features\.npz",
                               os.path.basename(mf)).group(1)
            mut = np.load(mf)
            protein_change = str(mut["protein_change"]) if "protein_change" in mut else aa_tag
            records += rows_from_npz(mut, gene, "mutant", protein_change, gene_start, strand)

            mut_out = f"{OUT_DIR}/{gene}_{aa_tag}_mutant_f3_sae_features.npz"
            nnz = save_filtered_npz(mut, mut_out)
            print(f"  {aa_tag} -> {os.path.basename(mut_out)}  (nnz={nnz})")

        df = pd.DataFrame(records, columns=["gene", "condition", "protein_change",
                                             "feature", "array_offset", "genomic_pos",
                                             "activation"])
        out_csv = f"{OUT_DIR}/{gene}_target_features_wt_vs_mutants.csv"
        df.to_csv(out_csv, index=False)
        print(f"  -> {out_csv}  ({len(df):,} rows, {df['protein_change'].nunique()} conditions incl. WT)")


if __name__ == "__main__":
    main()
