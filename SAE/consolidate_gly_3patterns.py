"""
Consolidate all 38 ClinVar Gly-substitution candidate variants' full SAE
feature matrices (32768 features, evo2_7b_262k) plus the 3 WT gene
matrices into ONE combined file.

Output: "/home/azureuser/Evo2/evo2/Gly substitution/gly_3patterns_all_sae_features.pkl"
  A dict:
    "WT": {gene: {rows, cols, vals, shape, start, end, strand, chrom, region}}
    "variants": {f"{gene}_{aa_tag}": {rows, cols, vals, shape, start, end,
                 strand, chrom, region, pattern, clinvar_classification,
                 evo2_delta_score, aa_position, codon_base_position,
                 protein_change, genomic_pos, ref, alt}}
"""
import os, pickle, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/azureuser/Evo2/SAE")
from run_candidate_variants_sae import short_aa_change

FEATURES_DIR = "/home/azureuser/Evo2/SAE/features"
CANDIDATES_CSV = "/home/azureuser/Evo2/SAE/gly_3patterns_candidates.csv"
ORIG_CSV = "/home/azureuser/Evo2/evo2/Gly substitution/gly_variant_candidates_3patterns.csv"
OUT_DIR = "/home/azureuser/Evo2/evo2/Gly substitution"
OUT_PATH = os.path.join(OUT_DIR, "gly_3patterns_all_sae_features.pkl")

os.makedirs(OUT_DIR, exist_ok=True)


def npz_to_dict(npz):
    d = {k: npz[k] for k in npz.files}
    # unwrap 0-d numpy scalars to plain python
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.ndim == 0:
            d[k] = v.item()
    return d


def main():
    df = pd.read_csv(CANDIDATES_CSV)
    orig = pd.read_csv(ORIG_CSV)

    combined = {"WT": {}, "variants": {}}

    for gene in ["COL4A3", "COL4A4", "COL4A5"]:
        npz = np.load(f"{FEATURES_DIR}/{gene}_sae_features.npz")
        combined["WT"][gene] = npz_to_dict(npz)
        print(f"WT {gene}: shape={combined['WT'][gene]['shape']}")

    n_ok, n_fail = 0, 0
    for _, row in df.iterrows():
        gene = row["gene"]
        aa_tag = short_aa_change(row["protein_change"])
        key = f"{gene}_{aa_tag}"
        path = f"{FEATURES_DIR}/{gene}_{aa_tag}_mutant_sae_features.npz"
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            n_fail += 1
            continue
        npz = np.load(path)
        d = npz_to_dict(npz)

        orig_row = orig[(orig["gene"] == gene) &
                         (orig["genomic_position_GRCh38"] == row["GRCh38_pos"]) &
                         (orig["ref"] == row["ref"]) & (orig["alt"] == row["alt"])]
        if len(orig_row):
            r0 = orig_row.iloc[0]
            d["pattern"] = r0["pattern"]
            d["aa_position"] = r0["aa_position"]
            d["codon_base_position"] = r0["codon_base_position"]
            d["f11735_WT_value"] = r0["f11735_WT_value"]

        d["protein_change"] = row["protein_change"]
        d["clinvar_classification"] = row["clinvar_classification"]
        d["evo2_delta_score"] = row["evo2_delta_score"]
        d["genomic_pos"] = int(row["GRCh38_pos"])
        d["ref"] = row["ref"]
        d["alt"] = row["alt"]

        combined["variants"][key] = d
        n_ok += 1

    print(f"\nconsolidated: {n_ok} variants ok, {n_fail} missing")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Saved: {OUT_PATH}  [{size_mb:.1f} MB]")

    # verify round-trip
    with open(OUT_PATH, "rb") as f:
        check = pickle.load(f)
    print(f"\nVerify: WT genes={list(check['WT'].keys())}  n_variants={len(check['variants'])}")
    sample_key = next(iter(check["variants"]))
    print(f"sample variant '{sample_key}' keys: {list(check['variants'][sample_key].keys())}")


if __name__ == "__main__":
    main()
