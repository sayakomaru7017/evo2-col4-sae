"""
Compute Evo2 SAE (layer-26) feature activations for the FULL-LENGTH mutant
gene sequence of every candidate variant in evo2_candidates_continuous.csv
(COL4A3/COL4A4/COL4A5 glycine-substitution variants).

Loads the model + SAE once, then for each gene loads the cached wild-type
sequence and, for every variant on that gene, substitutes the single base
(handling minus-strand genes by taking the reverse complement of ref/alt
and mirroring the coordinate) and runs the same chunked forward pass used
for the wild-type COL4A3/4/5 features, so array positions line up with the
existing wild-type *_sae_features.npz files.

Output: SAE/features/<GENE>_<AA1><pos><AA1>_mutant_sae_features.npz
        (e.g. COL4A3_G366E_mutant_sae_features.npz)

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_candidate_variants_sae.py
"""
import os, re, time, pickle
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(HERE, "features")
CSV_PATH = os.path.join(OUT_DIR, "evo2_candidates_continuous.csv")

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
def complement(base: str) -> str:
    return base.translate(_COMP)

AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}

def short_aa_change(protein_change: str) -> str:
    m = re.match(r"^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})$", protein_change)
    aa_from, pos, aa_to = m.groups()
    return f"{AA3TO1[aa_from]}{pos}{AA3TO1[aa_to]}"


def build_mutant_seq(wt_seq: str, meta: dict, genomic_pos: int, ref: str, alt: str):
    """Return (mutant_seq, local_offset_0based). Handles +/- strand genes."""
    start, end, strand = meta["start"], meta["end"], meta["strand"]
    if strand == "+":
        local = genomic_pos - start
        exp_ref, exp_alt = ref, alt
    else:
        local = end - genomic_pos
        exp_ref, exp_alt = complement(ref), complement(alt)
    assert wt_seq[local] == exp_ref, (
        f"Expected {exp_ref} at local offset {local} (strand {strand}), "
        f"found {wt_seq[local]}"
    )
    mut_seq = wt_seq[:local] + exp_alt + wt_seq[local + 1:]
    return mut_seq, local


def main(csv_path=CSV_PATH, manifest_name="candidate_variants_manifest.csv"):
    df = pd.read_csv(csv_path)
    seqs = pickle.load(open(base.SEQ_PKL, "rb"))

    print("Loading evo2_7b_262k ...")
    t0 = time.time()
    evo = base.ObservableEvo2("evo2_7b_262k")
    print(f"  ready in {time.time()-t0:.1f}s  device={evo.device}  dtype={evo.dtype}")
    base.enable_hcl_kernel(evo)

    sae_path = hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
        repo_type="model",
    )
    sae = base.load_sae(sae_path, d_hidden=evo.d_hidden, device=evo.device,
                        dtype=torch.bfloat16, expansion=8)
    print(f"  SAE loaded: {evo.d_hidden*8} features, top-k 64, layer {base.SAE_LAYER}\n")

    manifest = []
    for gene, sub in df.groupby("gene", sort=False):
        meta = seqs[gene]
        wt_seq = meta["seq"]
        print(f"=== {gene}  {meta['region']}  {meta['strand']}  {len(wt_seq):,} bp "
              f"({len(sub)} variants) ===")

        for _, row in sub.iterrows():
            aa_tag = short_aa_change(row["protein_change"])
            out = os.path.join(OUT_DIR, f"{gene}_{aa_tag}_mutant_sae_features.npz")
            if os.path.exists(out):
                print(f"  [{aa_tag}] already exists, skipping -> {os.path.basename(out)}")
                manifest.append(dict(gene=gene, protein_change=row["protein_change"],
                                      aa_tag=aa_tag, npz=os.path.basename(out), skipped=True))
                continue

            mut_seq, local = build_mutant_seq(
                wt_seq, meta, int(row["GRCh38_pos"]), row["ref"], row["alt"]
            )
            print(f"  [{aa_tag}] chr{row['GRCh38_chr']}:{row['GRCh38_pos']:,} "
                  f"{row['ref']}>{row['alt']}  (array offset={local:,})")

            t0 = time.time()
            feats = base.get_features(mut_seq, evo, sae)
            dt = time.time() - t0
            L, F = feats.shape
            rows_, cols_ = np.nonzero(feats)
            vals_ = feats[rows_, cols_].astype(np.float32)

            np.savez_compressed(
                out,
                rows=rows_.astype(np.int32), cols=cols_.astype(np.int32), vals=vals_,
                shape=np.array([L, F], dtype=np.int64),
                gene=gene, region=meta["region"],
                chrom=meta["chrom"], start=meta["start"], end=meta["end"],
                strand=meta["strand"], seq_len=L, n_features=F,
                topk=64, layer=base.SAE_LAYER,
                protein_change=row["protein_change"],
                variant=f"chr{row['GRCh38_chr']}:{row['GRCh38_pos']} {row['ref']}>{row['alt']}",
                variant_array_offset=local,
                clinvar_classification=row["clinvar_classification"],
                evo2_delta_score=row["evo2_delta_score"],
            )
            nnz = len(vals_)
            print(f"    {L:,} x {F:,}  nnz={nnz:,} ({nnz/L:.1f}/pos)  {dt:.1f}s "
                  f"-> {os.path.basename(out)}  [{os.path.getsize(out)/1e6:.1f} MB]")
            manifest.append(dict(gene=gene, protein_change=row["protein_change"],
                                  aa_tag=aa_tag, npz=os.path.basename(out), skipped=False))
            del feats
            torch.cuda.empty_cache()

    mf = pd.DataFrame(manifest)
    mf.to_csv(os.path.join(OUT_DIR, manifest_name), index=False)
    print("\n=== Manifest ===")
    print(mf.to_string(index=False))
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
