"""
Compute Evo2 SAE (layer-26, evo2_7b_262k, 32768 features) full-gene-length
feature activations for the 13 additional NC1-domain COL4A5 variants
(9 from NC1_Evo2_validation_candidates_new_only.csv + 4 population-tolerated
controls from NC1_population_tolerated_candidates_for_f26750.csv).

Same method as run_nc1_41_sae.py / run_candidate_variants_sae.py (full
gene-length forward pass, chunked/stitched for sequences > 130,000bp).

Output: NC1 domain/追加分/<GENE>_<AA_TAG>_mutant_sae_features.npz
"""
import os, time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base
import run_candidate_variants_sae as cand

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "nc1_13_additional_candidates.csv")
OUT_DIR = "/home/azureuser/Evo2/NC1 domain/追加分"


def main():
    df = pd.read_csv(CSV_PATH)
    seqs = __import__("pickle").load(open(base.SEQ_PKL, "rb"))
    df["aa_tag"] = df["protein_change"].map(cand.short_aa_change)

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
            aa_tag = row["aa_tag"]
            fname = f"{gene}_{aa_tag}_mutant_sae_features.npz"
            out = os.path.join(OUT_DIR, fname)
            if os.path.exists(out):
                print(f"  [{aa_tag}] already exists, skipping -> {fname}")
                manifest.append(dict(gene=gene, protein_change=row["protein_change"],
                                      aa_tag=aa_tag, npz=fname, skipped=True))
                continue

            mut_seq, local = cand.build_mutant_seq(
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
                  f"-> {fname}  [{os.path.getsize(out)/1e6:.1f} MB]")
            manifest.append(dict(gene=gene, protein_change=row["protein_change"],
                                  aa_tag=aa_tag, npz=fname, skipped=False))
            del feats
            torch.cuda.empty_cache()

    mf = pd.DataFrame(manifest)
    mf.to_csv(os.path.join(HERE, "nc1_13_additional_manifest.csv"), index=False)
    print("\n=== Manifest ===")
    print(mf.to_string(index=False))
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
