"""
Compute Evo2 SAE (layer-26) feature activations for LAMB2, NID1, NPHS1
(GBM / slit-diaphragm genes), full gene length, same pipeline as run_col4_sae.py.

All three genes are <130 kb (single-chunk forward pass, no stitching needed):
  LAMB2  chr3:49,121,113-49,133,118   -   12,006 bp
  NID1   chr1:235,975,830-236,083,915 -  108,086 bp
  NPHS1  chr19:35,825,372-35,869,287  -   43,916 bp

Output: SAE/features/<GENE>_sae_features.npz (same schema as COL4 pipeline)

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_new_genes_sae.py
"""
import os, time, pickle
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base

HERE    = os.path.dirname(os.path.abspath(__file__))
SEQ_PKL = os.path.join(HERE, "new_genes_sequences.pkl")
OUT_DIR = os.path.join(HERE, "features")

GENES = {
    "LAMB2": {"chrom": "3",  "start": 49_121_113,  "end": 49_133_118,  "strand": "-"},
    "NID1":  {"chrom": "1",  "start": 235_975_830, "end": 236_083_915, "strand": "-"},
    "NPHS1": {"chrom": "19", "start": 35_825_372,  "end": 35_869_287,  "strand": "-"},
    "APOL1": {"chrom": "22", "start": 36_253_071,  "end": 36_267_530,  "strand": "+"},
    "NPHS2": {"chrom": "1",  "start": 179_550_494, "end": 179_575_954, "strand": "-"},
}
GENE_ORDER = ["LAMB2", "NID1", "NPHS1", "APOL1", "NPHS2"]


def load_or_fetch_sequences() -> dict:
    seqs = {}
    if os.path.exists(SEQ_PKL):
        print(f"Loading cached sequences from {SEQ_PKL}")
        seqs = pickle.load(open(SEQ_PKL, "rb"))

    missing = [g for g in GENES if g not in seqs]
    if missing:
        print(f"Fetching {len(missing)} missing gene(s) from Ensembl REST API: {missing}")
        for gene in missing:
            g = GENES[gene]
            chrom, start, end, strand = g["chrom"], g["start"], g["end"], g["strand"]
            region = f"chr{chrom}:{start:,}-{end:,}"
            print(f"  {gene}  {region}  {strand}  ({end-start+1:,} bp)")
            raw = base.fetch_ensembl(chrom, start, end)
            seq = base.revcomp(raw) if strand == "-" else raw
            seqs[gene] = {"seq": seq, "region": region, "chrom": f"chr{chrom}",
                           "start": start, "end": end, "strand": strand}
            time.sleep(0.5)
        pickle.dump(seqs, open(SEQ_PKL, "wb"))
        print(f"  Saved sequence cache to {SEQ_PKL}\n")
    return seqs


def main():
    seqs = load_or_fetch_sequences()

    print("\nLoading evo2_7b_262k ...")
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
    for gene in GENE_ORDER:
        out = os.path.join(OUT_DIR, f"{gene}_sae_features.npz")
        if os.path.exists(out):
            print(f"=== {gene}: already exists, skipping -> {os.path.basename(out)} ===\n")
            continue
        meta = seqs[gene]
        seq = meta["seq"]
        print(f"=== {gene}  {meta['region']}  {meta['strand']}  {len(seq):,} bp ===")
        t0 = time.time()
        feats = base.get_features(seq, evo, sae)
        dt = time.time() - t0
        L, F = feats.shape
        rows, cols = np.nonzero(feats)
        vals = feats[rows, cols].astype(np.float32)

        out = os.path.join(OUT_DIR, f"{gene}_sae_features.npz")
        np.savez_compressed(
            out,
            rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
            shape=np.array([L, F], dtype=np.int64),
            gene=gene, region=meta["region"],
            chrom=meta["chrom"], start=meta["start"], end=meta["end"],
            strand=meta["strand"], seq_len=L, n_features=F,
            topk=64, layer=base.SAE_LAYER,
        )
        nnz = len(vals)
        size = os.path.getsize(out) / 1e6
        print(f"  {L:,} x {F:,}  nnz={nnz:,} ({nnz/L:.1f}/pos)  {dt:.1f}s  "
              f"-> {os.path.basename(out)}  [{size:.1f} MB]\n")
        manifest.append(dict(gene=gene, region=meta["region"], seq_len=L,
                              n_features=F, nnz=nnz, npz=os.path.basename(out)))
        del feats
        torch.cuda.empty_cache()

    import pandas as pd
    mf = pd.DataFrame(manifest)
    mf.to_csv(os.path.join(OUT_DIR, "new_genes_manifest.csv"), index=False)
    print("=== Manifest ===\n" + mf.to_string(index=False))
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
