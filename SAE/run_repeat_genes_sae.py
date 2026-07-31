"""
Compute Evo2 SAE (layer-26) feature activations for 9 repeat-expansion /
periodic-motif genes, full gene length, same pipeline as run_col4_sae.py.

  HTT       chr4:3,041,363-3,243,957     +  202,595 bp  (two-chunk stitch)
  FMR1      chrX:147,911,858-147,951,125 +   39,268 bp
  DMPK      chr19:45,769,709-45,782,794  -   13,086 bp
  FXN       chr9:69,035,730-69,079,076   +   43,347 bp
  CNBP      chr3:129,167,291-129,184,008 -   16,718 bp
  C9orf72   chr9:27,535,640-27,573,895   -   38,256 bp
  LORICRIN  chr1:153,259,687-153,262,124 +    2,438 bp
  ELN       chr7:74,027,773-74,069,909   +   42,137 bp
  TPM1      chr15:63,042,130-63,071,915  +   29,786 bp

Output: SAE/features/<GENE>_sae_features.npz (same schema as COL4 pipeline)

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_repeat_genes_sae.py
"""
import os, time, pickle
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base

HERE    = os.path.dirname(os.path.abspath(__file__))
SEQ_PKL = os.path.join(HERE, "repeat_genes_sequences.pkl")
OUT_DIR = os.path.join(HERE, "features")

GENES = {
    "HTT":      {"chrom": "4",  "start": 3_041_363,   "end": 3_243_957,   "strand": "+"},
    "FMR1":     {"chrom": "X",  "start": 147_911_858, "end": 147_951_125, "strand": "+"},
    "DMPK":     {"chrom": "19", "start": 45_769_709,  "end": 45_782_794,  "strand": "-"},
    "FXN":      {"chrom": "9",  "start": 69_035_730,  "end": 69_079_076,  "strand": "+"},
    "CNBP":     {"chrom": "3",  "start": 129_167_291, "end": 129_184_008, "strand": "-"},
    "C9orf72":  {"chrom": "9",  "start": 27_535_640,  "end": 27_573_895,  "strand": "-"},
    "LORICRIN": {"chrom": "1",  "start": 153_259_687, "end": 153_262_124, "strand": "+"},
    "ELN":      {"chrom": "7",  "start": 74_027_773,  "end": 74_069_909,  "strand": "+"},
    "TPM1":     {"chrom": "15", "start": 63_042_130,  "end": 63_071_915,  "strand": "+"},
}
GENE_ORDER = ["HTT", "FMR1", "DMPK", "FXN", "CNBP", "C9orf72", "LORICRIN", "ELN", "TPM1"]


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
    mf.to_csv(os.path.join(OUT_DIR, "repeat_genes_manifest.csv"), index=False)
    print("=== Manifest ===\n" + mf.to_string(index=False))
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
