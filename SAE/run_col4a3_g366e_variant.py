"""
Compute Evo2 SAE (layer-26) feature activations for the mutant COL4A3 sequence
carrying p.Gly366Glu (chr2:227,259,860 G>A, c.1097G>A), and save alongside the
existing wild-type COL4A3_sae_features.npz for comparison.

Reuses the model/SAE loading + chunked forward-pass machinery from
run_col4_sae.py so the mutant features are computed with the exact same
chunking (two-chunk stitch) as the cached wild-type COL4A3 features, keeping
array positions directly comparable.

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python SAE/run_col4a3_g366e_variant.py
"""
import os, time, pickle
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "features", "COL4A3_G366E_mutant_sae_features.npz")

GENOMIC_POS = 227_259_860   # chr2, GRCh38, 1-based
REF, ALT    = "G", "A"


def main():
    seqs = pickle.load(open(base.SEQ_PKL, "rb"))
    meta = seqs["COL4A3"]
    wt_seq = meta["seq"]
    assert meta["strand"] == "+"

    local = GENOMIC_POS - meta["start"]   # 0-based offset into wt_seq
    assert wt_seq[local] == REF, f"Expected {REF} at offset {local}, found {wt_seq[local]}"

    mut_seq = wt_seq[:local] + ALT + wt_seq[local + 1:]
    print(f"COL4A3 {meta['region']}  L={len(wt_seq):,}")
    print(f"  variant: chr2:{GENOMIC_POS:,} {REF}>{ALT}  (p.Gly366Glu)  array offset={local:,}")
    print(f"  wt  context: {wt_seq[local-15:local+16]}")
    print(f"  mut context: {mut_seq[local-15:local+16]}")

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

    print("=== COL4A3 mutant (p.Gly366Glu) forward pass ===")
    t0 = time.time()
    feats = base.get_features(mut_seq, evo, sae)
    dt = time.time() - t0
    L, F = feats.shape
    rows, cols = np.nonzero(feats)
    vals = feats[rows, cols].astype(np.float32)

    np.savez_compressed(
        OUT,
        rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
        shape=np.array([L, F], dtype=np.int64),
        gene="COL4A3", region=meta["region"],
        chrom=meta["chrom"], start=meta["start"], end=meta["end"],
        strand=meta["strand"], seq_len=L, n_features=F,
        topk=64, layer=base.SAE_LAYER,
        variant="chr2:227259860 G>A p.Gly366Glu c.1097G>A",
        variant_array_offset=local,
    )
    nnz = len(vals)
    print(f"  {L:,} x {F:,}  nnz={nnz:,} ({nnz/L:.1f}/pos)  {dt:.1f}s  -> {os.path.basename(OUT)}"
          f"  [{os.path.getsize(OUT)/1e6:.1f} MB]")
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
