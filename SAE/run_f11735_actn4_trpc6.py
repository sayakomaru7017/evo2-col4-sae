"""
Check feature 11735's firing pattern on ACTN4 (alpha-actinin-4, spectrin-repeat
domain) and TRPC6 (ankyrin-repeat domain), two other podocyte/FSGS genes with
their OWN distinct repetitive protein folds (helical repeats), NOT Gly-X-Y.
External-validity test: does f11735 respond to repetitive protein structure
in general, or specifically to the Gly-X-Y collagen periodicity, even within
the same kidney-disease-gene research context as NPHS1/NPHS2?

Same method as the rest of this project: evo2_7b_262k, layer-26 Goodfire SAE
(BatchTopK, k=64, expansion x8, 32768 features), full gene length via
run_col4_sae.get_features (chunked/stitched for >130,000bp).

Output: SAE/f11735/{ACTN4,TRPC6}_f11735_profile.npz
"""
import os, time, pickle
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_PKL = os.path.join(HERE, "f11735", "actn4_trpc6_sequences.pkl")
OUT_DIR = os.path.join(HERE, "f11735")
FEAT = 11735


def main():
    seqs = pickle.load(open(SEQ_PKL, "rb"))

    print("Loading evo2_7b_262k ...")
    t0 = time.time()
    evo = base.ObservableEvo2("evo2_7b_262k")
    print(f"  ready in {time.time()-t0:.1f}s")
    base.enable_hcl_kernel(evo)

    sae_path = hf_hub_download(
        repo_id="Goodfire/Evo-2-Layer-26-Mixed",
        filename="sae-layer26-mixed-expansion_8-k_64.pt",
        repo_type="model",
    )
    sae = base.load_sae(sae_path, d_hidden=evo.d_hidden, device=evo.device,
                        dtype=torch.bfloat16, expansion=8)
    print(f"  SAE loaded: {evo.d_hidden*8} features, top-k 64, layer {base.SAE_LAYER}\n")

    for gene, meta in seqs.items():
        seq = meta["seq"]
        print(f"=== {gene}  {meta['region']}  {meta['strand']}  {len(seq):,} bp ===")

        t0 = time.time()
        feats = base.get_features(seq, evo, sae)
        dt = time.time() - t0
        L, F = feats.shape

        f_col = feats[:, FEAT]
        fire_pos = np.nonzero(f_col)[0]
        fire_vals = f_col[fire_pos]
        n_fire = len(fire_pos)
        print(f"  {L:,} bp, {dt:.1f}s, f{FEAT} fires {n_fire} times ({100*n_fire/L:.4f}% of positions)")

        if n_fire:
            print("  fire contexts (first 30):")
            for p, v in list(zip(fire_pos, fire_vals))[:30]:
                lo, hi = max(0, p-15), min(L, p+16)
                print(f"    {p}:{seq[lo:p]}[{seq[p]}]{seq[p+1:hi]}={v:.3f}")

        np.savez_compressed(
            os.path.join(OUT_DIR, f"{gene}_f11735_profile.npz"),
            seq_len=L, fire_pos=fire_pos.astype(np.int32), fire_vals=fire_vals.astype(np.float32),
            seq=seq, header=meta["region"], revcomp=(meta["strand"] == "-"), n_features=F,
        )
        print(f"  Saved: {gene}_f11735_profile.npz\n")

        del feats
        torch.cuda.empty_cache()

    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
