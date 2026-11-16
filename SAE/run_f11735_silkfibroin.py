"""
Check feature 11735's firing pattern on Bombyx mori fibroin heavy chain
(FibH, GenBank OP839078.1, 15,960bp, "complete cds" so already in the
coding/5'->3' direction -- no revcomp needed).

FibH's core repetitive region encodes a (Gly-Ala)n / (Gly-Ser)n-type
2-residue period repeat (hexapeptide-ish crystalline beta-sheet motif,
e.g. GAGAGS), NOT the 3-residue Gly-X-Y collagen triple-helix period.
This fills in the "Gly-containing, 2-residue period" cell of the
specificity test (vs SRSF1 = non-Gly 2-residue period, DSPP = non-Gly
3-residue period, SFTPD/Scl2 = Gly-containing 3-residue period).

Same method as the rest of this project's SAE work: evo2_7b_262k,
layer-26 Goodfire SAE (BatchTopK, k=64, expansion x8, 32768 features),
single forward pass (15,960bp is well under the 130,000bp chunk limit).

Output: SAE/f11735/SilkFibroin_f11735_profile.npz
"""
import os, time
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base

HERE = os.path.dirname(os.path.abspath(__file__))
FASTA_PATH = os.path.join(HERE, "f11735", "SilkFibroin_FibH_OP839078.1.fasta")
OUT_PATH = os.path.join(HERE, "f11735", "SilkFibroin_f11735_profile.npz")
FEAT = 11735


def read_fasta(path):
    header, seq = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            header = line
        elif line:
            seq.append(line)
    return header, "".join(seq).upper()


def main():
    header, seq = read_fasta(FASTA_PATH)
    print(f"{header}  len={len(seq):,}")

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

    t0 = time.time()
    feats = base.get_features(seq, evo, sae)
    dt = time.time() - t0
    L, F = feats.shape

    f_col = feats[:, FEAT]
    fire_pos = np.nonzero(f_col)[0]
    fire_vals = f_col[fire_pos]
    n_fire = len(fire_pos)
    print(f"  {L:,} bp, {dt:.1f}s, f{FEAT} fires {n_fire} times ({100*n_fire/L:.3f}% of positions)")

    if n_fire:
        print("  fire contexts (first 30):")
        for p, v in list(zip(fire_pos, fire_vals))[:30]:
            lo, hi = max(0, p-15), min(L, p+16)
            print(f"    {p}:{seq[lo:p]}[{seq[p]}]{seq[p+1:hi]}={v:.3f}")

    np.savez_compressed(
        OUT_PATH,
        seq_len=L, fire_pos=fire_pos.astype(np.int32), fire_vals=fire_vals.astype(np.float32),
        seq=seq, header=header, revcomp=False, n_features=F,
    )
    print(f"\nSaved: {OUT_PATH}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
