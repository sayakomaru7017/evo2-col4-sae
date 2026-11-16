"""
Check feature 11735's firing pattern on 4 non-COL4 sequences, to test
whether it is specific to Gly-X-Y collagen repeats or responds more
generally to repetitive low-complexity sequence (including non-Gly
repeats and repeat periods that are not a multiple of 3 amino acids):

  - DSPP   (hg38, + strand): has a large Asp-Ser-Ser (DSS) acidic repeat
            domain, NOT a Gly repeat.
  - SFTPD  (hg38, - strand): has a short vertebrate collagen-like domain
            (Gly-X-Y repeats), a positive control in a very different
            overall gene context.
  - SRSF1  (hg38, - strand): has an RS domain (Arg-Ser dipeptide
            repeats), a 2-residue repeat period (not Gly, not a 3-residue
            period).
  - Scl2   (Streptococcus pyogenes, GenBank AY069936.1): a BACTERIAL
            collagen-like protein with genuine Gly-X-Y repeats, testing
            whether f11735 generalizes outside vertebrate sequence
            context entirely.

Method: identical to the rest of this project's SAE work --
evo2_7b_262k, layer-26 Goodfire SAE (BatchTopK, k=64, expansion x8,
32768 features), single forward pass (all 4 sequences are well under
the 130,000bp chunk limit). Minus-strand UCSC downloads are reverse-
complemented before scoring, consistent with how COL4A4/COL4A6 were
handled throughout this project (UCSC "hg38_dna" exports always give
the + strand of the range; strand=- in the header means revcomp first).

Output: SAE/f11735/<name>_f11735_profile.npz  (per-sequence) and
        SAE/f11735/f11735_novel_sequences_summary.csv
"""
import os, time, pickle
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import run_col4_sae as base

HERE = os.path.dirname(os.path.abspath(__file__))
FASTA_DIR = os.path.join(HERE, "f11735")
FEAT = 11735

SEQUENCES = [
    dict(name="DSPP", path="DSPP_hg38_genomic_plus.fasta", revcomp=False),
    dict(name="SFTPD", path="SFTPD_hg38_genomic_minus.fasta", revcomp=True),
    dict(name="SRSF1", path="SRSF1_hg38_genomic_minus.fasta", revcomp=True),
    dict(name="Scl2", path="Scl2_AY069936.1_DNA.fasta", revcomp=False),
]

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
def revcomp(seq):
    return seq.translate(_COMP)[::-1]

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

    summary_rows = []
    for item in SEQUENCES:
        header, raw_seq = read_fasta(os.path.join(FASTA_DIR, item["path"]))
        seq = revcomp(raw_seq) if item["revcomp"] else raw_seq
        name = item["name"]
        print(f"=== {name}  len={len(seq):,}  revcomp={item['revcomp']}  header={header} ===")

        t0 = time.time()
        feats = base.get_features(seq, evo, sae)   # (L, 32768)
        dt = time.time() - t0
        L, F = feats.shape

        f_col = feats[:, FEAT]
        fire_pos = np.nonzero(f_col)[0]
        fire_vals = f_col[fire_pos]
        n_fire = len(fire_pos)
        print(f"  {L:,} bp, {dt:.1f}s, f{FEAT} fires {n_fire} times "
              f"({100*n_fire/L:.3f}% of positions)")

        if n_fire:
            print("  fire contexts (first 20):")
            for p, v in list(zip(fire_pos, fire_vals))[:20]:
                lo, hi = max(0, p-15), min(L, p+16)
                print(f"    {p}:{seq[lo:p]}[{seq[p]}]{seq[p+1:hi]}={v:.3f}")

        np.savez_compressed(
            os.path.join(FASTA_DIR, f"{name}_f11735_profile.npz"),
            seq_len=L, fire_pos=fire_pos.astype(np.int32), fire_vals=fire_vals.astype(np.float32),
            seq=seq, header=header, revcomp=item["revcomp"], n_features=F,
        )

        summary_rows.append(dict(
            name=name, header=header, seq_len=L, revcomp=item["revcomp"],
            n_fire=n_fire, frac_fire=n_fire/L,
            fire_positions=",".join(map(str, fire_pos.tolist())),
        ))

        del feats
        torch.cuda.empty_cache()
        print()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(FASTA_DIR, "f11735_novel_sequences_summary.csv"), index=False)
    print("=== Summary ===")
    print(summary[["name", "seq_len", "n_fire", "frac_fire"]].to_string(index=False))
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
