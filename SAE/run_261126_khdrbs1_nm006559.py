"""
Recompute KHDRBS1 restricted to the representative RefSeq transcript
NM_006559.3's own genomic span (chr1:32013868-32043877, +, 30,010 bp),
replacing the earlier run which mistakenly used the longer NR_073499.2-based
range (chr1:32013868-32060850, 46,983 bp).

Same method/settings as the rest of the 2026-11-26 batch: evo2_7b_262k,
layer-26 Goodfire SAE (BatchTopK, k=64, expansion x8, 32768 features),
single forward pass (30,010bp <= 130,000bp chunk limit), no revcomp (+
strand gene, Ensembl fetch is already the literal plus strand).

Output (overwrites the earlier KHDRBS1 files under SAE/261126/analysis_data/):
  - KHDRBS1_sae_features.npz
  - KHDRBS1_input_sequence.fasta
"""
import os, sys, time
import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, "/home/azureuser/Evo2/SAE")
import run_col4_sae as base

OUT_DIR = "/home/azureuser/Evo2/SAE/261126/analysis_data"
CHROM, START, END, STRAND = "1", 32013868, 32043877, "+"
REFSEQ = "NM_006559.3"


def write_fasta(path, header, seq, width=70):
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(0, len(seq), width):
            f.write(seq[i:i+width] + "\n")


def main():
    seq = base.fetch_ensembl(CHROM, START, END)
    assert len(seq) == END - START + 1 == 30010
    print(f"KHDRBS1 (NM_006559.3 span): {len(seq):,} bp  strand={STRAND}  revcomp=no")

    region = f"chr{CHROM}:{START:,}-{END:,}"
    write_fasta(os.path.join(OUT_DIR, "KHDRBS1_input_sequence.fasta"),
                f">KHDRBS1 {region} strand={STRAND} {REFSEQ} "
                f"(gene-direction 5'->3', no revcomp; restricted to NM_006559.3 span)",
                seq)

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

    L = len(seq)
    method = "single_chunk" if L <= base.MAX_CHUNK else "2_chunk_overlap"
    t0 = time.time()
    feats = base.get_features(seq, evo, sae)
    dt = time.time() - t0
    Lf, F = feats.shape
    rows, cols = np.nonzero(feats)
    vals = feats[rows, cols].astype(np.float32)
    nnz = len(vals)

    out_npz = os.path.join(OUT_DIR, "KHDRBS1_sae_features.npz")
    np.savez_compressed(
        out_npz,
        rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
        shape=np.array([Lf, F], dtype=np.int64),
        gene="KHDRBS1", region=region, chrom=f"chr{CHROM}", start=START, end=END,
        strand=STRAND, seq_len=Lf, n_features=F, topk=64, layer=base.SAE_LAYER,
        method=method,
    )
    size_mb = os.path.getsize(out_npz) / 1e6
    log_line = (f"KHDRBS1 (NM_006559.3, {region}): {Lf:,} x {F:,}  nnz={nnz:,} "
                f"({nnz/Lf:.1f}/pos)  {dt:.1f}s -> {os.path.basename(out_npz)} "
                f"[{size_mb:.1f} MB]  method={method}\n"
                f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n"
                f"NOTE: replaces the earlier KHDRBS1 run, which used the longer "
                f"NR_073499.2-based span (chr1:32013868-32060850, 46,983bp) by mistake.")
    print(log_line)
    with open(os.path.join(OUT_DIR, "run_261126_khdrbs1_nm006559.log"), "w") as f:
        f.write(log_line + "\n")


if __name__ == "__main__":
    main()
