"""
Compute full-genome-region Evo2 SAE features (evo2_7b_262k, layer-26 Goodfire
SAE, BatchTopK k=64, expansion x8, 32768 features) for 5 genes requested on
2026-11-26:

  1. COL1A1   (recompute, chr17:50184101-50201631, -, 17,531 bp)
  2. SFTPD    (recompute, chr10:79937740-79949105, -, 11,366 bp)
  3. SRSF1    (recompute, chr17:58000919-58007246, -, 6,328 bp)
  4. KHDRBS1  (new,       chr1:32013868-32060850,  +, 46,983 bp)
  5. FibH     (new, Bombyx mori, RefSeq NM_001113262.1, complete CDS, 15,792 bp)

Strand handling (fixed from an earlier double-reverse-complement bug found
in the SFTPD/SRSF1 UCSC downloads used previously in this project):
  - All 4 human regions are fetched FRESH from the Ensembl REST API with an
    explicit forward-strand request (".../{chrom}:{start}..{end}:1"), which
    is guaranteed to return the literal plus-strand sequence regardless of
    the gene's own annotated strand.
  - Exactly one reverse-complement is then applied in this script for the
    minus-strand genes (COL1A1, SFTPD, SRSF1); KHDRBS1 (+) is used as-is.
  - This was verified against the user-provided UCSC "Get DNA" export for
    COL1A1 (strand=+ in its header -> identical to the Ensembl + fetch) and
    against the previously-saved SFTPD/SRSF1 UCSC files (strand=- in their
    header -> identical to revcomp(Ensembl +), proving those files were
    ALREADY in gene-direction and the old pipeline's extra revcomp() on top
    of them was a double-RC bug).
  - FibH uses the NCBI RefSeq CDS FASTA directly, 5'->3', no revcomp (it is
    already presented in coding-sense orientation). Contains one IUPAC
    ambiguity code (Y, position 13,055 0-based) inherited verbatim from the
    RefSeq record -- left unmodified.

All 5 sequences are <= 130,000 bp, so each uses a SINGLE forward pass
(method="single_chunk"), no chunk-stitching involved.

Output (under SAE/261126/analysis_data/):
  - <name>_sae_features.npz   (sparse rows/cols/vals + full metadata)
  - <name>_input_sequence.fasta (exact sequence given to the model)
"""
import os, sys, time
import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, "/home/azureuser/Evo2/SAE")
import run_col4_sae as base

HERE = "/home/azureuser/Evo2/SAE/261126"
OUT_DIR = os.path.join(HERE, "analysis_data")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_CHUNK = base.MAX_CHUNK


def read_fasta(path):
    header, seq = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            header = line
        elif line:
            seq.append(line)
    return header, "".join(seq).upper()


def write_fasta(path, header, seq, width=70):
    with open(path, "w") as f:
        f.write(header + "\n")
        for i in range(0, len(seq), width):
            f.write(seq[i:i+width] + "\n")


def build_human_inputs():
    """Fetch all 4 human regions fresh from Ensembl (guaranteed + strand),
    then revcomp only the minus-strand genes."""
    regions = {
        "COL1A1":  dict(chrom="17", start=50184101, end=50201631, strand="-",
                        refseq="NM_000088.4"),
        "SFTPD":   dict(chrom="10", start=79937740, end=79949105, strand="-",
                        refseq="NM_003019.5"),
        "SRSF1":   dict(chrom="17", start=58000919, end=58007246, strand="-",
                        refseq="NM_006924.5"),
        "KHDRBS1": dict(chrom="1",  start=32013868, end=32060850, strand="+",
                        refseq="NM_006559.3"),
    }
    out = {}
    for gene, g in regions.items():
        plus_seq = base.fetch_ensembl(g["chrom"], g["start"], g["end"])
        assert len(plus_seq) == g["end"] - g["start"] + 1, \
            f"{gene}: fetched length mismatch"
        seq = base.revcomp(plus_seq) if g["strand"] == "-" else plus_seq
        out[gene] = dict(seq=seq, chrom=f"chr{g['chrom']}", start=g["start"],
                          end=g["end"], strand=g["strand"], refseq=g["refseq"],
                          region=f"chr{g['chrom']}:{g['start']:,}-{g['end']:,}")
        print(f"  {gene}: {len(seq):,} bp  strand={g['strand']}  "
              f"(Ensembl + fetch, revcomp={'yes' if g['strand']=='-' else 'no'})")
    return out


def build_fibh_input():
    path = os.path.join(HERE, "CDS_FibH_NM_001113262.1.fasta.txt")
    header, seq = read_fasta(path)
    print(f"  FibH: {len(seq):,} bp  (NCBI RefSeq CDS, NM_001113262.1, used as-is)")
    if "Y" in seq:
        print(f"    NOTE: contains {seq.count('Y')} IUPAC ambiguity code(s) "
              f"(Y) inherited verbatim from the RefSeq record, left unmodified.")
    return dict(seq=seq, header=header, accession="NM_001113262.1")


def compute_and_save(name, seq, evo, sae, meta, log_lines):
    L = len(seq)
    method = "single_chunk" if L <= MAX_CHUNK else (
        "2_chunk_overlap" if L <= 2 * MAX_CHUNK else "nonoverlap_chunks")
    print(f"=== {name}  {L:,} bp  method={method} ===")
    log_lines.append(f"=== {name}  {L:,} bp  method={method} ===")

    t0 = time.time()
    feats = base.get_features(seq, evo, sae)
    dt = time.time() - t0
    Lf, F = feats.shape
    rows, cols = np.nonzero(feats)
    vals = feats[rows, cols].astype(np.float32)
    nnz = len(vals)

    npz_kwargs = dict(
        rows=rows.astype(np.int32), cols=cols.astype(np.int32), vals=vals,
        shape=np.array([Lf, F], dtype=np.int64),
        gene=name, seq_len=Lf, n_features=F, topk=64, layer=base.SAE_LAYER,
        method=method,
    )
    npz_kwargs.update(meta)

    out_npz = os.path.join(OUT_DIR, f"{name}_sae_features.npz")
    np.savez_compressed(out_npz, **npz_kwargs)
    size_mb = os.path.getsize(out_npz) / 1e6
    msg = (f"  {Lf:,} x {F:,}  nnz={nnz:,} ({nnz/Lf:.1f}/pos)  {dt:.1f}s "
           f"-> {os.path.basename(out_npz)}  [{size_mb:.1f} MB]")
    print(msg)
    log_lines.append(msg)

    del feats
    torch.cuda.empty_cache()
    return out_npz


def main():
    log_lines = []
    print("Building input sequences ...")
    human = build_human_inputs()
    fibh = build_fibh_input()

    # save exact input FASTAs used
    for gene, d in human.items():
        write_fasta(os.path.join(OUT_DIR, f"{gene}_input_sequence.fasta"),
                    f">{gene} {d['region']} strand={d['strand']} {d['refseq']} "
                    f"(gene-direction 5'->3', single revcomp={'yes' if d['strand']=='-' else 'no'})",
                    d["seq"])
    write_fasta(os.path.join(OUT_DIR, "FibH_input_sequence.fasta"),
                f">FibH NM_001113262.1 complete CDS (as-is, 5'->3', no revcomp)",
                fibh["seq"])

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

    for gene, d in human.items():
        meta = dict(chrom=d["chrom"], start=d["start"], end=d["end"],
                    strand=d["strand"], region=d["region"])
        compute_and_save(gene, d["seq"], evo, sae, meta, log_lines)

    meta = dict(chrom="N/A (Bombyx mori, GenBank)", start=1, end=len(fibh["seq"]),
                strand="N/A", region=f"NM_001113262.1:1-{len(fibh['seq'])} (CDS)")
    compute_and_save("FibH", fibh["seq"], evo, sae, meta, log_lines)

    with open(os.path.join(OUT_DIR, "run_261126_sae.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
        f.write(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
