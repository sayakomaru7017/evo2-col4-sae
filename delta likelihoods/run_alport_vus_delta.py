"""
Alport COL4A3/4/5 ClinVar VUS / Conflicting SNV delta-likelihood scoring with Evo2-7B.
Same pipeline as run_alport_delta.py (window=8192bp).
"""
import gzip
import os
import time
import numpy as np
import pandas as pd
from Bio import SeqIO

XLSX = "/home/azureuser/evo/alport/VUS_Conflicting_COL4A345.xlsx"
FASTA = {
    "2": "/home/azureuser/evo/alport/GRCh38.p14_chr2.fna.gz",
    "X": "/home/azureuser/evo/alport/GRCh38.p14_chrX.fna.gz",
}
WINDOW = 8192
OUT_CSV = "/home/azureuser/evo/alport/alport_vus_evo2_delta.csv"

# ---------- FASTA ----------
print("Loading reference FASTA...")
chrom_seq = {}
for chrom, path in FASTA.items():
    with gzip.open(path, "rt") as h:
        rec = next(SeqIO.parse(h, "fasta"))
        chrom_seq[chrom] = str(rec.seq).upper()
    print(f"  chr{chrom}: {len(chrom_seq[chrom]):,} bp  ({rec.id})")

# ---------- Load sheets ----------
print("\nLoading ClinVar xlsx...")
vus_df = pd.read_excel(XLSX, sheet_name="VUS(1124)")
con_df = pd.read_excel(XLSX, sheet_name="Conflicting(586)")
vus_df["label"] = "VUS"
con_df["label"] = "Conflicting"
df = pd.concat([vus_df, con_df], ignore_index=True)
print(f"  total rows: {len(df)}  (VUS={(df.label=='VUS').sum()}, Conflicting={(df.label=='Conflicting').sum()})")

# ---------- Filter to SNVs ----------
df = df[df["Variant type"] == "single nucleotide variant"].copy()
df = df[df["Canonical SPDI"].notna()].copy()
print(f"  SNVs with SPDI: {len(df)}")

parts = df["Canonical SPDI"].str.split(":")
df["spdi_acc"] = parts.str[0]
df["pos0"] = parts.str[1].astype(int)
df["ref"]  = parts.str[2]
df["alt"]  = parts.str[3]
df["chrom"] = df["GRCh38Chromosome"].astype(str)

df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)].copy()
df = df[df["chrom"].isin(["2", "X"])].copy()
df = df[df["ref"].isin(list("ACGT")) & df["alt"].isin(list("ACGT"))].copy()
df = df.drop_duplicates(subset=["chrom", "pos0", "ref", "alt"]).reset_index(drop=True)
df["pos0"] = df["pos0"].astype(int)
print(f"  after dedup, valid SNVs: {len(df)}  (VUS={(df.label=='VUS').sum()}, Conflicting={(df.label=='Conflicting').sum()})")

# ---------- Window extraction ----------
def extract(chrom, pos0, ref, alt):
    seq = chrom_seq[str(chrom)]
    half = WINDOW // 2
    pos0 = int(pos0)
    if pos0 < 0 or pos0 >= len(seq):
        return None, None, f"oob:pos={pos0}/len={len(seq)}"
    start = max(0, pos0 - half)
    end = min(len(seq), pos0 + half)
    ref_w = seq[start:end]
    snv_pos = pos0 - start
    if snv_pos < 0 or snv_pos >= len(ref_w):
        return None, None, f"snv_offset_bad"
    if ref_w[snv_pos] != ref:
        return None, None, f"ref_mismatch:{ref_w[snv_pos]}!={ref}@pos0={pos0}"
    var_w = ref_w[:snv_pos] + alt + ref_w[snv_pos + 1 :]
    return ref_w, var_w, None

ref_seqs = []
ref_to_idx = {}
ref_idx_per_var = []
var_seqs = []
keep_idx = []
bad = 0
for i, row in df.iterrows():
    rs, vs, err = extract(row["chrom"], row["pos0"], row["ref"], row["alt"])
    if err is not None:
        bad += 1
        continue
    if rs not in ref_to_idx:
        ref_to_idx[rs] = len(ref_seqs)
        ref_seqs.append(rs)
    ref_idx_per_var.append(ref_to_idx[rs])
    var_seqs.append(vs)
    keep_idx.append(i)

df = df.loc[keep_idx].reset_index(drop=True)
ref_idx_per_var = np.asarray(ref_idx_per_var)
print(f"  ref-mismatch dropped: {bad}")
print(f"  unique ref windows: {len(ref_seqs)} | variant windows: {len(var_seqs)}")

# ---------- Score with Evo2-7B ----------
import torch
from evo2 import Evo2

print("\nLoading Evo2-7B...")
t0 = time.time()
model = Evo2("evo2_7b")
print(f"  loaded in {time.time()-t0:.1f}s")

def score(seqs, label):
    print(f"\nScoring {len(seqs)} {label} sequences...")
    t0 = time.time()
    scores = model.score_sequences(seqs, batch_size=1)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s  ({dt/max(1,len(seqs)):.2f}s/seq)")
    return np.asarray(scores, dtype=np.float64)

ref_scores = score(ref_seqs, "ref")
var_scores = score(var_seqs, "var")

delta = var_scores - ref_scores[ref_idx_per_var]
df["ref_ll"] = ref_scores[ref_idx_per_var]
df["var_ll"] = var_scores
df["evo2_delta_score"] = delta

# ---------- Save ----------
out_cols = [
    "Name", "Gene(s)", "GRCh38Chromosome", "GRCh38Location",
    "ref", "alt", "Variant type", "Molecular consequence",
    "Germline classification", "label",
    "ref_ll", "var_ll", "evo2_delta_score",
]
df[out_cols].to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# ---------- Summary stats ----------
print("\n=== Δ-likelihood summary per group ===")
for g in ["VUS", "Conflicting"]:
    sub = df[df["label"] == g]
    if len(sub):
        d = sub["evo2_delta_score"]
        print(f"  {g}: n={len(sub)}, mean={d.mean():.5f}, median={d.median():.5f}, "
              f"min={d.min():.5f}, max={d.max():.5f}")

print("\n=== Per-gene Δ summary ===")
for g in ["COL4A3", "COL4A4", "COL4A5"]:
    sub = df[df["Gene(s)"].str.contains(g, na=False)]
    if len(sub):
        d = sub["evo2_delta_score"]
        print(f"  {g}: n={len(sub)}, mean={d.mean():.5f}, median={d.median():.5f}")

print(f"\nPeak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
