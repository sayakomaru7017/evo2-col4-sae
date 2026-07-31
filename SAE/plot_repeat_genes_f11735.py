"""
Plot ONLY f11735 activation across the full length of the 9 repeat-expansion /
periodic-motif genes: HTT, FMR1, DMPK, FXN, CNBP, C9orf72, LORICRIN, ELN, TPM1.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR    = "SAE/features"
FEAT_ID    = 11735
GENES      = ["HTT", "FMR1", "DMPK", "FXN", "CNBP", "C9orf72", "LORICRIN", "ELN", "TPM1"]
SMOOTH_WIN = 200
COLOR      = "#2a78d6"

ANNOT = {
    "HTT":      "CAG (polyQ), exon1",
    "FMR1":     "CGG, 5'UTR",
    "DMPK":     "CTG, 3'UTR",
    "FXN":      "GAA, intron1",
    "CNBP":     "CCTG, intron1",
    "C9orf72":  "GGGGCC, intron1",
    "LORICRIN": "Gly-loop, coding",
    "ELN":      "VPGVG/Gly-rich, coding",
    "TPM1":     "heptad (coiled-coil), coding",
}

def dense_for(npz, feat_id):
    rows, cols, vals = npz["rows"], npz["cols"], npz["vals"]
    d = np.zeros(int(npz["shape"][0]), dtype=np.float32)
    mask = cols == feat_id
    d[rows[mask]] = vals[mask]
    return d

fig, axes = plt.subplots(len(GENES), 1, figsize=(13, 2.3 * len(GENES)))
fig.suptitle(f"f{FEAT_ID} activation across repeat-expansion / periodic-motif genes (wild-type, GRCh38)",
             fontsize=12.5, fontweight="bold")

print(f"=== f{FEAT_ID} summary ===")
for ax, gene in zip(axes, GENES):
    npz = np.load(f"{OUT_DIR}/{gene}_sae_features.npz")
    start, end, strand = int(npz["start"]), int(npz["end"]), str(npz["strand"])
    L = int(npz["shape"][0])

    d = dense_for(npz, FEAT_ID)
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    smooth = np.convolve(d, kernel, mode="same")
    x_arr = np.arange(L)
    x_genomic = (start + x_arr if strand == "+" else end - x_arr) / 1e6
    order = np.argsort(x_genomic)
    ax.plot(x_genomic[order], smooth[order], color=COLOR, lw=1.1)
    ax.fill_between(x_genomic[order], smooth[order], color=COLOR, alpha=0.25, linewidth=0)

    strand_sym = "->" if strand == "+" else "<-"
    n = int((d > 0).sum())
    mx = float(d.max()) if n else 0.0
    ax.set_title(f"{gene}  chr{npz['chrom']}:{start:,}-{end:,}  {strand_sym}  ({L:,} bp)"
                 f"   [{ANNOT.get(gene,'')}]   n_active={n} max={mx:.2f}",
                 fontsize=9, loc="left")
    ax.set_ylabel(f"f{FEAT_ID}\n({SMOOTH_WIN}-bp avg)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    print(f"{gene:9s} n_active={n:4d}  max={mx:.3f}")

axes[-1].set_xlabel("Genomic position (Mb, GRCh38)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"{OUT_DIR}/repeat_genes_f11735.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
