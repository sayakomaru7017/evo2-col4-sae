"""
Plot f11735, f9170, f5026 activation across the full length of
LAMB2, NID1, NPHS1 (GBM / slit-diaphragm genes).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR   = "SAE/features"
FEATURES  = [11735, 9170, 5026]
COLORS    = {11735: "#2a78d6", 9170: "#eb6834", 5026: "#1baf7a"}
GENES     = ["LAMB2", "NID1", "NPHS1", "APOL1", "NPHS2"]
SMOOTH_WIN = 200

def dense_for(npz, feat_id):
    rows, cols, vals = npz["rows"], npz["cols"], npz["vals"]
    d = np.zeros(int(npz["shape"][0]), dtype=np.float32)
    mask = cols == feat_id
    d[rows[mask]] = vals[mask]
    return d

fig, axes = plt.subplots(len(GENES), 1, figsize=(13, 2.5 * len(GENES)))
fig.suptitle("f11735 / f9170 / f5026 activation across GBM/podocyte genes (wild-type, GRCh38)",
             fontsize=12.5, fontweight="bold")

for ax, gene in zip(axes, GENES):
    npz = np.load(f"{OUT_DIR}/{gene}_sae_features.npz")
    start, end, strand = int(npz["start"]), int(npz["end"]), str(npz["strand"])
    L = int(npz["shape"][0])

    for feat_id in FEATURES:
        d = dense_for(npz, feat_id)
        kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
        smooth = np.convolve(d, kernel, mode="same")
        # array index 0 = 5' end of the transcribed (strand-corrected) sequence
        x_arr = np.arange(L)
        x_genomic = (start + x_arr if strand == "+" else end - x_arr) / 1e6
        order = np.argsort(x_genomic)
        ax.plot(x_genomic[order], smooth[order], color=COLORS[feat_id], lw=1.1,
                label=f"f{feat_id}", alpha=0.85)

    strand_sym = "->" if strand == "+" else "<-"
    ax.set_title(f"{gene}  chr{npz['chrom']}:{start:,}-{end:,}  {strand_sym}  ({L:,} bp)",
                 fontsize=9.5, loc="left")
    ax.set_ylabel(f"activation\n({SMOOTH_WIN}-bp avg)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

axes[0].legend(loc="upper right", fontsize=9, frameon=False)
axes[-1].set_xlabel("Genomic position (Mb, GRCh38)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = f"{OUT_DIR}/new_genes_target_features.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

print("\n=== summary ===")
for gene in GENES:
    npz = np.load(f"{OUT_DIR}/{gene}_sae_features.npz")
    for feat_id in FEATURES:
        d = dense_for(npz, feat_id)
        n = int((d > 0).sum())
        mx = float(d.max()) if n else 0.0
        print(f"{gene:6s} f{feat_id}: n_active={n:4d}  max={mx:.3f}")
