"""Zoomed view of f20693 across just the NC1 domain region (+/-2kb) for each gene."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

FEAT_ID = 20693
OUT_DIR = "SAE/features"
GENES = ["COL4A3", "COL4A4", "COL4A5"]
PAD = 2000

NC1_SEGMENTS = {
    "COL4A3": [(227307790,227307919),(227308899,227309076),(227309204,227309318),
               (227310776,227310948),(227311786,227311867)],
    "COL4A4": [(227010313,227010442),(227008018,227008304),(227007328,227007588)],
    "COL4A5": [(108687565,108687694),(108692748,108692925),(108694807,108694921),
               (108695267,108695439),(108696297,108696375)],
}
exon_data = pickle.load(open("SAE/col4_exon_structure.pkl", "rb"))


def dense_for(npz, feat_id):
    rows, cols, vals = npz["rows"], npz["cols"], npz["vals"]
    d = np.zeros(int(npz["shape"][0]), dtype=np.float32)
    mask = cols == feat_id
    d[rows[mask]] = vals[mask]
    return d


fig, axes = plt.subplots(len(GENES), 1, figsize=(14, 8))
fig.suptitle(f"f{FEAT_ID} zoomed on the NC1 domain (+/-{PAD} bp), gray=exon red=NC1 exon",
             fontsize=12.5, fontweight="bold")

for ax, gene in zip(axes, GENES):
    npz = np.load(f"{OUT_DIR}/{gene}_sae_features.npz")
    start, end, strand = int(npz["start"]), int(npz["end"]), str(npz["strand"])
    L = int(npz["shape"][0])
    d = dense_for(npz, FEAT_ID)
    x_arr = np.arange(L)
    x_genomic = start + x_arr if strand == "+" else end - x_arr

    nc1_lo = min(s for s, e in NC1_SEGMENTS[gene]) - PAD
    nc1_hi = max(e for s, e in NC1_SEGMENTS[gene]) + PAD
    mask = (x_genomic >= nc1_lo) & (x_genomic <= nc1_hi)
    order = np.argsort(x_genomic[mask])
    xg = x_genomic[mask][order]
    dd = d[mask][order]

    ax.plot(xg, dd, color="#2a78d6", lw=0.8)
    for (es, ee) in exon_data[gene]:
        ax.axvspan(es, ee, color="#888888", alpha=0.15, lw=0)
    for (ns, ne) in NC1_SEGMENTS[gene]:
        ax.axvspan(ns, ne, color="#e34948", alpha=0.35, lw=0)

    ax.set_title(f"{gene}  chr{npz['chrom']}:{nc1_lo:,}-{nc1_hi:,}", fontsize=9.5, loc="left")
    ax.set_ylabel(f"f{FEAT_ID}", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel("Genomic position (GRCh38)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = f"{OUT_DIR}/f{FEAT_ID}_nc1_zoom.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
