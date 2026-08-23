"""
Plot f20693 (top NC1-domain-specific SAE feature, found by searching for
features that fire in NC1-domain exons but rarely in introns or the
triple-helical collagenous exons) across full-length COL4A3/4/5, with
NC1 domain and exon structure overlaid.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

FEAT_ID = 20693
OUT_DIR = "SAE/features"
GENES = ["COL4A3", "COL4A4", "COL4A5"]

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


def to_genomic_range(local_lo, local_hi, gene_start, gene_end, strand):
    if strand == "+":
        return gene_start + local_lo, gene_start + local_hi
    else:
        return gene_end - local_hi, gene_end - local_lo


fig, axes = plt.subplots(len(GENES), 1, figsize=(14, 9))
fig.suptitle(f"f{FEAT_ID}: candidate NC1-domain-specific SAE feature\n"
             "(fires in NC1 exons, rarely in introns or triple-helical exons)",
             fontsize=12.5, fontweight="bold")

for ax, gene in zip(axes, GENES):
    npz = np.load(f"{OUT_DIR}/{gene}_sae_features.npz")
    start, end, strand = int(npz["start"]), int(npz["end"]), str(npz["strand"])
    L = int(npz["shape"][0])
    d = dense_for(npz, FEAT_ID)

    x_arr = np.arange(L)
    x_genomic = (start + x_arr if strand == "+" else end - x_arr) / 1e6
    order = np.argsort(x_genomic)
    ax.plot(x_genomic[order], d[order], color="#2a78d6", lw=0.6)

    # shade exons lightly, NC1 exons in red
    for (es, ee) in exon_data[gene]:
        ax.axvspan(es/1e6, ee/1e6, color="#888888", alpha=0.12, lw=0)
    for (ns, ne) in NC1_SEGMENTS[gene]:
        ax.axvspan(ns/1e6, ne/1e6, color="#e34948", alpha=0.35, lw=0)

    strand_sym = "->" if strand == "+" else "<-"
    ax.set_title(f"{gene}  chr{npz['chrom']}:{start:,}-{end:,}  {strand_sym}  "
                 f"(gray=exon, red=NC1 domain exon)", fontsize=9.5, loc="left")
    ax.set_ylabel(f"f{FEAT_ID}\nactivation", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel("Genomic position (Mb, GRCh38)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = f"{OUT_DIR}/f{FEAT_ID}_nc1_domain_specificity.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
