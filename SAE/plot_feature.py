"""Plot feature 11876 activation across COL4A1-6, with specified ranges highlighted."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FEAT_ID = 11876
OUT_DIR = "SAE/features"
SMOOTH_WIN = 500  # bp smoothing window for display

GENES = {
    "COL4A1": {"gene_start":110_148_963,"gene_end":110_307_202,"strand":"-",
                "range_start":110_150_366,"range_end":110_162_359,"chrom":"chr13"},
    "COL4A2": {"gene_start":110_305_812,"gene_end":110_513_209,"strand":"+",
                "range_start":110_506_477,"range_end":110_512_188,"chrom":"chr13"},
    "COL4A3": {"gene_start":227_164_624,"gene_end":227_314_792,"strand":"+",
                "range_start":227_307_790,"range_end":227_311_864,"chrom":"chr2"},
    "COL4A4": {"gene_start":226_967_360,"gene_end":227_164_488,"strand":"-",
                "range_start":227_007_328,"range_end":227_010_442,"chrom":"chr2"},
    "COL4A5": {"gene_start":108_439_745,"gene_end":108_697_547,"strand":"+",
                "range_start":108_687_565,"range_end":108_696_375,"chrom":"chrX"},
    "COL4A6": {"gene_start":108_155_607,"gene_end":108_439_497,"strand":"-",
                "range_start":108_157_003,"range_end":108_160_592,"chrom":"chrX"},
}
GENE_ORDER = ["COL4A1","COL4A2","COL4A3","COL4A4","COL4A5","COL4A6"]

def arr_range(g):
    gs, ge, st = g["gene_start"], g["gene_end"], g["strand"]
    rs, re = g["range_start"], g["range_end"]
    return (rs-gs, re-gs+1) if st=="+" else (ge-re, ge-rs+1)

def arr_to_genomic(arr_pos, g):
    if g["strand"] == "+":
        return g["gene_start"] + arr_pos
    else:
        return g["gene_end"] - arr_pos

fig, axes = plt.subplots(6, 1, figsize=(14, 16))
fig.suptitle(f"SAE Feature {FEAT_ID} — activation across COL4A1–COL4A6\n"
             f"(shaded = specified analysis range)", fontsize=13, fontweight="bold")

for ax, gene in zip(axes, GENE_ORDER):
    g = GENES[gene]
    d = np.load(f"{OUT_DIR}/{gene}_sae_features.npz")
    rows = d["rows"].astype(np.int32)
    cols = d["cols"].astype(np.int32)
    vals = d["vals"].astype(np.float32)
    L    = int(d["shape"][0])

    # Extract feature activations into dense array
    feat_mask = cols == FEAT_ID
    dense = np.zeros(L, dtype=np.float32)
    dense[rows[feat_mask]] = vals[feat_mask]

    # Genomic x-axis positions (in Mb)
    x_arr = np.arange(L)
    x_genomic = arr_to_genomic(x_arr, g) / 1e6  # Mb

    # Smoothed signal for visibility
    kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    smooth = np.convolve(dense, kernel, mode="same")

    ax.fill_between(x_genomic, smooth, alpha=0.7, color="#2166ac", linewidth=0)
    ax.plot(x_genomic, smooth, color="#2166ac", linewidth=0.4, alpha=0.5)

    # Highlight specified range
    lo, hi = arr_range(g)
    x_lo = arr_to_genomic(lo, g) / 1e6
    x_hi = arr_to_genomic(hi - 1, g) / 1e6
    if x_lo > x_hi:
        x_lo, x_hi = x_hi, x_lo
    ax.axvspan(x_lo, x_hi, color="#d73027", alpha=0.25, zorder=3)
    ax.axvline(x_lo, color="#d73027", linewidth=1.2, linestyle="--", zorder=4)
    ax.axvline(x_hi, color="#d73027", linewidth=1.2, linestyle="--", zorder=4)

    # Labels
    strand_sym = "→" if g["strand"] == "+" else "←"
    range_len = g["range_end"] - g["range_start"] + 1
    ax.set_ylabel(f"activation\n({SMOOTH_WIN}-bp avg)", fontsize=7)
    ax.set_title(
        f"{gene}  {g['chrom']}:{g['gene_start']:,}–{g['gene_end']:,}  {g['strand']}  "
        f"[range: {g['range_start']:,}–{g['range_end']:,}, {range_len:,} bp]",
        fontsize=9, loc="left", pad=3
    )
    ax.tick_params(labelsize=7)
    ax.set_xlim(x_genomic[0], x_genomic[-1])
    ax.spines[["top","right"]].set_visible(False)

    # Annotate range with n_in / max value
    lo2, hi2 = (lo, hi) if lo < hi else (hi, lo)
    in_region = dense[lo2:hi2]
    n_pos = int((in_region > 0).sum())
    max_act = float(in_region.max()) if in_region.size > 0 else 0
    mid_x = (x_lo + x_hi) / 2
    ymax = smooth.max() if smooth.max() > 0 else 1
    ax.text(mid_x, ymax * 1.05, f"n={n_pos}", ha="center", va="bottom",
            fontsize=7, color="#d73027", fontweight="bold")

axes[-1].set_xlabel("Genomic position (Mb)", fontsize=9)

# Legend
blue_patch = mpatches.Patch(color="#2166ac", alpha=0.7, label=f"Feature {FEAT_ID} activation ({SMOOTH_WIN}-bp avg)")
red_patch  = mpatches.Patch(color="#d73027", alpha=0.3, label="Specified analysis range")
fig.legend(handles=[blue_patch, red_patch], loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, 0.00))

plt.tight_layout(rect=[0, 0.03, 1, 1])
out_path = f"SAE/features/feature_{FEAT_ID}_pattern.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
