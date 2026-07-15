"""
Visualize COL4A1-6 gene structure + feature 11876 activation.
All coordinates in transcript orientation (5'→3'), so minus-strand genes
are displayed with 5' end on the left.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FEAT_ID = 11876
OUT_DIR = "SAE/features"
SMOOTH  = 300

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
exon_data  = pickle.load(open("/tmp/col4_exons.pkl","rb"))

EXON_COL  = "#4393c3"
RANGE_COL = "#d73027"
ACT_COL   = "#2166ac"
NC1_COL   = "#f4a582"

def transcript_coords(g, exons_genomic):
    """Convert genomic exon coords to transcript (5'→3') coords (in bp from 5' end)."""
    gs, ge = g["gene_start"], g["gene_end"]
    if g["strand"] == "+":
        return [(s - gs, e - gs) for s, e in exons_genomic]
    else:
        return [(ge - e, ge - s) for s, e in exons_genomic]

def range_in_transcript(g):
    """Return analysis range in transcript coordinates."""
    gs, ge = g["gene_start"], g["gene_end"]
    rs, re = g["range_start"], g["range_end"]
    if g["strand"] == "+":
        return rs - gs, re - gs
    else:
        return ge - re, ge - rs

fig, axes = plt.subplots(
    nrows=12, ncols=1, figsize=(12, 22),
    gridspec_kw={"height_ratios": [0.55, 1.45] * 6, "hspace": 0.35}
)
fig.suptitle(f"Feature {FEAT_ID} — gene structure & activation  (COL4A1–COL4A6)\n"
             f"All genes shown 5'→3' (transcript orientation)",
             fontsize=11, fontweight="bold")

for i, gene in enumerate(GENE_ORDER):
    ax_g = axes[i * 2]
    ax_a = axes[i * 2 + 1]

    g  = GENES[gene]
    gs, ge = g["gene_start"], g["gene_end"]
    L  = ge - gs + 1

    ed   = exon_data[gene]
    exons_g = sorted([(s, e) for s, e in ed["exons"]])
    exons_t = transcript_coords(g, exons_g)   # transcript coords
    exons_t = sorted(exons_t)

    # range in transcript coords
    r_lo, r_hi = range_in_transcript(g)
    r_lo_kb = r_lo / 1e3
    r_hi_kb = r_hi / 1e3

    # feature activation (already in transcript orientation)
    d     = np.load(f"{OUT_DIR}/{gene}_sae_features.npz")
    rows_ = d["rows"].astype(np.int32)
    cols_ = d["cols"].astype(np.int32)
    vals_ = d["vals"].astype(np.float32)
    Lf    = int(d["shape"][0])
    dense = np.zeros(Lf, dtype=np.float32)
    mask  = cols_ == FEAT_ID
    dense[rows_[mask]] = vals_[mask]
    kern  = np.ones(SMOOTH) / SMOOTH
    smooth = np.convolve(dense, kern, mode="same")

    x_kb = np.arange(Lf) / 1e3   # transcript position in kb

    # ── Gene diagram ─────────────────────────────────────────────────────────
    ax_g.set_xlim(0, L / 1e3)
    ax_g.set_ylim(0, 1)
    ax_g.axis("off")

    # Intron backbone
    ax_g.plot([0, L/1e3], [0.5, 0.5], color="#888", lw=1.2, zorder=1)

    # Exons (transcript coords)
    for es, ee in exons_t:
        w = (ee - es) / 1e3
        r = mpatches.Rectangle((es/1e3, 0.18), max(w, 0.3), 0.64,
                                fc=EXON_COL, ec="none", zorder=2, alpha=0.85)
        ax_g.add_patch(r)

    # Last exon in transcript = NC1 domain (rightmost in exons_t)
    nc1_es, nc1_ee = exons_t[-1]
    r_nc1 = mpatches.Rectangle((nc1_es/1e3, 0.18), max((nc1_ee-nc1_es)/1e3, 0.3), 0.64,
                                fc=NC1_COL, ec="none", zorder=3, alpha=0.95)
    ax_g.add_patch(r_nc1)
    ax_g.text((nc1_es + nc1_ee) / 2e3, 0.96, "NC1",
              ha="center", va="top", fontsize=6.5,
              color="#b2182b", fontweight="bold")

    # Direction arrow (always pointing right = 5'→3')
    arr_mid = L / 2e3
    dx = L / 20e3
    ax_g.annotate("", xy=(arr_mid + dx, 0.5), xytext=(arr_mid, 0.5),
                  arrowprops=dict(arrowstyle="->", color="#555", lw=1.0))

    # Analysis range
    ax_g.axvspan(r_lo_kb, r_hi_kb, ymin=0, ymax=1,
                 color=RANGE_COL, alpha=0.22, zorder=4)
    ax_g.axvline(r_lo_kb, color=RANGE_COL, lw=1.1, ls="--", zorder=5)
    ax_g.axvline(r_hi_kb, color=RANGE_COL, lw=1.1, ls="--", zorder=5)

    rlen = g["range_end"] - g["range_start"] + 1
    strand_sym = "+" if g["strand"] == "+" else "−"
    ax_g.set_title(
        f"{gene}  {g['chrom']}:{gs:,}–{ge:,}  ({strand_sym})  "
        f"| range {g['range_start']:,}–{g['range_end']:,} ({rlen:,} bp)",
        fontsize=8.5, loc="left", pad=2, fontweight="bold"
    )

    # ── Activation ───────────────────────────────────────────────────────────
    ax_a.set_xlim(0, L / 1e3)
    ax_a.fill_between(x_kb, smooth, color=ACT_COL, alpha=0.75, lw=0)
    ax_a.axvspan(r_lo_kb, r_hi_kb, color=RANGE_COL, alpha=0.18, zorder=3)
    ax_a.axvline(r_lo_kb, color=RANGE_COL, lw=1.1, ls="--", zorder=4)
    ax_a.axvline(r_hi_kb, color=RANGE_COL, lw=1.1, ls="--", zorder=4)

    n_in = int((dense[r_lo:r_hi+1] > 0).sum())
    ymax = smooth.max() if smooth.max() > 0 else 1e-6
    ax_a.text((r_lo_kb + r_hi_kb) / 2, ymax * 1.08, f"n={n_in}",
              ha="center", va="bottom", fontsize=7.5,
              color=RANGE_COL, fontweight="bold")

    ax_a.set_ylabel("activation", fontsize=7)
    ax_a.tick_params(labelsize=6.5)
    ax_a.spines[["top","right"]].set_visible(False)
    if i < 5:
        ax_a.set_xticklabels([])
    else:
        ax_a.set_xlabel("Transcript position (kb from 5' end)", fontsize=9)

# Legend
handles = [
    mpatches.Patch(color=EXON_COL,  alpha=0.85, label="Exon"),
    mpatches.Patch(color=NC1_COL,   alpha=0.95, label="Last exon / NC1 domain"),
    mpatches.Patch(color=RANGE_COL, alpha=0.25, label="Specified analysis range"),
    mpatches.Patch(color=ACT_COL,   alpha=0.75, label=f"Feature {FEAT_ID} ({SMOOTH}-bp avg)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           fontsize=8.5, bbox_to_anchor=(0.5, 0.0), frameon=True)

out = f"SAE/features/feature_{FEAT_ID}_gene_structure.png"
fig.savefig(out, dpi=110, facecolor="white")
import os; print(f"Saved {out}  ({os.path.getsize(out)//1024} KB)")
