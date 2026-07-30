"""
Compare SAE feature firing patterns (f11735, f9170, f5026) between the
wild-type COL4A3 sequence and the p.Gly366Glu mutant (chr2:227,259,860 G>A,
c.1097G>A) around the variant site.

Left column : +/-3 kb context (line trace)
Right column: +/-40 bp base-resolution zoom (stem plot) around the variant
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR    = "SAE/features"
FEATURES   = [11735, 9170, 5026]
CTX_WINDOW = 3000   # bp each side, context panel
ZOOM_WINDOW = 40     # bp each side, base-resolution panel

COL_WT  = "#2a78d6"   # validated categorical slot 1 (blue)
COL_MUT = "#eb6834"   # validated categorical slot 2 (orange)

wt  = np.load(f"{OUT_DIR}/COL4A3_sae_features.npz")
mut = np.load(f"{OUT_DIR}/COL4A3_G366E_mutant_sae_features.npz")

gene_start = int(wt["start"])
L          = int(wt["shape"][0])
var_off    = int(mut["variant_array_offset"])
var_genomic = gene_start + var_off
assert var_off == 227_259_860 - gene_start

def dense_for(npz, feat_id):
    rows, cols, vals = npz["rows"], npz["cols"], npz["vals"]
    d = np.zeros(int(npz["shape"][0]), dtype=np.float32)
    mask = cols == feat_id
    d[rows[mask]] = vals[mask]
    return d

fig, axes = plt.subplots(len(FEATURES), 2, figsize=(15, 9),
                          gridspec_kw={"width_ratios": [2.2, 1]})
fig.suptitle(
    "COL4A3 SAE feature firing: wild-type vs p.Gly366Glu mutant\n"
    "chr2:227,259,860 G>A  (c.1097G>A)",
    fontsize=13, fontweight="bold"
)

for row, feat_id in enumerate(FEATURES):
    ax_ctx, ax_zoom = axes[row]
    d_wt  = dense_for(wt,  feat_id)
    d_mut = dense_for(mut, feat_id)

    # ---- context panel (+/- 3kb) ----
    lo, hi = max(0, var_off - CTX_WINDOW), min(L, var_off + CTX_WINDOW)
    x = gene_start + np.arange(lo, hi)
    ax_ctx.plot(x, d_wt[lo:hi],  color=COL_WT,  lw=1.4, label="wild-type", zorder=3)
    ax_ctx.plot(x, d_mut[lo:hi], color=COL_MUT, lw=1.4, label="mutant (Gly366Glu)",
                zorder=2, alpha=0.9)
    ax_ctx.axvline(var_genomic, color="#555555", lw=1, linestyle=":", zorder=1)
    ax_ctx.set_ylabel(f"f{feat_id}", fontsize=10, fontweight="bold")
    ax_ctx.spines[["top", "right"]].set_visible(False)
    ax_ctx.tick_params(labelsize=8)
    ax_ctx.set_title(f"context: +/-{CTX_WINDOW:,} bp", fontsize=8.5, loc="left", color="#666666")

    # ---- zoom panel (+/- 40bp), base resolution stems ----
    zlo, zhi = var_off - ZOOM_WINDOW, var_off + ZOOM_WINDOW + 1
    zx = np.arange(zlo, zhi) - var_off  # relative bp to variant
    zwt, zmut = d_wt[zlo:zhi], d_mut[zlo:zhi]

    ax_zoom.vlines(zx - 0.15, 0, zwt,  color=COL_WT,  lw=2, zorder=3)
    ax_zoom.vlines(zx + 0.15, 0, zmut, color=COL_MUT, lw=2, zorder=2)
    ax_zoom.axvline(0, color="#555555", lw=1, linestyle=":", zorder=1)
    ax_zoom.set_title("zoom: +/-40 bp (base resolution)", fontsize=8.5, loc="left", color="#666666")
    ax_zoom.spines[["top", "right"]].set_visible(False)
    ax_zoom.tick_params(labelsize=8)

    # direct-label the largest divergence in the zoom window
    diff = np.abs(zmut - zwt)
    if diff.max() > 0:
        pk = np.argmax(diff)
        ax_zoom.annotate(f"{'wt' if zwt[pk]>zmut[pk] else 'mut'} only\n({zx[pk]:+d} bp)",
                          xy=(zx[pk], max(zwt[pk], zmut[pk])),
                          xytext=(0, 10), textcoords="offset points",
                          fontsize=7.5, ha="center", color="#333333")

axes[0, 0].legend(loc="upper right", fontsize=9, frameon=False)
axes[-1, 0].set_xlabel("Genomic position (chr2, GRCh38)", fontsize=10)
axes[-1, 1].set_xlabel("bp relative to variant", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = f"{OUT_DIR}/col4a3_G366E_variant_features_comparison.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"Saved: {out_path}")

print("\nWhole-gene summary (positions where wt and mutant differ):")
for feat_id in FEATURES:
    d_wt, d_mut = dense_for(wt, feat_id), dense_for(mut, feat_id)
    diff = d_mut - d_wt
    nz = np.nonzero(diff)[0]
    at_variant = diff[var_off]
    print(f"  f{feat_id}: {len(nz)} differing positions genome-wide; "
          f"at variant base: wt={d_wt[var_off]:.3f} mut={d_mut[var_off]:.3f} (diff={at_variant:+.3f})")
