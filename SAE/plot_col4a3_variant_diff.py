"""
Show ONLY the difference (mutant - wild-type) in SAE feature activation
for f11735, f9170, f5026 around the COL4A3 p.Gly366Glu variant
(chr2:227,259,860 G>A, c.1097G>A).

Diverging encoding: positive (mutant gains activation) vs negative
(mutant loses activation relative to wild-type), gray zero baseline.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR     = "SAE/features"
FEATURES    = [11735, 9170, 5026]
CTX_WINDOW  = 3000
ZOOM_WINDOW = 40

COL_POS = "#e34948"   # gain in mutant
COL_NEG = "#2a78d6"   # loss in mutant

wt  = np.load(f"{OUT_DIR}/COL4A3_sae_features.npz")
mut = np.load(f"{OUT_DIR}/COL4A3_G366E_mutant_sae_features.npz")

gene_start = int(wt["start"])
L          = int(wt["shape"][0])
var_off    = int(mut["variant_array_offset"])
var_genomic = gene_start + var_off

def dense_for(npz, feat_id):
    rows, cols, vals = npz["rows"], npz["cols"], npz["vals"]
    d = np.zeros(int(npz["shape"][0]), dtype=np.float32)
    mask = cols == feat_id
    d[rows[mask]] = vals[mask]
    return d

fig, axes = plt.subplots(len(FEATURES), 2, figsize=(15, 8.5),
                          gridspec_kw={"width_ratios": [2.2, 1]})
fig.suptitle(
    "COL4A3 SAE feature activation — mutant minus wild-type (diff only)\n"
    "chr2:227,259,860 G>A  (c.1097G>A, p.Gly366Glu)   "
    "red = mutant gains activation, blue = mutant loses activation",
    fontsize=12.5, fontweight="bold"
)

print("=== differences (mutant - wild-type) ===")
for row, feat_id in enumerate(FEATURES):
    ax_ctx, ax_zoom = axes[row]
    diff_full = dense_for(mut, feat_id) - dense_for(wt, feat_id)

    # ---- context panel ----
    lo, hi = max(0, var_off - CTX_WINDOW), min(L, var_off + CTX_WINDOW)
    x = gene_start + np.arange(lo, hi)
    d = diff_full[lo:hi]
    colors = np.where(d >= 0, COL_POS, COL_NEG)
    ax_ctx.bar(x, d, color=colors, width=1.2, linewidth=0)
    ax_ctx.axvline(var_genomic, color="#555555", lw=1, linestyle=":", zorder=1)
    ax_ctx.axhline(0, color="#999999", lw=0.8)
    ax_ctx.set_ylabel(f"f{feat_id}\ndiff", fontsize=10, fontweight="bold")
    ax_ctx.spines[["top", "right"]].set_visible(False)
    ax_ctx.tick_params(labelsize=8)
    ax_ctx.set_title(f"context: +/-{CTX_WINDOW:,} bp", fontsize=8.5, loc="left", color="#666666")

    # ---- zoom panel ----
    zlo, zhi = var_off - ZOOM_WINDOW, var_off + ZOOM_WINDOW + 1
    zx = np.arange(zlo, zhi) - var_off
    zd = diff_full[zlo:zhi]
    zcolors = np.where(zd >= 0, COL_POS, COL_NEG)
    ax_zoom.bar(zx, zd, color=zcolors, width=0.8, linewidth=0)
    ax_zoom.axvline(0, color="#555555", lw=1, linestyle=":", zorder=1)
    ax_zoom.axhline(0, color="#999999", lw=0.8)
    ax_zoom.set_title("zoom: +/-40 bp (base resolution)", fontsize=8.5, loc="left", color="#666666")
    ax_zoom.spines[["top", "right"]].set_visible(False)
    ax_zoom.tick_params(labelsize=8)

    # console table: top differences genome-wide
    nz = np.nonzero(diff_full)[0]
    order = nz[np.argsort(-np.abs(diff_full[nz]))][:8]
    print(f"\nf{feat_id} (n={len(nz)} differing positions genome-wide):")
    print(f"  {'genomic pos':>13}  {'dist_to_variant':>15}  {'wt':>7}  {'mut':>7}  {'diff':>7}")
    for p in order:
        print(f"  {gene_start+p:>13,}  {p-var_off:>+15,}  {dense_for(wt,feat_id)[p]:>7.3f}  "
              f"{dense_for(mut,feat_id)[p]:>7.3f}  {diff_full[p]:>+7.3f}")

axes[-1, 0].set_xlabel("Genomic position (chr2, GRCh38)", fontsize=10)
axes[-1, 1].set_xlabel("bp relative to variant", fontsize=10)

# shared legend
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=COL_POS, label="mutant gains activation"),
                     Patch(color=COL_NEG, label="mutant loses activation")],
           loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01), frameon=False)

plt.tight_layout(rect=[0, 0.02, 1, 0.92])
out_path = f"{OUT_DIR}/col4a3_G366E_variant_features_diff.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"\nSaved: {out_path}")
