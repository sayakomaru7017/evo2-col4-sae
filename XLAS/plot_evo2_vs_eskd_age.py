"""Scatter plots: Evo2 delta score vs age at ESKD onset, per subgroup."""
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXCLUDE_IDS = {"⑥-A507", "⑧-DE-031", "⑬-Family_35", "⑮-Patient_18"}
AA3TO1 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H",
          "Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W",
          "Tyr":"Y","Val":"V"}

def wt_residue_1letter(p_hgvs):
    if not isinstance(p_hgvs, str): return None
    s = p_hgvs.replace("p.", "").replace("(", "").replace(")", "")
    m = re.match(r"([A-Za-z]{3})(\d+)", s)
    if m: return AA3TO1.get(m.group(1))
    m = re.match(r"([A-Za-z])(\d+)", s)
    if m: return m.group(1).upper()
    return None

df = pd.read_csv("/home/azureuser/Evo2/XLAS/xlas_cox_evo2_7b_delta.csv")
df = df[~df["Analysis_ID"].isin(EXCLUDE_IDS)].copy()
scored = df[df["evo2_delta_score"].notna()].copy()
scored["wt_residue"] = scored["p.HGVS_current"].apply(wt_residue_1letter)

groups = {
    "All COL4A5 variants": scored,
    "Truncating only": scored[scored["Truncating_status"] == "Truncating"],
    "Non-truncating only": scored[scored["Truncating_status"] == "Non-truncating"],
    "Missense only": scored[scored["Functional_variant_type"] == "Missense"],
    "Gly-substitution only": scored[(scored["Functional_variant_type"] == "Missense") & (scored["wt_residue"] == "G")],
}

COL_POINT = "#2a78d6"
COL_LINE = "#e34948"

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, (label, sub) in zip(axes, groups.items()):
    d = sub[(sub["ESKD_event"] == 1) & sub["ESKD_age_years"].notna() & sub["evo2_delta_score"].notna()]
    x, y = d["evo2_delta_score"].values, d["ESKD_age_years"].values
    ax.scatter(x, y, s=18, color=COL_POINT, alpha=0.65, edgecolor="none")
    if len(d) >= 4:
        pear = stats.pearsonr(x, y)
        spear = stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=COL_LINE, lw=2)
        ax.set_title(f"{label}  (n={len(d)})\nPearson r={pear.statistic:+.3f} p={pear.pvalue:.3f}  |  "
                     f"Spearman ρ={spear.statistic:+.3f} p={spear.pvalue:.3f}", fontsize=9.5)
    else:
        ax.set_title(f"{label}  (n={len(d)}, too few)", fontsize=9.5)
    ax.set_xlabel("Evo2 delta-likelihood score", fontsize=9)
    ax.set_ylabel("Age at ESKD (years)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

axes[-1].axis("off")
fig.suptitle("Evo2 (evo2_7b) delta-likelihood score vs age at ESKD onset -- XLAS COL4A5 cohort\n"
             "(4 flagged-discrepancy variants excluded; event-only patients, n per panel)",
             fontsize=12.5, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = "/home/azureuser/Evo2/XLAS/evo2_delta_vs_eskd_age.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
