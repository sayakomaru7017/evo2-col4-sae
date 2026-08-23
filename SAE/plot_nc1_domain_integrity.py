"""
f28426: NC1-domain-sum relative change (mutant vs WT) for 19 Alport NC1-domain
variants (14 pathogenic, 5 benign controls). Tests whether this feature's
TOTAL activation across the whole NC1 domain drops for pathogenic mutations.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

FEAT_ID = 28426
data = pickle.load(open("SAE/nc1_domain_sum_variant_effects.pkl", "rb"))

PATHOGENIC = {"C1616Y","S1492C","L1598R","R1661H","G1508S","C1588S","C1641R","F1650L",
              "M1603T","C1684Y","R1569P","C1638R","C1644G","C1570S"}

rows = []
for gene, tag, wt_sum, mut_sum, relchange in data:
    cls = "Pathogenic" if tag in PATHOGENIC else "Benign"
    rows.append((f"{gene} {tag}", cls, relchange[FEAT_ID] * 100))

rows.sort(key=lambda r: r[2])
labels = [r[0] for r in rows]
classes = [r[1] for r in rows]
vals = [r[2] for r in rows]
colors = ["#e34948" if c == "Pathogenic" else "#2a78d6" for c in classes]

fig, ax = plt.subplots(figsize=(8, 7))
y_pos = np.arange(len(labels))
ax.barh(y_pos, vals, color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("NC1-domain-sum relative change, mutant vs WT (%)", fontsize=10)
ax.set_title(f"f{FEAT_ID}: NC1-domain-wide activation change\n"
             "14 pathogenic (red) vs 5 benign (blue) COL4A3/4/5 NC1 variants",
             fontsize=11.5, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#e34948", label="Pathogenic"),
                    Patch(color="#2a78d6", label="Benign")],
          loc="lower right", fontsize=9, frameon=False)

plt.tight_layout()
out = f"SAE/features/f{FEAT_ID}_nc1_domain_integrity.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
