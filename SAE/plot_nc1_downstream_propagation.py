"""
f19166 / f26434: NC1-domain-specific features that show reduced activation in
NC1 exons DOWNSTREAM of (and NOT containing) the mutation site -- evidence of
a domain-wide functional-propagation signal rather than local sequence-motif
disruption, since Evo2 is causal and can only propagate mutation effects to
downstream positions.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

data = pickle.load(open("SAE/nc1_downstream_effect.pkl", "rb"))
data = [d for d in data if d[3] is not None]

PATHOGENIC = {"C1616Y","S1492C","L1598R","R1661H","G1508S","C1588S","C1641R","F1650L",
              "M1603T","C1684Y","R1569P","C1638R","C1644G","C1570S"}

FEATS = [19166, 26434]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, feat in zip(axes, FEATS):
    rows = []
    for gene, tag, var_local, wt_sum, relchange, n_down in data:
        cls = "Pathogenic" if tag in PATHOGENIC else "Benign"
        rows.append((f"{gene} {tag}", cls, relchange[feat] * 100))
    rows.sort(key=lambda r: r[2])
    labels = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    colors = ["#e34948" if r[1] == "Pathogenic" else "#2a78d6" for r in rows]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Downstream NC1-exon-sum relative change (%)", fontsize=9.5)
    ax.set_title(f"f{feat}", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#e34948", label="Pathogenic"),
                     Patch(color="#2a78d6", label="Benign")],
           loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=False)
fig.suptitle("Downstream-of-mutation NC1 exon activation change\n"
             "(mutation NOT in this exon -- tests causal propagation, not local motif loss)",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.90])
out = "SAE/features/nc1_downstream_propagation_f19166_f26434.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
