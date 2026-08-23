import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

CSV = "/home/azureuser/evo/alport/alport_evo2_delta.csv"
OUT = "/home/azureuser/evo/alport/alport_roc.png"

df = pd.read_csv(CSV)
y = (df["label"] == "P").astype(int).values
score = -df["evo2_delta_score"].values

fig, ax = plt.subplots(figsize=(6, 6), dpi=140)

# Overall
fpr, tpr, _ = roc_curve(y, score)
auc = roc_auc_score(y, score)
ax.plot(fpr, tpr, lw=2.5, color="#1a4480",
        label=f"All COL4A3/4/5  (n={len(df)}, AUROC={auc:.4f})")

# Per gene
colors = {"COL4A3": "#d62728", "COL4A4": "#2ca02c", "COL4A5": "#9467bd"}
for g, c in colors.items():
    sub = df[df["Gene(s)"].str.contains(g, na=False)]
    yy = (sub["label"] == "P").astype(int).values
    if len(set(yy)) != 2:
        continue
    ss = -sub["evo2_delta_score"].values
    fpr, tpr, _ = roc_curve(yy, ss)
    a = roc_auc_score(yy, ss)
    ax.plot(fpr, tpr, lw=1.5, color=c, alpha=0.85,
            label=f"{g}  (n={len(sub)}, AUROC={a:.4f})")

ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Evo2-7B zero-shot Δ-likelihood: Alport COL4A3/4/5\nClinVar Pathogenic vs Benign (window = 8192 bp)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT)
print(f"Saved: {OUT}")

# Also plot delta-likelihood distributions (stripplot-like)
import seaborn as sns
fig, ax = plt.subplots(figsize=(7, 3.5), dpi=140)
sns.boxplot(data=df, x="evo2_delta_score", y="label", order=["B", "P"],
            palette={"B": "#888", "P": "#d62728"}, showfliers=False, width=0.5, ax=ax)
sns.stripplot(data=df, x="evo2_delta_score", y="label", order=["B", "P"],
              palette={"B": "#444", "P": "#a01818"}, size=1.8, alpha=0.5, jitter=0.3, ax=ax)
ax.axvline(0, ls="--", color="k", lw=0.8)
ax.set_xlabel("Δ log-likelihood (var − ref)")
ax.set_ylabel("ClinVar label")
ax.set_title("Distribution of Evo2 Δ-likelihood by ClinVar label")
plt.tight_layout()
plt.savefig("/home/azureuser/evo/alport/alport_delta_dist.png")
print("Saved: /home/azureuser/evo/alport/alport_delta_dist.png")
