"""Stripplot of Evo2 delta-likelihood scores for COL4A5 variants, by mut_class."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("XLAS/COL4A5_Evo2_delta_scores.csv")
df = df[df["evo2_delta_score"].notna()]

order = df.groupby("mut_class")["evo2_delta_score"].median().sort_values().index.tolist()

plt.figure(figsize=(7, 4))
p = sns.stripplot(
    data=df, x="evo2_delta_score", y="mut_class", hue="mut_class",
    order=order, palette="Set2", size=3, jitter=0.3, legend=False,
)
sns.boxplot(
    x="evo2_delta_score", y="mut_class", data=df, order=order,
    showmeans=True, meanline=True, meanprops={"visible": False},
    medianprops={"color": "k", "ls": "-", "lw": 2},
    whiskerprops={"visible": False}, showfliers=False, showbox=False,
    showcaps=False, ax=p, zorder=10,
)
plt.xlabel("Evo2 delta likelihood score (var - ref, mean log-lik)")
plt.ylabel("COL4A5 mut_class")
plt.title(f"COL4A5 variants (n={len(df)}) -- Evo2 zero-shot delta score, BRCA1-notebook method")
plt.tight_layout()
plt.savefig("XLAS/COL4A5_delta_score_by_class.png", dpi=150, bbox_inches="tight")
print("Saved: XLAS/COL4A5_delta_score_by_class.png")
