"""Boxplots: ESKD age and Evo2 delta score, Truncating vs Non-truncating."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

EXCLUDE_IDS = {"⑥-A507", "⑧-DE-031", "⑬-Family_35", "⑮-Patient_18"}
df = pd.read_csv("/home/azureuser/Evo2/XLAS/xlas_cox_evo2_7b_delta.csv")
df = df[~df["Analysis_ID"].isin(EXCLUDE_IDS)].copy()
scored = df[df["evo2_delta_score"].notna()].copy()
scored = scored[scored["Truncating_status"].isin(["Truncating", "Non-truncating"])].copy()

COL_T = "#e34948"
COL_NT = "#2a78d6"
order = ["Truncating", "Non-truncating"]
pal = {"Truncating": COL_T, "Non-truncating": COL_NT}

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

age_d = scored[scored["ESKD_event"] == 1]
sns.boxplot(data=age_d, x="Truncating_status", y="ESKD_age_years", order=order,
            palette=pal, showfliers=False, width=0.5, ax=axes[0])
sns.stripplot(data=age_d, x="Truncating_status", y="ESKD_age_years", order=order,
              color="black", size=3, alpha=0.4, jitter=0.25, ax=axes[0])
axes[0].set_title(f"Age at ESKD onset\n(Mann-Whitney p<0.00001)", fontsize=10.5)
axes[0].set_xlabel("")
axes[0].set_ylabel("Age at ESKD (years)")
axes[0].spines[["top", "right"]].set_visible(False)

sns.boxplot(data=scored, x="Truncating_status", y="evo2_delta_score", order=order,
            palette=pal, showfliers=False, width=0.5, ax=axes[1])
sns.stripplot(data=scored, x="Truncating_status", y="evo2_delta_score", order=order,
              color="black", size=3, alpha=0.4, jitter=0.25, ax=axes[1])
axes[1].set_title(f"Evo2 delta-likelihood score\n(Mann-Whitney p=3.6e-6)", fontsize=10.5)
axes[1].set_xlabel("")
axes[1].set_ylabel("Evo2 delta-likelihood score")
axes[1].spines[["top", "right"]].set_visible(False)

fig.suptitle("Truncating vs Non-truncating: both ESKD age and Evo2 score differ significantly",
             fontsize=12.5, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = "/home/azureuser/Evo2/XLAS/truncating_vs_nontruncating_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
