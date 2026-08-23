"""
Truncating vs Non-truncating group comparison, and decomposition of the
overall Evo2-score vs ESKD-age association: is it explained by the
truncating/non-truncating dichotomy alone, or is there a residual
within-type gradient?

1. Group comparison: ESKD_age_years (event==1) by Truncating_status
   -- Mann-Whitney U, t-test
2. Group comparison: evo2_delta_score (all scored) by Truncating_status
   -- Mann-Whitney U, t-test
3. Univariate Cox: Truncating_status alone predicting time-to-ESKD
4. Multivariable Cox: evo2_delta_score + Truncating_status together
   -- does evo2_delta_score's effect survive adjustment for group?
   -- does Truncating_status remain significant after adjusting for score?
"""
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter

EXCLUDE_IDS = {"⑥-A507", "⑧-DE-031", "⑬-Family_35", "⑮-Patient_18"}

df = pd.read_csv("/home/azureuser/Evo2/XLAS/xlas_cox_evo2_7b_delta.csv")
df = df[~df["Analysis_ID"].isin(EXCLUDE_IDS)].copy()
scored = df[df["evo2_delta_score"].notna()].copy()
scored = scored[scored["Truncating_status"].isin(["Truncating", "Non-truncating"])].copy()
scored["is_truncating"] = (scored["Truncating_status"] == "Truncating").astype(int)
print(f"n = {len(scored)}  (Truncating={scored['is_truncating'].sum()}, "
      f"Non-truncating={(scored['is_truncating']==0).sum()})")

# --- 1. ESKD age by group (event==1 only) ---
age_t = scored[(scored["is_truncating"] == 1) & (scored["ESKD_event"] == 1)]["ESKD_age_years"].dropna()
age_nt = scored[(scored["is_truncating"] == 0) & (scored["ESKD_event"] == 1)]["ESKD_age_years"].dropna()
print(f"\n=== 1. ESKD age at onset: Truncating (n={len(age_t)}) vs Non-truncating (n={len(age_nt)}) ===")
print(f"  Truncating:     mean={age_t.mean():.1f}  median={age_t.median():.1f}")
print(f"  Non-truncating: mean={age_nt.mean():.1f}  median={age_nt.median():.1f}")
u = stats.mannwhitneyu(age_t, age_nt)
t = stats.ttest_ind(age_t, age_nt, equal_var=False)
print(f"  Mann-Whitney U p={u.pvalue:.5f}   Welch t-test p={t.pvalue:.5f}")

# --- 2. evo2_delta_score by group (all scored) ---
sc_t = scored[scored["is_truncating"] == 1]["evo2_delta_score"]
sc_nt = scored[scored["is_truncating"] == 0]["evo2_delta_score"]
print(f"\n=== 2. Evo2 delta-score: Truncating (n={len(sc_t)}) vs Non-truncating (n={len(sc_nt)}) ===")
print(f"  Truncating:     mean={sc_t.mean():.5f}  median={sc_t.median():.5f}")
print(f"  Non-truncating: mean={sc_nt.mean():.5f}  median={sc_nt.median():.5f}")
u2 = stats.mannwhitneyu(sc_t, sc_nt)
t2 = stats.ttest_ind(sc_t, sc_nt, equal_var=False)
print(f"  Mann-Whitney U p={u2.pvalue:.2e}   Welch t-test p={t2.pvalue:.2e}")

# --- 3. Univariate Cox: Truncating_status alone ---
d = scored[scored["Analysis_time_years"].notna() & (scored["Analysis_time_years"] > 0)].copy()
print(f"\n=== 3. Univariate Cox: Truncating_status alone (n={len(d)}) ===")
cph1 = CoxPHFitter()
cph1.fit(d[["Analysis_time_years", "ESKD_event", "is_truncating"]],
         duration_col="Analysis_time_years", event_col="ESKD_event")
s1 = cph1.summary
print(f"  HR (Truncating vs Non-truncating) = {s1.loc['is_truncating','exp(coef)']:.3f}  "
      f"(95% CI {s1.loc['is_truncating','exp(coef) lower 95%']:.3f}-{s1.loc['is_truncating','exp(coef) upper 95%']:.3f})  "
      f"p={s1.loc['is_truncating','p']:.2e}")

# --- 4. Multivariable Cox: score + group together ---
d["delta_score_x100"] = d["evo2_delta_score"] * 100
print(f"\n=== 4. Multivariable Cox: evo2_delta_score + Truncating_status together (n={len(d)}) ===")
cph2 = CoxPHFitter()
cph2.fit(d[["Analysis_time_years", "ESKD_event", "delta_score_x100", "is_truncating"]],
         duration_col="Analysis_time_years", event_col="ESKD_event")
s2 = cph2.summary
print(s2[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_string())
print(f"\n  evo2_delta_score HR/+0.01 (adjusted for group) = {s2.loc['delta_score_x100','exp(coef)']:.3f}  "
      f"p={s2.loc['delta_score_x100','p']:.4f}")
print(f"  Truncating HR (adjusted for score)              = {s2.loc['is_truncating','exp(coef)']:.3f}  "
      f"p={s2.loc['is_truncating','p']:.4f}")

# --- 5. Within-group correlation reminder + between-group effect size comparison ---
print("\n=== 5. Interpretation aid: variance decomposition ===")
overall_var = scored["evo2_delta_score"].var()
between_group_component = (
    scored.groupby("is_truncating")["evo2_delta_score"].apply(lambda x: len(x) * (x.mean() - scored["evo2_delta_score"].mean())**2).sum()
    / len(scored)
)
print(f"  evo2_delta_score total variance = {overall_var:.7f}")
print(f"  between-group (truncating vs non) component = {between_group_component:.7f}  "
      f"({100*between_group_component/overall_var:.1f}% of total variance)")
