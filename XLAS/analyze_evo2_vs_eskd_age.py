"""
Correlation / Cox-regression analysis: Evo2 delta-likelihood score vs
age at ESKD (dialysis) onset, for the XLAS COL4A5 Cox cohort.

Excludes the 4 variants flagged with a genuine protein-annotation
discrepancy in xlas_coordinate_verification.csv review
(⑥-A507, ⑧-DE-031, ⑬-Family_35, ⑮-Patient_18).

Groups analyzed:
  1. All scored COL4A5 variants
  2. Truncating only / Non-truncating only (Truncating_status column)
  3. Missense only (Functional_variant_type == "Missense")
  4. Gly-substitution missense only (p.HGVS wild-type residue == Gly)

For each group:
  - Pearson & Spearman correlation (evo2_delta_score vs ESKD_age_years),
    restricted to patients who reached ESKD (ESKD_event==1) -- direct
    answer to "is delta score correlated with age at ESKD".
  - Cox proportional-hazards regression (evo2_delta_score as covariate,
    time=Analysis_time_years, event=ESKD_event) using ALL scored patients
    in the group (including right-censored, i.e. not yet at ESKD) -- the
    statistically appropriate method given the censored cohort structure.
"""
import re
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter

EXCLUDE_IDS = {"⑥-A507", "⑧-DE-031", "⑬-Family_35", "⑮-Patient_18"}

AA3TO1 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H",
          "Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W",
          "Tyr":"Y","Val":"V"}

def wt_residue_1letter(p_hgvs):
    if not isinstance(p_hgvs, str):
        return None
    s = p_hgvs.replace("p.", "").replace("(", "").replace(")", "")
    m = re.match(r"([A-Za-z]{3})(\d+)", s)
    if m:
        return AA3TO1.get(m.group(1))
    m = re.match(r"([A-Za-z])(\d+)", s)
    if m:
        return m.group(1).upper()
    return None


def corr_report(sub, label):
    d = sub[(sub["ESKD_event"] == 1) & sub["ESKD_age_years"].notna() & sub["evo2_delta_score"].notna()]
    print(f"\n--- {label}: simple correlation (ESKD_event==1 only) ---")
    print(f"  n = {len(d)}")
    if len(d) < 4:
        print("  too few events for correlation")
        return
    pear = stats.pearsonr(d["evo2_delta_score"], d["ESKD_age_years"])
    spear = stats.spearmanr(d["evo2_delta_score"], d["ESKD_age_years"])
    print(f"  Pearson  r={pear.statistic:+.3f}  p={pear.pvalue:.4f}")
    print(f"  Spearman rho={spear.statistic:+.3f}  p={spear.pvalue:.4f}")


def cox_report(sub, label):
    d = sub[sub["evo2_delta_score"].notna() & sub["Analysis_time_years"].notna() & sub["ESKD_event"].notna()].copy()
    d = d[d["Analysis_time_years"] > 0]
    print(f"\n--- {label}: Cox proportional-hazards (all scored incl. censored) ---")
    print(f"  n = {len(d)}  (events={int(d['ESKD_event'].sum())}, censored={int((d['ESKD_event']==0).sum())})")
    if len(d) < 10 or d["ESKD_event"].sum() < 5:
        print("  too few events for Cox regression")
        return
    # rescale: raw delta scores are ~ -0.03..0, "per unit" HRs are numerically
    # degenerate; report HR per 0.01-unit change instead (x100 covariate).
    d["delta_score_x100"] = d["evo2_delta_score"] * 100
    cph = CoxPHFitter()
    try:
        cph.fit(d[["Analysis_time_years", "ESKD_event", "delta_score_x100"]],
                duration_col="Analysis_time_years", event_col="ESKD_event")
        s = cph.summary
        hr = s.loc["delta_score_x100", "exp(coef)"]
        p = s.loc["delta_score_x100", "p"]
        ci_lo = s.loc["delta_score_x100", "exp(coef) lower 95%"]
        ci_hi = s.loc["delta_score_x100", "exp(coef) upper 95%"]
        print(f"  HR per +0.01 increase in delta-score = {hr:.3f}  (95% CI {ci_lo:.3f}-{ci_hi:.3f})  p={p:.4f}")
        print(f"  (delta-score becoming MORE negative/disruptive by 0.01 -> HR x {1/hr:.3f} for reaching ESKD sooner)")
    except Exception as e:
        print(f"  Cox fit failed: {e}")


def main():
    df = pd.read_csv("/home/azureuser/Evo2/XLAS/xlas_cox_evo2_7b_delta.csv")
    before = len(df)
    df = df[~df["Analysis_ID"].isin(EXCLUDE_IDS)].copy()
    print(f"Excluded {before - len(df)} rows: {EXCLUDE_IDS}")
    scored = df[df["evo2_delta_score"].notna()].copy()
    print(f"Scored rows remaining: {len(scored)}")

    # --- 1. All variants ---
    corr_report(scored, "ALL COL4A5 variants")
    cox_report(scored, "ALL COL4A5 variants")

    # --- 2. Truncating vs Non-truncating ---
    trunc = scored[scored["Truncating_status"] == "Truncating"]
    nontrunc = scored[scored["Truncating_status"] == "Non-truncating"]
    print(f"\n[Truncating_status basis: n_Truncating={len(trunc)}, n_Non-truncating={len(nontrunc)}, "
          f"excluded Unknown/See-paper/NaN={len(scored)-len(trunc)-len(nontrunc)}]")
    corr_report(trunc, "Truncating only")
    cox_report(trunc, "Truncating only")
    corr_report(nontrunc, "Non-truncating only")
    cox_report(nontrunc, "Non-truncating only")

    # --- 3. Missense only ---
    missense = scored[scored["Functional_variant_type"] == "Missense"].copy()
    corr_report(missense, "Missense only")
    cox_report(missense, "Missense only")

    # --- 4. Gly-substitution missense only ---
    missense["wt_residue"] = missense["p.HGVS_current"].apply(wt_residue_1letter)
    gly = missense[missense["wt_residue"] == "G"].copy()
    print(f"\n[Gly-substitution basis: {len(gly)}/{len(missense)} missense variants have wild-type residue = Gly]")
    corr_report(gly, "Gly-substitution missense only")
    cox_report(gly, "Gly-substitution missense only")

    scored.to_csv("/home/azureuser/Evo2/XLAS/xlas_cox_evo2_7b_delta_excl4.csv", index=False)
    print("\nSaved cleaned dataset: XLAS/xlas_cox_evo2_7b_delta_excl4.csv")


if __name__ == "__main__":
    main()
