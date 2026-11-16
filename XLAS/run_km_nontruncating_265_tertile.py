"""
Kaplan-Meier survival analysis (ESKD-free survival vs age) for the
Non-truncating cohort (n=265, ESKD events=120) from
XLAS/○手作業最終盤.xlsx ("all症例" sheet), stratified by tertile of
the Evo2 delta-likelihood score (Evo2score column).

Same 265-patient cohort used for the earlier Cox regression
(HR 1.043 per 0.001 increase, 95% CI 0.966-1.126, p=0.28) and Spearman
correlation (rho=0.003, p=0.980) results.

Event definition:
    event = 1 if ESKD_age_years is present (reached ESKD)
    event = 0 if only Age_without_ESKD_years is present (censored)
    time  = ESKD_age_years if event else Age_without_ESKD_years

Tertile definition (pd.qcut, ascending by Evo2score):
    T1 = lowest Evo2score (most negative / most damaging)
    T2 = middle third
    T3 = highest Evo2score (closest to zero or positive / most benign)

Usage
-----
    cd /home/azureuser/Evo2
    source .venv/bin/activate
    python XLAS/run_km_nontruncating_265_tertile.py

Output
------
    XLAS/km_nontruncating_265_tertile.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(HERE, "○手作業最終盤.xlsx")
SHEET = "all症例"
OUT_PNG = os.path.join(HERE, "km_nontruncating_265_tertile.png")


def load_nontruncating_cohort():
    wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
    ws = wb[SHEET]
    headers = [c.value for c in ws[1]]
    data = [row for row in ws.iter_rows(min_row=2, values_only=True)]
    df = pd.DataFrame(data, columns=headers)

    nt = df[df["Truncating_status"] == "Non-truncating"].copy()
    nt["Evo2score"] = pd.to_numeric(nt["Evo2score"], errors="coerce")
    nt["event"] = nt["ESKD_age_years"].notna().astype(int)
    nt["time"] = nt["ESKD_age_years"].where(nt["event"] == 1, nt["Age_without_ESKD_years"])
    nt["time"] = pd.to_numeric(nt["time"], errors="coerce")
    return nt


def main():
    nt = load_nontruncating_cohort()
    print(f"Non-truncating cohort: n={len(nt)}, events={nt['event'].sum()}")
    assert nt["time"].isna().sum() == 0, "missing survival time"

    nt["tertile"] = pd.qcut(
        nt["Evo2score"], 3,
        labels=["T1 (low score / more damaging)", "T2 (mid)", "T3 (high score / more benign)"],
    )
    print(nt.groupby("tertile", observed=True)["Evo2score"].agg(["min", "max", "mean", "count"]))
    print(nt.groupby("tertile", observed=True)["event"].sum())

    res = multivariate_logrank_test(nt["time"], nt["tertile"], nt["event"])
    print(f"\nmultivariate log-rank (3 groups): chi2={res.test_statistic:.3f}  p={res.p_value:.4f}")

    pw = pairwise_logrank_test(nt["time"], nt["tertile"], nt["event"])
    print("\npairwise log-rank p-values:")
    print(pw.summary[["p"]])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {
        "T1 (low score / more damaging)": "#d62728",
        "T2 (mid)": "#ff7f0e",
        "T3 (high score / more benign)": "#2ca02c",
    }
    for tert, sub in nt.groupby("tertile", observed=True):
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], event_observed=sub["event"],
                label=f"{tert}  n={len(sub)}, events={sub['event'].sum()}")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors.get(tert))

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("ESKD-free survival probability")
    ax.set_title(
        f"Non-truncating (n=265): Kaplan-Meier by Evo2 delta-likelihood tertile\n"
        f"T1 = lowest score (most damaging)  →  T3 = highest score (most benign)\n"
        f"log-rank p={res.p_value:.3f}"
    )
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
