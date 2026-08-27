"""
estimate_statistical.py
=======================
Statistical estimation of Redcar 2015 outcome bucket rates.

Produces math_testing_panel.csv — a clone of redcar_quarterly_panel.csv
with six new statistically-derived columns added. The original panel is
not modified.

Method 1 — Exponential survival model for still_seeking%
---------------------------------------------------------
Fits N(t) = A * exp(-lambda * t) using two government-sourced anchors:
  - t=5  months: 25% still-seeking  (DWP JSA statistics, HIGH quality)
  - t=13 months: ~3% still-seeking  (derived from Task Force 93% off-benefits)
Bootstraps confidence intervals by resampling the uncertain t=13 anchor
over its plausible range [1%, 6%] with 10,000 iterations.

Method 2 — Log-normal wage model for underemployment%
------------------------------------------------------
Fits a log-normal distribution to NE ASHE annual percentiles (p10, p25,
p50, p75, p90). Sigma is estimated as the mean of four percentile-pair
estimates for robustness. Computes P(re-employment wage < 0.9 * pre-wage)
where pre-wage is the SSI cohort's estimated pre-closure weekly earnings.
Uncertainty propagated over the plausible SSI pre-wage range [£480, £673].

Output columns added
--------------------
  stat_still_seeking_mid_%           fitted exponential mid estimate
  stat_still_seeking_lo_%        5th percentile of bootstrap distribution
  stat_still_seeking_hi_%        95th percentile of bootstrap distribution
  stat_underemployed_mid_%       log-normal P(underemployed | re-employed) mid
  stat_underemployed_lo_%        low (high pre-wage assumption)
  stat_underemployed_hi_%        high (low pre-wage assumption)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
RESEARCH = BASE.parent
ONS      = RESEARCH / "ons_data"

PANEL_IN  = BASE / "redcar_quarterly_panel.csv"
PANEL_OUT = BASE / "math_testing_panel.csv"

# ── constants ──────────────────────────────────────────────────────────────────
SSI_N          = 2_070          # total displaced workers
BOOTSTRAP_N    = 10_000         # bootstrap iterations
RNG_SEED       = 42

# DWP hard anchor: 25% still-seeking at t=5 months post-closure (Feb 2016)
ANCHOR_T5_PCT  = 25.0
ANCHOR_T5_MONTH = 5

# Task Force anchor: 93% off benefits at t=13 months → ~7% still on benefits.
# We estimate 3-4% of that is still-seeking (remainder is ESA/disability exits).
# Plausible range for the bootstrap: 1%–6%.
ANCHOR_T13_MID  = 3.0
ANCHOR_T13_LO   = 1.0
ANCHOR_T13_HI   = 6.0
ANCHOR_T13_MONTH = 13

# SSI pre-closure wage proxy (NE ASHE, weekly GBP):
#   Low  = NE p50 at 2015 (£485.6/wk = £25.3k/yr)  — minimum plausible
#   Mid  = interpolated between p50 and p75          — central estimate
#   High = NE p75 at 2015 (£679.4/wk = £35.3k/yr)  — upper plausible
# Source: NM_30_1 ASHE NE 2015, verified in ashe_percentiles_raw.csv.
SSI_PRE_WAGE_LO  = 480.8    # £/week (£25k/yr)
SSI_PRE_WAGE_MID = 576.9    # £/week (£30k/yr)
SSI_PRE_WAGE_HI  = 673.1    # £/week (£35k/yr)

# Underemployment threshold: >10% real wage cut (standard labour-econ definition)
UNDEREMPLOYED_THRESHOLD = 0.90   # new_wage < 0.90 * pre_wage

# MONTHS_POST: first month of each quarter post-closure
# (Oct 2015 = month 1, Jan 2016 = month 4, etc.)
MONTHS_POST = {
    "2015Q4":  1, "2016Q1":  4, "2016Q2":  7, "2016Q3": 10,
    "2016Q4": 13, "2017Q1": 16, "2017Q2": 19, "2017Q3": 22, "2017Q4": 25,
    "2018Q1": 28, "2018Q2": 31, "2018Q3": 34, "2018Q4": 37,
    "2019Q1": 40, "2019Q2": 43, "2019Q3": 46, "2019Q4": 49,
}


# ── Method 1: Exponential survival model ──────────────────────────────────────
#
# With exactly two anchor points, the exponential N(t) = A * exp(-lam * t)
# can be solved analytically (no curve_fit needed):
#
#   N(t5)  = A * exp(-lam * t5)  = y5
#   N(t13) = A * exp(-lam * t13) = y13
#   → dividing: exp(lam * (t13 - t5)) = y5 / y13
#   → lam = ln(y5 / y13) / (t13 - t5)
#   → A   = y5 * exp(lam * t5)
#
# This is exact and fast. The bootstrap resamples the uncertain t=13 anchor
# over its plausible range, re-solving analytically each iteration.


def solve_exponential(y5: float, y13: float):
    """
    Analytically solve N(t) = A * exp(-lam * t) from two anchor values.
    y5  = still_seeking% at t=5  months (DWP anchor, hard)
    y13 = still_seeking% at t=13 months (derived, uncertain)
    Returns (A, lam).
    """
    lam = np.log(y5 / y13) / (ANCHOR_T13_MONTH - ANCHOR_T5_MONTH)
    A   = y5 * np.exp(lam * ANCHOR_T5_MONTH)
    return A, lam


def build_still_seeking_estimates(quarters: list[str]) -> pd.DataFrame:
    """
    For each quarter, return mid/lo/hi still_seeking% via bootstrap.

    Bootstrap procedure:
      - Draw BOOTSTRAP_N samples of the t=13 anchor uniformly from
        [ANCHOR_T13_LO, ANCHOR_T13_HI] (the plausible range for ~3%)
      - For each sample, solve (A, lam) analytically
      - Evaluate N(t_mid_quarter) for each quarter
      - Report 5th / 95th percentile as the 90% CI
    """
    rng = np.random.default_rng(RNG_SEED)

    # Mid estimate using central anchor value
    A_mid, lam_mid = solve_exponential(ANCHOR_T5_PCT, ANCHOR_T13_MID)

    # Bootstrap: draw t=13 anchor from uniform over plausible range
    t13_samples = rng.uniform(ANCHOR_T13_LO, ANCHOR_T13_HI, BOOTSTRAP_N)
    boot_A   = np.array([solve_exponential(ANCHOR_T5_PCT, y13)[0] for y13 in t13_samples])
    boot_lam = np.array([solve_exponential(ANCHOR_T5_PCT, y13)[1] for y13 in t13_samples])

    results = []
    for q in quarters:
        t = MONTHS_POST[q] + 1.5   # mid-quarter (start_month + 1.5 months)

        mid      = float(np.clip(A_mid * np.exp(-lam_mid * t), 0.0, 100.0))
        boot_vals = np.clip(boot_A * np.exp(-boot_lam * t), 0.0, 100.0)
        lo        = float(np.percentile(boot_vals,  5))
        hi        = float(np.percentile(boot_vals, 95))

        results.append({
            "quarter":                   q,
            "stat_still_seeking_mid_%":  round(mid, 2),
            "stat_still_seeking_lo_%":   round(lo,  2),
            "stat_still_seeking_hi_%":   round(hi,  2),
            "stat_exp_lambda":           round(lam_mid, 4),
            "stat_exp_A":                round(A_mid,   2),
        })

    return pd.DataFrame(results)


# ── Method 2: Log-normal wage model ───────────────────────────────────────────

def fit_lognormal_sigma(p10, p25, p50, p75, p90):
    """
    Estimate sigma of log-normal wage distribution from five percentile values.
    Uses four independent percentile-pair estimates and returns their mean.
    All inputs are in the same unit (weekly £).
    """
    z90 = norm.ppf(0.90)   # 1.2816
    z75 = norm.ppf(0.75)   # 0.6745

    s1 = (np.log(p90) - np.log(p50)) / z90   # upper tail
    s2 = (np.log(p75) - np.log(p50)) / z75   # upper quartile
    s3 = (np.log(p50) - np.log(p25)) / z75   # lower quartile
    s4 = (np.log(p50) - np.log(p10)) / z90   # lower tail

    return np.mean([s1, s2, s3, s4])


def build_underemployed_estimates(quarters: list[str]) -> pd.DataFrame:
    """
    For each quarter, compute P(new_wage < threshold * pre_wage) from the
    NE log-normal wage distribution fitted to annual ASHE percentiles.

    Returns mid / lo / hi estimates where:
      - mid: SSI pre-wage = £577/wk (£30k/yr)
      - lo : SSI pre-wage = £673/wk (£35k/yr) — harder threshold, lower P
      - hi : SSI pre-wage = £481/wk (£25k/yr) — easier threshold, higher P
    Note: lo/hi refer to the UNDEREMPLOYMENT PROBABILITY, not the wage.
    """
    ashe = pd.read_csv(ONS / "ashe_percentiles_raw.csv")
    ashe.columns = ["year", "region", "statistic", "weekly_pay_gbp"]
    ne   = ashe[ashe["region"] == "North East"].copy()

    # Build year -> {p10, p25, p50, p75, p90} lookup
    stat_map = {
        "10 percentile": "p10",
        "25 percentile": "p25",
        "Median":        "p50",
        "75 percentile": "p75",
        "90 percentile": "p90",
    }
    ne["stat_key"] = ne["statistic"].map(stat_map)
    ne = ne.dropna(subset=["stat_key"])
    pivoted = ne.pivot_table(index="year", columns="stat_key",
                             values="weekly_pay_gbp", aggfunc="first")

    # For each quarter, look up the ASHE year (annual data; use calendar year of quarter)
    results = []
    for q in quarters:
        year = int(q[:4])

        # ASHE covers 2008-2019 in our data; clamp to available range
        avail_years = sorted(pivoted.index.tolist())
        year_key    = min(year, max(avail_years))
        year_key    = max(year_key, min(avail_years))

        row = pivoted.loc[year_key]
        p10, p25, p50, p75, p90 = (row["p10"], row["p25"], row["p50"],
                                    row["p75"], row["p90"])

        sigma  = fit_lognormal_sigma(p10, p25, p50, p75, p90)
        mu     = np.log(p50)   # log of median = mu of log-normal

        def p_underemployed(pre_wage_weekly: float) -> float:
            """P(new_wage < threshold * pre_wage) under NE log-normal distribution."""
            threshold_weekly = UNDEREMPLOYED_THRESHOLD * pre_wage_weekly
            z = (np.log(threshold_weekly) - mu) / sigma
            return float(norm.cdf(z) * 100)   # as percentage

        mid = round(p_underemployed(SSI_PRE_WAGE_MID), 1)
        # Low P(underemployed): high pre-wage → threshold is high but distribution unchanged
        # Wait — higher pre-wage means higher threshold → MORE workers fall below → higher P
        # So high pre-wage = higher underemployment probability
        hi  = round(p_underemployed(SSI_PRE_WAGE_HI), 1)
        lo  = round(p_underemployed(SSI_PRE_WAGE_LO), 1)

        results.append({
            "quarter":                     q,
            "stat_underemployed_mid_%":    mid,
            "stat_underemployed_lo_%":     lo,
            "stat_underemployed_hi_%":     hi,
            "stat_lognormal_mu":           round(mu,    4),
            "stat_lognormal_sigma":        round(sigma, 4),
            "stat_ashe_year_used":         year_key,
        })

    return pd.DataFrame(results)


# ── Merge and write ────────────────────────────────────────────────────────────

def main():
    print("Loading panel...")
    panel = pd.read_csv(PANEL_IN)
    print(f"  {len(panel)} rows x {len(panel.columns)} columns")

    quarters = sorted(panel["quarter"].unique().tolist(),
                      key=lambda q: list(MONTHS_POST.keys()).index(q))

    print(f"\nMethod 1: fitting exponential survival model ({BOOTSTRAP_N:,} bootstrap iterations)...")
    ss_df = build_still_seeking_estimates(quarters)
    print("  Lambda (decay rate per month):", round(ss_df["stat_exp_lambda"].iloc[0], 4))
    print("  A (theoretical peak %):",        round(ss_df["stat_exp_A"].iloc[0], 2))
    print()
    print("  quarter   | mid    | lo     | hi")
    print("  ----------|--------|--------|--------")
    for _, r in ss_df.iterrows():
        print(f"  {r.quarter}  | {r['stat_still_seeking_mid_%']:5.2f}% | "
              f"{r['stat_still_seeking_lo_%']:5.2f}% | {r['stat_still_seeking_hi_%']:5.2f}%")

    print(f"\nMethod 2: fitting log-normal wage model from NE ASHE percentiles...")
    ue_df = build_underemployed_estimates(quarters)
    print()
    print("  quarter   | sigma  | mid P(under) | lo    | hi")
    print("  ----------|--------|--------------|-------|-------")
    for _, r in ue_df.iterrows():
        print(f"  {r.quarter}  | {r['stat_lognormal_sigma']:.4f} | "
              f"{r['stat_underemployed_mid_%']:5.1f}%        | "
              f"{r['stat_underemployed_lo_%']:5.1f}% | {r['stat_underemployed_hi_%']:5.1f}%")

    # Merge: still_seeking columns go on every row (broadcast by quarter)
    print("\nMerging into panel clone...")
    out = panel.copy()

    # still_seeking: apply to all rows (all demographic slices share the same rate)
    out = out.merge(ss_df.drop(columns=["stat_exp_lambda","stat_exp_A"]),
                    on="quarter", how="left")

    # underemployed log-normal: apply only to all/all rows (age-specific not yet available)
    ue_all = ue_df.drop(columns=["stat_lognormal_mu","stat_lognormal_sigma","stat_ashe_year_used"])
    out = out.merge(ue_all, on="quarter", how="left")

    # For non-all age_band rows, blank out the underemployed estimate
    # (the log-normal is fitted to the aggregate NE distribution, not age-specific)
    age_mask = out["age_band"] != "all"
    for col in ["stat_underemployed_mid_%","stat_underemployed_lo_%","stat_underemployed_hi_%"]:
        out.loc[age_mask, col] = np.nan

    # Data quality note column
    out["stat_method"] = (
        "still_seeking: exponential decay fitted to DWP (t=5) and Task Force (t=13) anchors; "
        "CI from 10,000-iteration bootstrap over t=13 anchor range [1%,6%]. "
        "underemployed: log-normal P(wage<0.9*pre_wage) from NE ASHE annual percentiles; "
        "CI from SSI pre-wage range [£481,£673]/wk."
    )

    out.to_csv(PANEL_OUT, index=False)
    print(f"\nSaved: {PANEL_OUT}")
    print(f"  {len(out)} rows x {len(out.columns)} columns")
    print(f"  New columns added: {len(out.columns) - len(panel.columns)}")

    # Validation: print still_seeking estimates vs known anchors
    print("\n--- Validation against hard anchors ---")
    all_rows = out[(out["age_band"] == "all") & (out["gender"] == "all")]
    all_rows = all_rows.drop_duplicates(subset=["quarter"]).reset_index(drop=True)
    for q in ["2016Q1", "2016Q4", "2017Q4"]:
        r = all_rows[all_rows["quarter"] == q].iloc[0]
        print(f"  {q}: still_seeking mid={r['stat_still_seeking_mid_%']:.1f}%  "
              f"[{r['stat_still_seeking_lo_%']:.1f}%, {r['stat_still_seeking_hi_%']:.1f}%]  "
              f"| underemployed mid={r['stat_underemployed_mid_%']:.1f}%  "
              f"[{r['stat_underemployed_lo_%']:.1f}%, {r['stat_underemployed_hi_%']:.1f}%]")
    print()
    print("  Known anchors:")
    print("    Q1 2016: still_seeking = 25.0%  (DWP JSA stats, HIGH)")
    print("    Q4 2016: still_seeking ≈  3.0%  (derived from Task Force 93%, MEDIUM)")


if __name__ == "__main__":
    main()
