"""
estimate_statistical.py
=======================
Statistical estimation of Redcar 2015 SSI displacement outcome rates.
Reads from and writes back to math_testing_panel.csv (updates in place).

Method 1 — Exponential survival model for still_seeking%
---------------------------------------------------------
Fits N(t) = A * exp(-lambda * t) from two government anchors:
  t=5 months  : 25.0% still-seeking  (DWP JSA statistics, HIGH)
  t=13 months : ~3.0% still-seeking  (derived: Task Force 93% off-benefits)
Bootstraps 90% CI by resampling the uncertain t=13 anchor over [1%, 6%]
with 10,000 iterations. Solved analytically each iteration (exact, no curve_fit).

  stat_still_seeking_mid_%   fitted exponential, mid estimate
  stat_still_seeking_lo_%    5th percentile of bootstrap
  stat_still_seeking_hi_%    95th percentile of bootstrap

Method 2 — Log-normal wage model for aggregate underemployment%
---------------------------------------------------------------
Fits LogNormal(mu, sigma) to NE ASHE annual percentiles (p10/p25/p50/p75/p90).
Sigma averaged from four percentile-pair estimates for robustness.
Computes P(re-employment wage < 0.9 * pre_wage) across the SSI pre-wage range.
Applied to all/all rows only (aggregate).

  stat_underemployed_mid_%   P(underemployed) mid (pre-wage = £577/wk)
  stat_underemployed_lo_%    lo (pre-wage = £481/wk, easier threshold)
  stat_underemployed_hi_%    hi (pre-wage = £673/wk, harder threshold)

Method 2b — Age-specific underemployment%
------------------------------------------
Same log-normal model but uses age-band-specific NE ASHE medians and sigmas
(from ashe_age_* panel columns, scaled from national Table 6 ratios).
Pre-shock wage adjusted by age using the same national ASHE age ratios.
Applied to age-specific rows (age_band in 16-24, 25-49, 50+).

  stat_underemployed_16_24_mid_%    P(underemployed | age=16-24) mid
  stat_underemployed_16_24_lo_%
  stat_underemployed_16_24_hi_%
  stat_underemployed_25_49_mid_%
  stat_underemployed_25_49_lo_%
  stat_underemployed_25_49_hi_%
  stat_underemployed_50plus_mid_%
  stat_underemployed_50plus_lo_%
  stat_underemployed_50plus_hi_%

Method 3 — Bayesian Dirichlet-Multinomial for retrain/share_similar/exit split
-------------------------------------------------------------------------------
Models the 3-way outcome split (retrain, share_similar, exit) conditional on
not being still_seeking and not being underemployed. Uses:

  Prior: Dirichlet(alpha) from cross-episode meta-analysis:
    Hartz IV reforms (Germany 2003-2008)    : retrain~30%, similar~40%, exit~30%
    China Shock (US mfg, 2001-2015)         : retrain~20%, similar~30%, exit~50%
    Port Talbot / Wales steel (early 2024)  : retrain~25%, similar~45%, exit~30%
    Prior mean: retrain~25%, similar~40%, exit~35%
    Concentration kappa=20 (moderate confidence)
    alpha_prior = (5, 8, 7)

  Likelihood: soft observations from RC LA outcome estimates in panel
    Uses mean of the 5 scenario rows per quarter (lo through hi)
    Scaled to pseudo-counts: N_resolved * proportion * RC_EVIDENCE_WEIGHT
    RC_EVIDENCE_WEIGHT = 0.3 (derived estimates ≠ primary data)

  Posterior: Dirichlet(alpha_prior + obs)
    Mean = alpha_post / sum(alpha_post)
    90% CI: sampled from Dirichlet posterior (10,000 draws)

Per-quarter absolute estimates:
  remaining(t) = (1 - still_seeking(t)/100) * (1 - P_underemployed/100)
  retrain(t)   = pi_retrain * remaining(t) * 100
  ...etc.

  stat_retrain_mid_%          absolute % of SSI workers who retrained, mid
  stat_retrain_lo_%           5th percentile
  stat_retrain_hi_%           95th percentile
  stat_share_similar_mid_%
  stat_share_similar_lo_%
  stat_share_similar_hi_%
  stat_exit_mid_%
  stat_exit_lo_%
  stat_exit_hi_%
  stat_dirichlet_kappa        posterior concentration (sum of alpha_post)
  stat_dirichlet_pi_retrain   posterior mean split probability for retrain
  stat_dirichlet_pi_similar
  stat_dirichlet_pi_exit
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
RESEARCH   = BASE.parent
ONS        = RESEARCH / "ons_data"
PANEL_PATH = BASE / "math_testing_panel.csv"   # read from AND write to

# ── constants ──────────────────────────────────────────────────────────────────
SSI_N           = 3_500          # total SSI workers affected
BOOTSTRAP_N     = 10_000
RNG_SEED        = 42

# Method 1 anchors
ANCHOR_T5_PCT   = 25.0
ANCHOR_T5_MONTH =  5
ANCHOR_T13_MID  =  3.0
ANCHOR_T13_LO   =  1.0
ANCHOR_T13_HI   =  6.0
ANCHOR_T13_MONTH = 13

# Method 2/2b: SSI pre-closure weekly wage proxy (NE ASHE, full-time)
SSI_PRE_WAGE_LO  = 480.8    # £/wk — NE p50 2015
SSI_PRE_WAGE_MID = 576.9    # £/wk — between p50 and p75
SSI_PRE_WAGE_HI  = 673.1    # £/wk — NE p75 2015
UNDEREMPLOYED_THRESHOLD = 0.90   # new_wage < 90% of pre_wage

# Method 2b: age-wage ratios (national ASHE Table 6, 2016-2019 mean)
# Ratio = age_band_median / all_employee_median, applied to SSI_PRE_WAGE
# Lower bound uses ratio - half IQR of across-year variation; upper = ratio + half IQR
AGE_WAGE_RATIOS = {
    "16-24":  {"mid": 0.840, "lo": 0.820, "hi": 0.856},
    "25-49":  {"mid": 1.109, "lo": 1.095, "hi": 1.122},
    "50+":    {"mid": 1.049, "lo": 1.034, "hi": 1.064},
}
AGE_BAND_COL = {"16-24": "16_24", "25-49": "25_49", "50+": "50plus"}

# Method 3: Dirichlet prior
# (retrain, share_similar, exit) — concentration kappa=20
DIRICHLET_PRIOR = np.array([5.0, 8.0, 7.0])   # prior means: 25% / 40% / 35%
DIRICHLET_CATS  = ["retrain", "share_similar", "exit"]
RC_EVIDENCE_WEIGHT = 0.3   # weight of derived RC LA pseudo-observations

# Quarter → months post-closure (month 1 = October 2015)
MONTHS_POST = {
    "2015Q4":  1, "2016Q1":  4, "2016Q2":  7, "2016Q3": 10,
    "2016Q4": 13, "2017Q1": 16, "2017Q2": 19, "2017Q3": 22, "2017Q4": 25,
    "2018Q1": 28, "2018Q2": 31, "2018Q3": 34, "2018Q4": 37,
    "2019Q1": 40, "2019Q2": 43, "2019Q3": 46, "2019Q4": 49,
}


# ── helpers ────────────────────────────────────────────────────────────────────

def fit_lognormal_sigma(p10, p25, p50, p75, p90) -> float:
    z90 = norm.ppf(0.90)
    z75 = norm.ppf(0.75)
    estimates = [
        (np.log(p90) - np.log(p50)) / z90,
        (np.log(p75) - np.log(p50)) / z75,
        (np.log(p50) - np.log(p25)) / z75,
        (np.log(p50) - np.log(p10)) / z90,
    ]
    return float(np.mean([e for e in estimates if np.isfinite(e)]))


def p_underemployed(pre_wage: float, mu: float, sigma: float) -> float:
    """P(new_wage < 0.9 * pre_wage) under LogNormal(mu, sigma). Returns %."""
    threshold = UNDEREMPLOYED_THRESHOLD * pre_wage
    z = (np.log(threshold) - mu) / sigma
    return float(norm.cdf(z) * 100)


# ── Method 1: Exponential survival model ─────────────────────────────────────

def solve_exponential(y5: float, y13: float):
    lam = np.log(y5 / y13) / (ANCHOR_T13_MONTH - ANCHOR_T5_MONTH)
    A   = y5 * np.exp(lam * ANCHOR_T5_MONTH)
    return A, lam


def build_still_seeking_estimates(quarters: list) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    A_mid, lam_mid = solve_exponential(ANCHOR_T5_PCT, ANCHOR_T13_MID)
    t13_samples    = rng.uniform(ANCHOR_T13_LO, ANCHOR_T13_HI, BOOTSTRAP_N)
    boot_A   = np.array([solve_exponential(ANCHOR_T5_PCT, y)[0] for y in t13_samples])
    boot_lam = np.array([solve_exponential(ANCHOR_T5_PCT, y)[1] for y in t13_samples])

    results = []
    for q in quarters:
        t         = MONTHS_POST[q] + 1.5
        mid       = float(np.clip(A_mid * np.exp(-lam_mid * t), 0, 100))
        boot_vals = np.clip(boot_A * np.exp(-boot_lam * t), 0, 100)
        results.append({
            "quarter":                  q,
            "stat_still_seeking_mid_%": round(mid, 2),
            "stat_still_seeking_lo_%":  round(float(np.percentile(boot_vals,  5)), 2),
            "stat_still_seeking_hi_%":  round(float(np.percentile(boot_vals, 95)), 2),
            "_lam_mid":                 lam_mid,
            "_A_mid":                   A_mid,
        })
    return pd.DataFrame(results)


# ── Method 2: Aggregate log-normal underemployment ───────────────────────────

def build_underemployed_estimates(quarters: list) -> pd.DataFrame:
    ashe = pd.read_csv(ONS / "ashe_percentiles_raw.csv")
    ashe.columns = ["year", "region", "statistic", "weekly_pay_gbp"]
    ne   = ashe[ashe["region"] == "North East"].copy()
    stat_map = {"10 percentile": "p10", "25 percentile": "p25",
                "Median": "p50", "75 percentile": "p75", "90 percentile": "p90"}
    ne["stat_key"] = ne["statistic"].map(stat_map)
    ne = ne.dropna(subset=["stat_key"])
    pivoted = ne.pivot_table(index="year", columns="stat_key",
                             values="weekly_pay_gbp", aggfunc="first")
    avail_years = sorted(pivoted.index.tolist())

    results = []
    for q in quarters:
        year_key = max(min(int(q[:4]), max(avail_years)), min(avail_years))
        row      = pivoted.loc[year_key]
        p10, p25, p50, p75, p90 = (row["p10"], row["p25"], row["p50"],
                                    row["p75"], row["p90"])
        sigma = fit_lognormal_sigma(p10, p25, p50, p75, p90)
        mu    = np.log(p50)
        results.append({
            "quarter":                   q,
            "stat_underemployed_mid_%":  round(p_underemployed(SSI_PRE_WAGE_MID, mu, sigma), 1),
            "stat_underemployed_lo_%":   round(p_underemployed(SSI_PRE_WAGE_LO,  mu, sigma), 1),
            "stat_underemployed_hi_%":   round(p_underemployed(SSI_PRE_WAGE_HI,  mu, sigma), 1),
            "_mu":                       mu,
            "_sigma":                    sigma,
        })
    return pd.DataFrame(results)


# ── Method 2b: Age-specific log-normal underemployment ───────────────────────

def build_age_underemployed_estimates(quarters: list, panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each quarter, compute age-specific P(underemployed) using:
      - ashe_age_{band}_median_est and ashe_age_{band}_sigma from panel
      - SSI pre-wage scaled by age ratio: pre_wage_age = SSI_PRE_WAGE * ratio_band

    Returns one row per (quarter, age_band) triplet.
    """
    # Pull one row per quarter with the ASHE age columns
    # (these are constant within a quarter regardless of row type)
    ashe_q = (panel[panel["age_band"] == "all"][["quarter"]
              + [c for c in panel.columns if c.startswith("ashe_age_")]]
              .drop_duplicates("quarter"))

    results = []
    for q in quarters:
        ashe_row = ashe_q[ashe_q["quarter"] == q]
        if ashe_row.empty:
            continue
        ashe_row = ashe_row.iloc[0]

        for band, col_sfx in AGE_BAND_COL.items():
            median = ashe_row.get(f"ashe_age_{col_sfx}_median_est", np.nan)
            sigma  = ashe_row.get(f"ashe_age_{col_sfx}_sigma",      np.nan)

            if pd.isna(median) or pd.isna(sigma) or median <= 0 or sigma <= 0:
                results.append({
                    "quarter": q, "age_band": band,
                    f"stat_underemployed_{col_sfx}_mid_%": np.nan,
                    f"stat_underemployed_{col_sfx}_lo_%":  np.nan,
                    f"stat_underemployed_{col_sfx}_hi_%":  np.nan,
                })
                continue

            mu    = np.log(median)
            ratio = AGE_WAGE_RATIOS[band]

            results.append({
                "quarter":  q,
                "age_band": band,
                f"stat_underemployed_{col_sfx}_mid_%":
                    round(p_underemployed(SSI_PRE_WAGE_MID * ratio["mid"], mu, sigma), 1),
                f"stat_underemployed_{col_sfx}_lo_%":
                    round(p_underemployed(SSI_PRE_WAGE_LO  * ratio["lo"],  mu, sigma), 1),
                f"stat_underemployed_{col_sfx}_hi_%":
                    round(p_underemployed(SSI_PRE_WAGE_HI  * ratio["hi"],  mu, sigma), 1),
            })
    return pd.DataFrame(results)


# ── Method 3: Dirichlet-Multinomial for retrain/share_similar/exit ───────────

def get_rc_la_evidence(panel: pd.DataFrame) -> dict:
    """
    Extract RC LA outcome proportions (retrain, share_similar, exit) from panel.
    Uses all/all rows. For quarters with multiple scenario rows, takes the mean.
    Returns {quarter: {retrain_pct, similar_pct, exit_pct}} (conditional on
    being resolved and not underemployed).
    """
    all_rows = panel[(panel["age_band"] == "all") & (panel["gender"] == "all")]
    evidence = {}

    for q, grp in all_rows.groupby("quarter"):
        r = grp["outcome_retrain_%"].dropna()
        s = grp["outcome_share_similar_%"].dropna()
        e = grp["outcome_exit_%"].dropna()
        if len(r) == 0 or len(s) == 0 or len(e) == 0:
            continue
        r_m, s_m, e_m = r.mean(), s.mean(), e.mean()
        total = r_m + s_m + e_m
        if total > 0:
            evidence[q] = {
                "retrain_pct":  r_m / total,
                "similar_pct":  s_m / total,
                "exit_pct":     e_m / total,
            }
    return evidence


def fit_dirichlet_posterior(rc_evidence: dict,
                             ss_df: pd.DataFrame,
                             ue_df: pd.DataFrame) -> np.ndarray:
    """
    Compute posterior Dirichlet alpha by combining cross-episode prior with
    soft RC LA observations.

    For each quarter with RC LA evidence, derive pseudo-observation counts:
      N_in_bucket(t) = SSI_N * (1 - still_seeking(t)/100)
                     * (1 - underemployed(t)/100)
      obs_k = N_in_bucket * proportion_k * RC_EVIDENCE_WEIGHT

    Posterior alpha = DIRICHLET_PRIOR + sum_over_quarters(obs_k)
    Uses all available quarters with evidence to pool observations.
    """
    ss_lookup = dict(zip(ss_df["quarter"], ss_df["stat_still_seeking_mid_%"]))
    ue_lookup = dict(zip(ue_df["quarter"], ue_df["stat_underemployed_mid_%"]))

    obs_total = np.zeros(3)   # (retrain, similar, exit)
    quarters_used = []

    for q, ev in rc_evidence.items():
        ss_pct = ss_lookup.get(q, np.nan)
        ue_pct = ue_lookup.get(q, np.nan)
        if np.isnan(ss_pct) or np.isnan(ue_pct):
            continue

        n_resolved     = SSI_N * (1 - ss_pct / 100)
        n_in_bucket    = n_resolved * (1 - ue_pct / 100)
        n_in_bucket    = max(n_in_bucket, 0)

        obs_total[0] += n_in_bucket * ev["retrain_pct"] * RC_EVIDENCE_WEIGHT
        obs_total[1] += n_in_bucket * ev["similar_pct"] * RC_EVIDENCE_WEIGHT
        obs_total[2] += n_in_bucket * ev["exit_pct"]    * RC_EVIDENCE_WEIGHT
        quarters_used.append(q)

    alpha_post = DIRICHLET_PRIOR + obs_total
    print(f"  Dirichlet posterior alpha (retrain, similar, exit): "
          f"{alpha_post.round(1)} [from {len(quarters_used)} quarters]")
    print(f"  Prior alpha: {DIRICHLET_PRIOR}")
    print(f"  RC LA obs added: {obs_total.round(1)}")
    pi = alpha_post / alpha_post.sum()
    print(f"  Posterior mean split: retrain={pi[0]:.1%}, "
          f"similar={pi[1]:.1%}, exit={pi[2]:.1%}")
    return alpha_post


def build_dirichlet_estimates(quarters: list,
                               ss_df: pd.DataFrame,
                               ue_df: pd.DataFrame,
                               alpha_post: np.ndarray) -> pd.DataFrame:
    """
    For each quarter, compute absolute % estimates for retrain/share_similar/exit
    by sampling from the posterior Dirichlet and scaling by the resolved
    non-underemployed fraction.

    remaining(t) = (1 - still_seeking(t)) * (1 - underemployed(t))
    outcome_k(t) = pi_k * remaining(t)    [sampled distribution]
    """
    rng = np.random.default_rng(RNG_SEED + 1)

    # Sample from posterior Dirichlet: (BOOTSTRAP_N, 3) matrix of pi triplets
    pi_samples = rng.dirichlet(alpha_post, size=BOOTSTRAP_N)   # each row sums to 1

    ss_lookup = dict(zip(ss_df["quarter"], ss_df["stat_still_seeking_mid_%"]))
    ue_lookup = dict(zip(ue_df["quarter"], ue_df["stat_underemployed_mid_%"]))

    pi_post = alpha_post / alpha_post.sum()

    results = []
    for q in quarters:
        ss_pct = ss_lookup.get(q, 0.0)
        ue_pct = ue_lookup.get(q, 0.0)

        remaining = (1 - ss_pct / 100) * (1 - ue_pct / 100)  # fraction of SSI workers

        # Sampled absolute outcomes: (BOOTSTRAP_N, 3) * scalar
        sampled = pi_samples * remaining * 100   # convert to %

        for i, cat in enumerate(DIRICHLET_CATS):
            mid = round(float(pi_post[i] * remaining * 100), 1)
            lo  = round(float(np.percentile(sampled[:, i],  5)), 1)
            hi  = round(float(np.percentile(sampled[:, i], 95)), 1)
            results.append({
                "quarter": q,
                "category": cat,
                "mid": mid, "lo": lo, "hi": hi,
            })

    df = pd.DataFrame(results)
    pivot = df.pivot(index="quarter", columns="category", values=["mid", "lo", "hi"])
    pivot.columns = [f"stat_{cat}_{stat}_%" for stat, cat in pivot.columns]
    pivot = pivot.reset_index()
    pivot["stat_dirichlet_kappa"]        = round(float(alpha_post.sum()), 1)
    pivot["stat_dirichlet_pi_retrain"]   = round(float(pi_post[0]), 4)
    pivot["stat_dirichlet_pi_similar"]   = round(float(pi_post[1]), 4)
    pivot["stat_dirichlet_pi_exit"]      = round(float(pi_post[2]), 4)
    return pivot


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("estimate_statistical.py")
    print("=" * 60)
    print(f"Reading: {PANEL_PATH.name}")
    panel = pd.read_csv(PANEL_PATH)
    print(f"  {len(panel)} rows x {len(panel.columns)} columns")

    quarters = sorted(panel["quarter"].unique().tolist(),
                      key=lambda q: MONTHS_POST[q])

    # Drop existing stat_* columns to rebuild cleanly
    stat_cols = [c for c in panel.columns if c.startswith("stat_")]
    if stat_cols:
        panel = panel.drop(columns=stat_cols)
        print(f"  Dropped {len(stat_cols)} existing stat_* columns")

    # ── Method 1 ──────────────────────────────────────────────────────────────
    print(f"\n[Method 1] Exponential survival model ({BOOTSTRAP_N:,} bootstrap iters)...")
    ss_df = build_still_seeking_estimates(quarters)
    lam   = ss_df["_lam_mid"].iloc[0]
    A     = ss_df["_A_mid"].iloc[0]
    print(f"  Lambda = {lam:.4f}/month  |  half-life = {np.log(2)/lam:.2f} months")
    print(f"  A (theoretical peak) = {A:.2f}%")
    print(f"  2016Q1 mid = {ss_df[ss_df.quarter=='2016Q1']['stat_still_seeking_mid_%'].values[0]:.1f}%  "
          f"[anchor: 25.0%]")
    print(f"  2016Q4 mid = {ss_df[ss_df.quarter=='2016Q4']['stat_still_seeking_mid_%'].values[0]:.1f}%  "
          f"[anchor: ~3.0%]")

    # ── Method 2 ──────────────────────────────────────────────────────────────
    print(f"\n[Method 2] Aggregate log-normal underemployment...")
    ue_df = build_underemployed_estimates(quarters)
    print(f"  2016Q1 sigma = {ue_df[ue_df.quarter=='2016Q1']['_sigma'].values[0]:.4f}")
    print(f"  2016Q1 P(underemployed) mid = {ue_df[ue_df.quarter=='2016Q1']['stat_underemployed_mid_%'].values[0]:.1f}%  "
          f"[{ue_df[ue_df.quarter=='2016Q1']['stat_underemployed_lo_%'].values[0]:.1f}%–"
          f"{ue_df[ue_df.quarter=='2016Q1']['stat_underemployed_hi_%'].values[0]:.1f}%]")

    # ── Method 2b ─────────────────────────────────────────────────────────────
    print(f"\n[Method 2b] Age-specific log-normal underemployment...")
    ue_age_df = build_age_underemployed_estimates(quarters, panel)
    for band, sfx in AGE_BAND_COL.items():
        row = ue_age_df[(ue_age_df["quarter"] == "2016Q1") &
                        (ue_age_df["age_band"] == band)]
        if not row.empty:
            mid = row[f"stat_underemployed_{sfx}_mid_%"].values[0]
            lo  = row[f"stat_underemployed_{sfx}_lo_%"].values[0]
            hi  = row[f"stat_underemployed_{sfx}_hi_%"].values[0]
            print(f"  2016Q1 {band}: P(underemployed) = {mid:.1f}%  [{lo:.1f}%–{hi:.1f}%]")

    # ── Method 3 ──────────────────────────────────────────────────────────────
    print(f"\n[Method 3] Bayesian Dirichlet-Multinomial...")
    print(f"  Prior (cross-episode): retrain={DIRICHLET_PRIOR[0]/DIRICHLET_PRIOR.sum():.0%}, "
          f"similar={DIRICHLET_PRIOR[1]/DIRICHLET_PRIOR.sum():.0%}, "
          f"exit={DIRICHLET_PRIOR[2]/DIRICHLET_PRIOR.sum():.0%}  "
          f"(kappa={DIRICHLET_PRIOR.sum():.0f})")

    rc_evidence  = get_rc_la_evidence(panel)
    print(f"  RC LA evidence available for {len(rc_evidence)} quarters")
    alpha_post   = fit_dirichlet_posterior(rc_evidence, ss_df, ue_df)
    dir_df       = build_dirichlet_estimates(quarters, ss_df, ue_df, alpha_post)

    print(f"\n  Per-quarter absolute estimates (mid%):")
    print(f"  {'quarter':<10} | {'still_seek':>10} | {'underempl':>9} | "
          f"{'retrain':>8} | {'similar':>8} | {'exit':>8} | sum")
    for q in ["2015Q4","2016Q1","2016Q4","2017Q4","2019Q4"]:
        ss   = ss_df[ss_df.quarter==q]["stat_still_seeking_mid_%"].values[0]
        ue   = ue_df[ue_df.quarter==q]["stat_underemployed_mid_%"].values[0]
        dr   = dir_df[dir_df.quarter==q]
        ret  = dr["stat_retrain_mid_%"].values[0]
        sim  = dr["stat_share_similar_mid_%"].values[0]
        ex   = dr["stat_exit_mid_%"].values[0]
        total = ss + ue*(1-ss/100) + ret + sim + ex
        print(f"  {q:<10} | {ss:>10.1f} | {ue*(1-ss/100):>9.1f} | "
              f"{ret:>8.1f} | {sim:>8.1f} | {ex:>8.1f} | ~{total:.0f}%")

    # ── Merge all into panel ───────────────────────────────────────────────────
    print(f"\nMerging into panel...")
    out = panel.copy()

    # Method 1: broadcast to all rows by quarter
    ss_clean = ss_df.drop(columns=["_lam_mid", "_A_mid"])
    out = out.merge(ss_clean, on="quarter", how="left")

    # Method 2: apply to all rows (aggregate)
    ue_clean = ue_df[["quarter", "stat_underemployed_mid_%",
                       "stat_underemployed_lo_%", "stat_underemployed_hi_%"]]
    out = out.merge(ue_clean, on="quarter", how="left")

    # Blank out Method 2 for age-specific rows (use 2b there instead)
    age_mask = out["age_band"] != "all"
    for col in ["stat_underemployed_mid_%", "stat_underemployed_lo_%",
                "stat_underemployed_hi_%"]:
        out.loc[age_mask, col] = np.nan

    # Method 2b: merge per (quarter, age_band) for age-specific rows
    for band, sfx in AGE_BAND_COL.items():
        cols = ["quarter", "age_band",
                f"stat_underemployed_{sfx}_mid_%",
                f"stat_underemployed_{sfx}_lo_%",
                f"stat_underemployed_{sfx}_hi_%"]
        sub = ue_age_df[ue_age_df["age_band"] == band][cols]

        # Rename to generic columns, then apply only to matching age_band rows
        sub = sub.rename(columns={
            f"stat_underemployed_{sfx}_mid_%": f"_ue_{sfx}_mid",
            f"stat_underemployed_{sfx}_lo_%":  f"_ue_{sfx}_lo",
            f"stat_underemployed_{sfx}_hi_%":  f"_ue_{sfx}_hi",
        })
        out = out.merge(sub.drop(columns="age_band"), on="quarter", how="left")

        band_mask = out["age_band"] == band
        out.loc[band_mask, "stat_underemployed_mid_%"] = out.loc[band_mask, f"_ue_{sfx}_mid"]
        out.loc[band_mask, "stat_underemployed_lo_%"]  = out.loc[band_mask, f"_ue_{sfx}_lo"]
        out.loc[band_mask, "stat_underemployed_hi_%"]  = out.loc[band_mask, f"_ue_{sfx}_hi"]
        out = out.drop(columns=[f"_ue_{sfx}_mid", f"_ue_{sfx}_lo", f"_ue_{sfx}_hi"])

    # Add age-specific raw columns too (alongside the generic ones)
    ue_age_wide = ue_age_df.drop(columns="age_band")
    ue_age_wide = ue_age_wide.groupby("quarter").first().reset_index()
    out = out.merge(ue_age_wide.drop(columns=[c for c in ue_age_wide.columns
                                               if c in out.columns and c != "quarter"],
                                     errors="ignore"),
                    on="quarter", how="left")

    # Method 3: broadcast to all rows by quarter
    out = out.merge(dir_df, on="quarter", how="left")

    # Summary stat_method column
    out["stat_method"] = (
        "M1:exp_decay(DWP_t5+TaskForce_t13,bootstrap=10000); "
        "M2:lognormal_agg(NE_ASHE_p10-p90); "
        "M2b:lognormal_age(ASHE_Table6_scaled_to_NE); "
        "M3:dirichlet_multinomial(cross_episode_prior+RC_LA_soft_obs)"
    )

    out.to_csv(PANEL_PATH, index=False)
    new_stat_cols = [c for c in out.columns if c.startswith("stat_")]
    print(f"\nSaved: {PANEL_PATH}")
    print(f"  {len(out)} rows x {len(out.columns)} columns")
    print(f"  stat_* columns: {len(new_stat_cols)}")
    print(f"  {new_stat_cols}")


if __name__ == "__main__":
    main()
