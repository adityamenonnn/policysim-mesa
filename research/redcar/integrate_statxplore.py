"""
integrate_statxplore.py
=======================
Parses the DWP Stat-Xplore Alternative Claimant Count (ACC) download
(ACC 4 - Age by Gender, Redcar and Cleveland LA) and integrates real
age x gender claimant counts into the math_testing_panel.

What this replaces
------------------
ssi_age_share_prior  — was estimated (ESTIMATED_LOW) from industry norms
ssi_gender_share_prior — was estimated (ESTIMATED_LOW) from steel sector norms

What this adds
--------------
Real quarterly claimant counts and shares by age band and gender for
Redcar & Cleveland LA from the ACC dataset. Uses the excess-over-baseline
approach to isolate the SSI shock composition:
  - Baseline = mean of 2019 months (post-shock stable period)
  - Excess at each month = actual - baseline (clamped at 0)
  - The distribution of excess claimants approximates SSI worker demographics

Model age band mapping (Stat-Xplore → panel):
  16-19 + 20-24          → 16-24
  25-29 + 30-34 + 35-39 + 40-44 + 45-49 → 25-49
  50-54 + 55-59 + 60 plus → 50+

New columns added to math_testing_panel.csv
-------------------------------------------
  acc_16_24_male_q_mean     real RC LA claimants aged 16-24, male, quarterly mean
  acc_16_24_female_q_mean   real RC LA claimants aged 16-24, female, quarterly mean
  acc_25_49_male_q_mean     etc.
  acc_25_49_female_q_mean
  acc_50plus_male_q_mean
  acc_50plus_female_q_mean
  acc_total_q_mean          total RC LA ACC claimants (all ages, both genders)
  acc_16_24_pct             16-24 share of total RC LA claimants (%)
  acc_25_49_pct             25-49 share
  acc_50plus_pct            50+ share
  acc_male_pct              male share of total RC LA claimants (%)
  acc_female_pct            female share
  acc_excess_16_24_pct      16-24 share of EXCESS claimants (SSI shock proxy)
  acc_excess_25_49_pct      25-49 share of excess
  acc_excess_50plus_pct     50+ share of excess
  acc_excess_male_pct       male share of excess claimants
  acc_data_quality          always HIGH (primary DWP administrative data)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
XLSX_PATH = Path("/Users/adityamenon/Downloads/table_2026-08-27_13-00-53.xlsx")
PANEL_IN  = BASE / "math_testing_panel.csv"
PANEL_OUT = BASE / "math_testing_panel.csv"   # update in place

# ── constants ──────────────────────────────────────────────────────────────────
SHEET = "0. Redcar and Cleveland"

# Stat-Xplore age band → model age band
AGE_MAP = {
    "16-19":   "16-24",
    "20-24":   "16-24",
    "25-29":   "25-49",
    "30-34":   "25-49",
    "35-39":   "25-49",
    "40-44":   "25-49",
    "45-49":   "25-49",
    "50-54":   "50+",
    "55-59":   "50+",
    "60 plus": "50+",
}

QUARTERS = {
    "2015Q4": ["October 2015", "November 2015", "December 2015"],
    "2016Q1": ["January 2016", "February 2016", "March 2016"],
    "2016Q2": ["April 2016", "May 2016", "June 2016"],
    "2016Q3": ["July 2016", "August 2016", "September 2016"],
    "2016Q4": ["October 2016", "November 2016", "December 2016"],
    "2017Q1": ["January 2017", "February 2017", "March 2017"],
    "2017Q2": ["April 2017", "May 2017", "June 2017"],
    "2017Q3": ["July 2017", "August 2017", "September 2017"],
    "2017Q4": ["October 2017", "November 2017", "December 2017"],
    "2018Q1": ["January 2018", "February 2018", "March 2018"],
    "2018Q2": ["April 2018", "May 2018", "June 2018"],
    "2018Q3": ["July 2018", "August 2018", "September 2018"],
    "2018Q4": ["October 2018", "November 2018", "December 2018"],
    "2019Q1": ["January 2019", "February 2019", "March 2019"],
    "2019Q2": ["April 2019", "May 2019", "June 2019"],
    "2019Q3": ["July 2019", "August 2019", "September 2019"],
    "2019Q4": ["October 2019", "November 2019", "December 2019"],
}

# 2019 months used as post-shock baseline (SSI shock fully absorbed by then)
BASELINE_MONTHS = [
    "January 2019", "February 2019", "March 2019",
    "April 2019", "May 2019", "June 2019",
    "July 2019", "August 2019", "September 2019",
    "October 2019", "November 2019", "December 2019",
]


# ── Step 1: Parse the Excel file into tidy format ─────────────────────────────

def parse_statxplore(xlsx_path: Path) -> pd.DataFrame:
    """
    Parse the Stat-Xplore ACC xlsx into a tidy DataFrame with columns:
      month, age_band_raw, gender, count
    """
    raw = pd.read_excel(xlsx_path, sheet_name=SHEET, header=None)

    # Find the row indices for Month, Gender, and first data row
    # Row 10 = months, Row 11 = genders, Row 13+ = age band data
    month_row  = raw.iloc[10]
    gender_row = raw.iloc[11]

    # Build column index: list of (month_label, gender) for data columns
    # Columns 2 onwards are data; columns alternate Male/Female per month
    col_meta = []
    current_month = None
    for col_idx in range(2, len(raw.columns)):
        m = month_row.iloc[col_idx]
        g = gender_row.iloc[col_idx]
        if pd.notna(m):
            current_month = str(m).strip()
        if pd.notna(g) and current_month:
            col_meta.append((col_idx, current_month, str(g).strip()))

    # Parse age band rows (rows 13 to 22 = 10 age bands)
    records = []
    for row_idx in range(13, 23):
        row = raw.iloc[row_idx]
        age_raw = str(row.iloc[1]).strip()
        if age_raw not in AGE_MAP and age_raw != "Unknown or missing age band":
            continue
        if age_raw == "Unknown or missing age band":
            continue
        for col_idx, month, gender in col_meta:
            val = row.iloc[col_idx]
            if pd.isna(val) or str(val).strip() in ["..", "-", ""]:
                count = np.nan
            else:
                try:
                    count = float(str(val).replace(",", ""))
                except ValueError:
                    count = np.nan
            records.append({
                "month":       month,
                "age_band_raw": age_raw,
                "gender":      gender,
                "count":       count,
            })

    df = pd.DataFrame(records)
    df["model_age_band"] = df["age_band_raw"].map(AGE_MAP)
    return df


# ── Step 2: Aggregate to model age bands and compute quarterly means ───────────

def build_quarterly_lookup(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    For each quarter, compute:
      - Mean monthly claimant count by model_age_band x gender
      - Total (all ages, both genders)
      - Age band shares of total
      - Gender shares of total
      - Excess over 2019 baseline by age band x gender
      - Age band and gender shares of excess (SSI shock proxy)
    """
    # First aggregate Stat-Xplore age bands → model age bands
    agg = (tidy.groupby(["month", "model_age_band", "gender"])["count"]
               .sum(min_count=1)
               .reset_index())

    # 2019 baseline: mean by model_age_band x gender over 2019 months
    base = (agg[agg["month"].isin(BASELINE_MONTHS)]
              .groupby(["model_age_band", "gender"])["count"]
              .mean()
              .rename("baseline"))

    agg = agg.merge(base.reset_index(), on=["model_age_band", "gender"], how="left")
    agg["excess"] = (agg["count"] - agg["baseline"]).clip(lower=0)

    results = []
    for q, months in QUARTERS.items():
        q_data = agg[agg["month"].isin(months)]

        def qmean(age, gender):
            sub = q_data[(q_data["model_age_band"] == age) &
                         (q_data["gender"] == gender)]["count"]
            return round(sub.mean(), 1) if len(sub) > 0 else np.nan

        def qexcess(age, gender):
            sub = q_data[(q_data["model_age_band"] == age) &
                         (q_data["gender"] == gender)]["excess"]
            return round(sub.mean(), 1) if len(sub) > 0 else np.nan

        # Quarterly means by cell
        r = {
            "quarter":               q,
            "acc_16_24_male_q_mean":   qmean("16-24", "Male"),
            "acc_16_24_female_q_mean": qmean("16-24", "Female"),
            "acc_25_49_male_q_mean":   qmean("25-49", "Male"),
            "acc_25_49_female_q_mean": qmean("25-49", "Female"),
            "acc_50plus_male_q_mean":  qmean("50+",   "Male"),
            "acc_50plus_female_q_mean":qmean("50+",   "Female"),
        }

        # Totals
        total_16_24 = (r["acc_16_24_male_q_mean"] or 0) + (r["acc_16_24_female_q_mean"] or 0)
        total_25_49 = (r["acc_25_49_male_q_mean"] or 0) + (r["acc_25_49_female_q_mean"] or 0)
        total_50plus = (r["acc_50plus_male_q_mean"] or 0) + (r["acc_50plus_female_q_mean"] or 0)
        total_male   = (r["acc_16_24_male_q_mean"] or 0) + (r["acc_25_49_male_q_mean"] or 0) + (r["acc_50plus_male_q_mean"] or 0)
        total_female = (r["acc_16_24_female_q_mean"] or 0) + (r["acc_25_49_female_q_mean"] or 0) + (r["acc_50plus_female_q_mean"] or 0)
        total_all    = total_16_24 + total_25_49 + total_50plus

        r["acc_total_q_mean"] = round(total_all, 1)

        # Age band shares of total RC LA claimants
        r["acc_16_24_pct"]   = round(100 * total_16_24  / total_all, 1) if total_all > 0 else np.nan
        r["acc_25_49_pct"]   = round(100 * total_25_49  / total_all, 1) if total_all > 0 else np.nan
        r["acc_50plus_pct"]  = round(100 * total_50plus / total_all, 1) if total_all > 0 else np.nan
        r["acc_male_pct"]    = round(100 * total_male   / total_all, 1) if total_all > 0 else np.nan
        r["acc_female_pct"]  = round(100 * total_female / total_all, 1) if total_all > 0 else np.nan

        # Excess claimants (SSI shock proxy)
        ex_16_24  = (qexcess("16-24", "Male") or 0) + (qexcess("16-24", "Female") or 0)
        ex_25_49  = (qexcess("25-49", "Male") or 0) + (qexcess("25-49", "Female") or 0)
        ex_50plus = (qexcess("50+",   "Male") or 0) + (qexcess("50+",   "Female") or 0)
        ex_male   = (qexcess("16-24", "Male") or 0) + (qexcess("25-49", "Male") or 0) + (qexcess("50+", "Male") or 0)
        ex_female = (qexcess("16-24", "Female") or 0) + (qexcess("25-49", "Female") or 0) + (qexcess("50+", "Female") or 0)
        ex_total  = ex_16_24 + ex_25_49 + ex_50plus

        r["acc_excess_16_24_pct"]  = round(100 * ex_16_24  / ex_total, 1) if ex_total > 0 else np.nan
        r["acc_excess_25_49_pct"]  = round(100 * ex_25_49  / ex_total, 1) if ex_total > 0 else np.nan
        r["acc_excess_50plus_pct"] = round(100 * ex_50plus / ex_total, 1) if ex_total > 0 else np.nan
        r["acc_excess_male_pct"]   = round(100 * ex_male   / ex_total, 1) if ex_total > 0 else np.nan
        r["acc_data_quality"]      = "HIGH"

        results.append(r)

    return pd.DataFrame(results)


# ── Step 3: Merge into panel ───────────────────────────────────────────────────

def main():
    print("Parsing Stat-Xplore xlsx...")
    tidy = parse_statxplore(XLSX_PATH)
    print(f"  Parsed {len(tidy)} records across {tidy['month'].nunique()} months")
    print(f"  Age bands: {sorted(tidy['age_band_raw'].unique())}")
    print(f"  Genders:   {tidy['gender'].unique().tolist()}")

    print("\nBuilding quarterly lookup...")
    lookup = build_quarterly_lookup(tidy)

    print("\n  quarter    | 16-24% | 25-49% | 50+%  | male% | excess_male%")
    print("  -----------|--------|--------|-------|-------|-------------")
    for _, r in lookup.iterrows():
        print(f"  {r.quarter}  |  {r['acc_16_24_pct']:5.1f} |  {r['acc_25_49_pct']:5.1f} | {r['acc_50plus_pct']:5.1f} | {r['acc_male_pct']:5.1f} | {r.get('acc_excess_male_pct', np.nan)}")

    print(f"\nLoading panel ({PANEL_IN.name})...")
    panel = pd.read_csv(PANEL_IN)
    print(f"  {len(panel)} rows x {len(panel.columns)} columns")

    # Drop any existing acc_ columns to avoid duplication on re-run
    acc_cols = [c for c in panel.columns if c.startswith("acc_")]
    if acc_cols:
        panel = panel.drop(columns=acc_cols)
        print(f"  Dropped {len(acc_cols)} existing acc_ columns")

    panel = panel.merge(lookup, on="quarter", how="left")
    print(f"  After merge: {len(panel)} rows x {len(panel.columns)} columns")
    print(f"  New columns: {[c for c in panel.columns if c.startswith('acc_')]}")

    panel.to_csv(PANEL_OUT, index=False)
    print(f"\nSaved: {PANEL_OUT}")

    # Summary: compare ACC-derived shares vs previous estimates
    print("\n--- Comparison: real ACC shares vs previous ESTIMATED_LOW priors ---")
    print("  (These are RC LA claimant shares, not SSI workforce shares directly)")
    print(f"  {'quarter':<10} | 16-24 real | 25-49 real | 50+ real | male real | prev age est      | prev gender est")
    print(f"  -----------|------------|------------|----------|-----------|-------------------|----------------")
    for _, r in lookup.iterrows():
        print(f"  {r.quarter:<10} |     {r['acc_16_24_pct']:5.1f}% |     {r['acc_25_49_pct']:5.1f}% |   {r['acc_50plus_pct']:5.1f}% |    {r['acc_male_pct']:5.1f}% | 10% / 40% / 50%   | 95% / 5%")


if __name__ == "__main__":
    main()
