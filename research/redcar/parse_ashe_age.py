"""
parse_ashe_age.py
=================
Downloads ONS ASHE Table 6 (age group, full-time, weekly gross pay) ZIP files
for 2016-2019, extracts national age-band wage distributions, then scales them
to North East regional level using NE aggregate medians already in the panel.

No manual download needed — runs fully automatically.

Method
------
ASHE Table 6.1a (Full-Time sheet) gives NATIONAL percentiles by age group:
  16-17, 18-21, 22-29, 30-39, 40-49, 50-59, 60+

Regional breakdown by age is not published by ONS. We approximate NE age-band
wages using two steps:

  1. National age ratios (scale factors):
       ratio_band = national_median_band / national_median_all
     These ratios capture how much younger/older workers earn relative to the
     average and are relatively stable across regions.

  2. Apply to NE aggregate median (already in panel from NM_30_1):
       ne_estimated_median_band = ne_aggregate_median * ratio_band

  3. Log-normal sigma comes from the SHAPE of the national distribution
     (p10/p25/p50/p75/p90 ratios). Wage inequality within age groups is
     similar across regions, so national sigma transfers to NE estimates.

Model age band mapping
----------------------
  16-24  -> national 22-29 band (proxy; 16-21 are tiny in full-time workforce)
  25-49  -> jobs-weighted mean of national 30-39 and 40-49
  50+    -> jobs-weighted mean of national 50-59 and 60+

Columns added to math_testing_panel.csv
-----------------------------------------
  ashe_age_16_24_median_est   NE estimated median weekly pay, 16-24 (£)
  ashe_age_16_24_p10_est      scaled from national 22-29 distribution
  ashe_age_16_24_p25_est
  ashe_age_16_24_p75_est
  ashe_age_16_24_p90_est
  ashe_age_16_24_sigma        log-normal sigma (from national shape)
  (same set for 25_49 and 50plus)
  ashe_age_nat_ratio_16_24    national 22-29/all median ratio (scale factor)
  ashe_age_nat_ratio_25_49
  ashe_age_nat_ratio_50plus
  ashe_age_data               HIGH_SCALED (NE scaled from national ratios)

Quarter-to-year matching
-------------------------
  2015Q4, 2016Q1-Q4  ->  2016 ASHE (April 2016 reference)
  2017Q1-Q4          ->  2017 ASHE
  2018Q1-Q4          ->  2018 ASHE
  2019Q1-Q4          ->  2019 ASHE

Usage
-----
  cd /Users/adityamenon/Documents/PolicySim/policysim-mesa
  python3 research/redcar/parse_ashe_age.py
"""

from pathlib import Path
import io
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

BASE       = Path(__file__).parent
CACHE_DIR  = BASE / "_ashe_cache"
PANEL_PATH = BASE / "math_testing_panel.csv"

CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (policysim-mesa data pull)"}

# ONS Table 6 revised ZIP URLs for each year
ZIP_URLS = {
    2016: "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/agegroupashetable6/2016revised/table62016revised.zip",
    2017: "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/agegroupashetable6/2017revised/table62017revised.zip",
    2018: "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/agegroupashetable6/2018revised/table62018revised.zip",
    2019: "https://www.ons.gov.uk/file?uri=/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/agegroupashetable6/2019revised/table62019revised.zip",
}

YEAR_TO_QUARTERS = {
    2016: ["2015Q4", "2016Q1", "2016Q2", "2016Q3", "2016Q4"],
    2017: ["2017Q1", "2017Q2", "2017Q3", "2017Q4"],
    2018: ["2018Q1", "2018Q2", "2018Q3", "2018Q4"],
    2019: ["2019Q1", "2019Q2", "2019Q3", "2019Q4"],
}

# Row labels to look for in the Full-Time sheet (col 0)
# These appear as-is in the ASHE spreadsheet
ASHE_BANDS = {
    "all":   ["all employees", "all"],
    "16_17": ["16-17"],
    "18_21": ["18-21"],
    "22_29": ["22-29"],
    "30_39": ["30-39"],
    "40_49": ["40-49"],
    "50_59": ["50-59"],
    "60plus": ["60+", "60 and over"],
}

# Column positions in the Full-Time sheet (row 4 = header):
# 0=Description, 1=Code, 2=Jobs(000s), 3=Median, 4=Ann%chg, 5=Mean,
# 6=Ann%chg, 7=p10, 8=p20, 9=p25, 10=p30, 11=p40, 12=p60, 13=p70,
# 14=p75, 15=p80, 16=p90
COL_JOBS   = 2
COL_MEDIAN = 3
COL_P10    = 7
COL_P25    = 9
COL_P75    = 14
COL_P90    = 16


def lognormal_sigma(p10, p25, p50, p75, p90) -> float:
    z90 = norm.ppf(0.90)
    z75 = norm.ppf(0.75)
    ests = []
    for num, den in [(p90, p50), (p75, p50)]:
        if num and den and num > 0 and den > 0:
            ests.append((np.log(num) - np.log(den)) / (z90 if num == p90 else z75))
    for num, den in [(p50, p25), (p50, p10)]:
        if num and den and num > 0 and den > 0:
            ests.append((np.log(num) - np.log(den)) / (z75 if den == p25 else z90))
    return round(float(np.mean(ests)), 4) if ests else np.nan


def safe_float(val) -> float | None:
    try:
        f = float(str(val).replace(",", "").strip())
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


def download_zip(year: int) -> bytes:
    cache = CACHE_DIR / f"ashe_table6_{year}.zip"
    if cache.exists():
        print(f"    [cache] ashe_table6_{year}.zip")
        return cache.read_bytes()
    print(f"    [fetch] Table 6 {year}...")
    r = requests.get(ZIP_URLS[year], headers=HEADERS, timeout=180)
    r.raise_for_status()
    cache.write_bytes(r.content)
    time.sleep(1.5)
    return r.content


def parse_year(year: int) -> dict:
    """
    Download and parse ASHE Table 6.1a for one year.
    Returns {band_key: {jobs, median, p10, p25, p75, p90}} for national data.
    """
    raw_zip = download_zip(year)
    z = zipfile.ZipFile(io.BytesIO(raw_zip))

    # Find the gross weekly pay file
    target = next((n for n in z.namelist()
                   if "6.1a" in n and "gross" in n.lower()), None)
    if target is None:
        # Fallback: first file with "6.1" in name
        target = next((n for n in z.namelist() if "6.1" in n and "cv" not in n.lower()), None)
    if target is None:
        print(f"    WARNING: cannot find Table 6.1a in {year} ZIP")
        return {}

    print(f"    Parsing: {target}")
    df = pd.read_excel(z.open(target), sheet_name="Full-Time", header=None)

    result = {}
    for band_key, search_terms in ASHE_BANDS.items():
        for row_i in range(len(df)):
            label = str(df.iloc[row_i, 0]).strip().lower()
            if any(t.lower() == label for t in search_terms):
                row = df.iloc[row_i]
                result[band_key] = {
                    "jobs":   safe_float(row.iloc[COL_JOBS]),
                    "median": safe_float(row.iloc[COL_MEDIAN]),
                    "p10":    safe_float(row.iloc[COL_P10]),
                    "p25":    safe_float(row.iloc[COL_P25]),
                    "p75":    safe_float(row.iloc[COL_P75]),
                    "p90":    safe_float(row.iloc[COL_P90]),
                }
                break

    # Print what we got
    for k, v in result.items():
        print(f"      {k}: median={v.get('median')}, p10={v.get('p10')}, p90={v.get('p90')}")

    return result


def aggregate_bands(nat: dict) -> dict:
    """
    Collapse ASHE bands into model age bands (16-24, 25-49, 50+).
    Uses jobs-weighted means for merged bands; uses 22-29 as proxy for 16-24.
    Returns {model_band: {median, p10, p25, p75, p90, sigma, ratio_to_all}}.
    """
    all_median = nat.get("all", {}).get("median")
    model = {}

    # 16-24: use 22-29 as proxy (best available; 16-21 rarely full-time)
    band_22_29 = nat.get("22_29", {})
    model["16_24"] = {
        "median": band_22_29.get("median"),
        "p10":    band_22_29.get("p10"),
        "p25":    band_22_29.get("p25"),
        "p75":    band_22_29.get("p75"),
        "p90":    band_22_29.get("p90"),
    }

    # 25-49: jobs-weighted mean of 30-39 and 40-49
    b30 = nat.get("30_39", {})
    b40 = nat.get("40_49", {})
    j30 = b30.get("jobs") or 1
    j40 = b40.get("jobs") or 1
    total = j30 + j40
    model["25_49"] = {}
    for stat in ["median", "p10", "p25", "p75", "p90"]:
        v30 = b30.get(stat)
        v40 = b40.get(stat)
        if v30 and v40:
            model["25_49"][stat] = (v30 * j30 + v40 * j40) / total
        elif v30:
            model["25_49"][stat] = v30
        else:
            model["25_49"][stat] = v40

    # 50+: jobs-weighted mean of 50-59 and 60+
    b50 = nat.get("50_59", {})
    b60 = nat.get("60plus", {})
    j50 = b50.get("jobs") or 1
    j60 = b60.get("jobs") or 1
    total60 = j50 + j60
    model["50plus"] = {}
    for stat in ["median", "p10", "p25", "p75", "p90"]:
        v50 = b50.get(stat)
        v60 = b60.get(stat)
        if v50 and v60:
            model["50plus"][stat] = (v50 * j50 + v60 * j60) / total60
        elif v50:
            model["50plus"][stat] = v50
        else:
            model["50plus"][stat] = v60

    # Compute sigma and ratio_to_all for each model band
    for band in ["16_24", "25_49", "50plus"]:
        b = model[band]
        b["sigma"] = lognormal_sigma(
            b.get("p10"), b.get("p25"), b.get("median"),
            b.get("p75"), b.get("p90")
        )
        b["ratio_to_all"] = round(b["median"] / all_median, 4) if (
            b.get("median") and all_median) else np.nan

    return model


def scale_to_ne(model_nat: dict, ne_aggregate_median: float) -> dict:
    """
    Scale national age-band medians to NE regional level.
    All percentiles are shifted by the same ratio (preserves shape/sigma).
    ne_aggregate_median: NE full-time gross weekly pay median (from NM_30_1).
    """
    ne_model = {}
    for band, stats in model_nat.items():
        ratio = stats.get("ratio_to_all", np.nan)
        ne_median_est = ne_aggregate_median * ratio if (
            not np.isnan(ratio) and ne_aggregate_median) else np.nan

        # Scale all percentiles by the same ratio (preserves distribution shape)
        ne_model[band] = {
            "median_est": round(ne_median_est, 1) if ne_median_est else np.nan,
            "sigma":      stats.get("sigma", np.nan),
            "ratio":      ratio,
        }
        for pct in ["p10", "p25", "p75", "p90"]:
            nat_pct = stats.get(pct)
            nat_median = stats.get("median")
            if nat_pct and nat_median and ne_median_est:
                ne_model[band][f"{pct}_est"] = round(
                    ne_median_est * (nat_pct / nat_median), 1)
            else:
                ne_model[band][f"{pct}_est"] = np.nan

    return ne_model


def get_ne_median_for_year(panel: pd.DataFrame, year: int) -> float | None:
    """
    Pull NE aggregate full-time median from the panel for a given ASHE year.
    Uses the ashe_ne_median_weekly column (from NM_30_1 pull in build_redcar_panel).
    Falls back to nearest available year.
    """
    # Map year to a representative quarter already in panel
    year_quarter_map = {
        2016: "2016Q2", 2017: "2017Q2",
        2018: "2018Q2", 2019: "2019Q2",
    }
    q = year_quarter_map.get(year, "2016Q2")

    # The column is ne_median_weekly_pay_gbp (from NM_30_1 in build_redcar_panel)
    candidates = [c for c in panel.columns if "ne_median_weekly" in c.lower() or
                  ("ne" in c.lower() and "median" in c.lower() and "weekly" in c.lower())]
    if not candidates:
        print(f"    WARNING: no ASHE median column in panel for year {year}")
        return None

    col = candidates[0]
    rows = panel[panel["quarter"] == q]
    if rows.empty:
        return None
    val = rows[col].dropna().iloc[0] if not rows[col].dropna().empty else None
    return float(val) if val else None


def build_lookup(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, quarters in YEAR_TO_QUARTERS.items():
        print(f"\n  Year {year}:")
        nat = parse_year(year)
        if not nat:
            for q in quarters:
                row = {"quarter": q, "ashe_age_data": "UNAVAILABLE"}
                rows.append(row)
            continue

        model_nat = aggregate_bands(nat)
        ne_median  = get_ne_median_for_year(panel, year)
        print(f"    NE aggregate median (year {year}): {ne_median}")

        if ne_median:
            ne_model = scale_to_ne(model_nat, ne_median)
        else:
            # Fallback: use national values directly
            ne_model = {b: {**{f"{p}_est": v for p, v in s.items()
                               if p in ["p10","p25","p75","p90"]},
                            "median_est": s.get("median"),
                            "sigma": s.get("sigma"),
                            "ratio": s.get("ratio_to_all")}
                        for b, s in model_nat.items()}

        for q in quarters:
            row = {"quarter": q, "ashe_age_data": "HIGH_SCALED"}
            for band in ["16_24", "25_49", "50plus"]:
                b = ne_model.get(band, {})
                col = f"ashe_age_{band}"
                row[f"{col}_median_est"] = round(b.get("median_est", np.nan), 1)
                row[f"{col}_p10_est"]    = b.get("p10_est", np.nan)
                row[f"{col}_p25_est"]    = b.get("p25_est", np.nan)
                row[f"{col}_p75_est"]    = b.get("p75_est", np.nan)
                row[f"{col}_p90_est"]    = b.get("p90_est", np.nan)
                row[f"{col}_sigma"]      = b.get("sigma", np.nan)
                row[f"ashe_age_nat_ratio_{band}"] = b.get("ratio", np.nan)
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("parse_ashe_age.py")
    print("=" * 50)
    print("Downloading ASHE Table 6 (age group, full-time, weekly gross pay)")
    print("Source: ONS ASHE Table 6 revised ZIPs, 2016-2019\n")

    panel = pd.read_csv(PANEL_PATH)
    print(f"Panel: {len(panel)} rows x {len(panel.columns)} columns")

    lookup = build_lookup(panel)

    print("\n--- National age ratios to all-employee median ---")
    ratio_cols = [c for c in lookup.columns if "nat_ratio" in c]
    display = lookup[["quarter"] + ratio_cols + ["ashe_age_data"]].drop_duplicates("quarter")
    print(display.to_string(index=False))

    print("\n--- NE estimated medians (£/week, full-time) ---")
    med_cols = [c for c in lookup.columns if "median_est" in c]
    display2 = lookup[["quarter"] + med_cols].drop_duplicates("quarter")
    print(display2.to_string(index=False))

    print(f"\nLoading panel for merge ...")
    ashe_cols = [c for c in lookup.columns if c != "quarter"]
    existing  = [c for c in ashe_cols if c in panel.columns]
    if existing:
        panel = panel.drop(columns=existing)
        print(f"  Dropped {len(existing)} existing ashe_age_ columns")

    panel = panel.merge(lookup, on="quarter", how="left")
    print(f"  After merge: {len(panel)} rows x {len(panel.columns)} columns")
    print(f"  New columns: {ashe_cols}")

    panel.to_csv(PANEL_PATH, index=False)
    print(f"\nSaved: {PANEL_PATH}")


if __name__ == "__main__":
    main()
