"""
fetch_redcar_extra.py
=====================
Pulls additional Nomis/ONS series and merges them as quarterly columns into
math_testing_panel.csv.

What this adds
--------------
1. North East ACC claimant count (NM_162_1) by age band:

  ne_acc_16_24_q_mean   mean monthly ACC claimants aged 16-24, North East
  ne_acc_25_49_q_mean   mean monthly ACC claimants aged 25-49, North East
  ne_acc_50plus_q_mean  mean monthly ACC claimants aged 50+, North East
  ne_acc_total_q_mean   total ACC claimants (all ages), North East
  ne_acc_16_24_pct      16-24 share of NE ACC total (%)
  ne_acc_25_49_pct      25-49 share
  ne_acc_50plus_pct     50+ share
  ne_acc_data           always HIGH (DWP administrative data via Nomis)

  rc_vs_ne_16_24_idx    RC LA 16-24 share / NE 16-24 share (>1 = RC skews younger)
  rc_vs_ne_25_49_idx    same ratio for 25-49 band
  rc_vs_ne_50plus_idx   same ratio for 50+ band

2. National monthly redundancy notifications (ONS BEAO timeseries):
  beao_redundancies_q_mean  mean monthly redundancy notifications, GB (000s)
  beao_redundancies_data    HIGH / UNAVAILABLE

Why NE and not RC LA?
  We already have RC LA ACC by age×gender from Stat-Xplore (acc_* columns).
  The NE regional total gives a baseline: if RC LA shows a higher young-adult
  share than NE, that confirms the SSI shock drove a distinctive age skew beyond
  the background claimant population.

Why not vacancies or JSA-by-duration?
  Nomis vacancy datasets (NM_5_1, NM_19_1-24_1, NM_89_1) do not cover 2015-2019
  at any sub-national geography: the series end by 2012.
  JSA duration stock data (NM_2_1) ends October 1998.
  These are not bugs — Nomis retired these series when UC rolled out.
  For vacancy data, use the ONS LFS regional tables or DWP Stat-Xplore directly.

Usage:
  cd /Users/adityamenon/Documents/PolicySim/policysim-mesa
  python3 research/redcar/fetch_redcar_extra.py
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
CACHE_DIR  = BASE / "_nomis_cache"
PANEL_PATH = BASE / "math_testing_panel.csv"
NOMIS      = "https://www.nomisweb.co.uk/api/v01/dataset"

CACHE_DIR.mkdir(exist_ok=True)

NE_GEO = "E12000001"   # North East ITL1

# NM_162_1 age codes that match our model bands
# (verified via discover: 2=16-24, 3=25-49, 4=50+)
AGE_CODES = {2: "16_24", 3: "25_49", 4: "50plus"}

# ── quarters ───────────────────────────────────────────────────────────────────
QUARTERS = {
    "2015Q4": ["October 2015",   "November 2015",  "December 2015"],
    "2016Q1": ["January 2016",   "February 2016",  "March 2016"],
    "2016Q2": ["April 2016",     "May 2016",       "June 2016"],
    "2016Q3": ["July 2016",      "August 2016",    "September 2016"],
    "2016Q4": ["October 2016",   "November 2016",  "December 2016"],
    "2017Q1": ["January 2017",   "February 2017",  "March 2017"],
    "2017Q2": ["April 2017",     "May 2017",       "June 2017"],
    "2017Q3": ["July 2017",      "August 2017",    "September 2017"],
    "2017Q4": ["October 2017",   "November 2017",  "December 2017"],
    "2018Q1": ["January 2018",   "February 2018",  "March 2018"],
    "2018Q2": ["April 2018",     "May 2018",       "June 2018"],
    "2018Q3": ["July 2018",      "August 2018",    "September 2018"],
    "2018Q4": ["October 2018",   "November 2018",  "December 2018"],
    "2019Q1": ["January 2019",   "February 2019",  "March 2019"],
    "2019Q2": ["April 2019",     "May 2019",       "June 2019"],
    "2019Q3": ["July 2019",      "August 2019",    "September 2019"],
    "2019Q4": ["October 2019",   "November 2019",  "December 2019"],
}


# ── helpers ────────────────────────────────────────────────────────────────────

def get_csv(url: str, cache_file: Path) -> pd.DataFrame:
    """Fetch a Nomis CSV endpoint (rate-limited). Uses cache if present."""
    if cache_file.exists():
        print(f"    [cache] {cache_file.name}")
        return pd.read_csv(cache_file)
    print(f"    [fetch] {url[:90]}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    cache_file.write_bytes(r.content)
    time.sleep(1.2)
    df = pd.read_csv(cache_file)
    if df.empty:
        cache_file.unlink()   # don't cache empty results
        raise ValueError(f"Empty response from Nomis for {cache_file.name}")
    return df


def to_quarter_means(monthly_df: pd.DataFrame,
                     month_col: str,
                     value_cols: list[str]) -> pd.DataFrame:
    """
    Collapse a monthly DataFrame to quarterly means.
    month_col must contain strings matching QUARTERS values
    (e.g. 'October 2015').
    """
    rows = []
    for q, months in QUARTERS.items():
        subset = monthly_df[monthly_df[month_col].isin(months)]
        row = {"quarter": q}
        for col in value_cols:
            vals = pd.to_numeric(subset[col], errors="coerce")
            row[col] = round(vals.mean(), 1) if not vals.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── NE ACC by age band (NM_162_1) ─────────────────────────────────────────────

def fetch_ne_acc_by_age() -> pd.DataFrame:
    """
    Pull NM_162_1 claimant count for North East, broken down by model age bands.
    measure=1 = claimant count (not seasonally adjusted).
    age codes 2,3,4 = 16-24, 25-49, 50+ (verified via discover).
    """
    print("\n[1] NE ACC by age band (NM_162_1) ...")
    age_param = ",".join(str(c) for c in AGE_CODES)
    url = (f"{NOMIS}/NM_162_1.data.csv"
           f"?geography={NE_GEO}"
           f"&date=2015-10...2019-12"
           f"&gender=0"
           f"&age={age_param}"
           f"&measure=1&measures=20100"
           f"&select=date_name,age_name,obs_value")
    raw = get_csv(url, CACHE_DIR / "ne_acc_by_age_raw.csv")
    raw.columns = ["month", "age_label", "claimants"]
    raw["claimants"] = pd.to_numeric(raw["claimants"], errors="coerce")

    # Map Nomis age labels to our short codes
    label_map = {
        "Aged 16-24": "16_24",
        "Aged 25-49": "25_49",
        "Aged 50+":   "50plus",
    }
    raw["band"] = raw["age_label"].map(label_map)
    raw = raw[raw["band"].notna()]

    # Pivot: one column per age band per month
    pivot = (raw.pivot_table(index="month", columns="band",
                             values="claimants", aggfunc="sum")
               .reset_index())
    band_cols = ["16_24", "25_49", "50plus"]
    for b in band_cols:
        if b not in pivot.columns:
            pivot[b] = np.nan

    pivot["total"] = pivot[band_cols].sum(axis=1)

    q = to_quarter_means(pivot, "month", band_cols + ["total"])
    q = q.rename(columns={
        "16_24":  "ne_acc_16_24_q_mean",
        "25_49":  "ne_acc_25_49_q_mean",
        "50plus": "ne_acc_50plus_q_mean",
        "total":  "ne_acc_total_q_mean",
    })

    # Age shares of NE total
    for band, col in [("16_24", "ne_acc_16_24_q_mean"),
                      ("25_49", "ne_acc_25_49_q_mean"),
                      ("50plus", "ne_acc_50plus_q_mean")]:
        q[f"ne_acc_{band}_pct"] = (
            q[col] / q["ne_acc_total_q_mean"] * 100
        ).round(1)

    q["ne_acc_data"] = "HIGH"
    print(f"  OK — {len(raw)} monthly×band rows")
    print(f"  2015Q4: total={q.loc[q.quarter=='2015Q4','ne_acc_total_q_mean'].values[0]:,.0f}, "
          f"16-24={q.loc[q.quarter=='2015Q4','ne_acc_16_24_pct'].values[0]}%")
    return q


# ── BEAO: national redundancy notifications (ONS timeseries) ──────────────────

# Month label used by ONS timeseries API → matches QUARTERS month strings
_ONS_MONTH_MAP = {
    "January": "January",   "February": "February", "March": "March",
    "April": "April",       "May": "May",            "June": "June",
    "July": "July",         "August": "August",      "September": "September",
    "October": "October",   "November": "November",  "December": "December",
}


def fetch_beao() -> pd.DataFrame:
    """
    ONS BEAO timeseries: monthly redundancy notifications, GB (000s).
    3-month rolling average published by ONS (e.g. Oct 2015 = Aug-Oct mean).
    Returns quarterly lookup with beao_redundancies_q_mean.
    """
    print("\n[2] BEAO redundancy notifications (ONS timeseries) ...")
    cache = CACHE_DIR / "beao_monthly.csv"

    if cache.exists():
        print(f"    [cache] {cache.name}")
        monthly = pd.read_csv(cache)
    else:
        url = ("https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork"
               "/redundancies/timeseries/beao/lms/data")
        headers = {"User-Agent": "Mozilla/5.0 (policysim-mesa data pull)"}
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  FAILED: {e}")
            q = pd.DataFrame({"quarter": list(QUARTERS)})
            q["beao_redundancies_q_mean"] = np.nan
            q["beao_redundancies_data"] = "UNAVAILABLE"
            return q

        months_raw = data.get("months", [])
        records = []
        for pt in months_raw:
            year = pt.get("year", "")
            month = pt.get("month", "")
            val = pt.get("value", "")
            if year in [str(y) for y in range(2015, 2020)]:
                try:
                    records.append({"month": f"{month} {year}",
                                    "beao": float(val)})
                except ValueError:
                    pass

        monthly = pd.DataFrame(records)
        monthly.to_csv(cache, index=False)
        print(f"    [fetch] {url[:80]}...")

    q = to_quarter_means(monthly, "month", ["beao"])
    q = q.rename(columns={"beao": "beao_redundancies_q_mean"})
    q["beao_redundancies_data"] = "HIGH"
    print(f"  OK — {len(monthly)} monthly rows in 2015-2019")
    print(f"  2015Q4 mean: {q.loc[q.quarter=='2015Q4','beao_redundancies_q_mean'].values[0]:.1f}k notifications")
    return q


# ── RC vs NE comparison indices ────────────────────────────────────────────────

def build_rc_vs_ne_index(panel: pd.DataFrame,
                          ne_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RC LA age share / NE age share for each band and quarter.
    Uses acc_*_pct columns (RC LA, from Stat-Xplore) and ne_acc_*_pct (NE).
    Index > 1 means RC LA is more heavily weighted toward that band than NE.
    Only computed for rows where RC LA data exists (all/all row type).
    """
    # Build a per-quarter mapping from the ne lookup
    ne_q = ne_lookup[["quarter", "ne_acc_16_24_pct",
                       "ne_acc_25_49_pct", "ne_acc_50plus_pct"]].copy()

    # Drop pre-existing ne_acc_pct columns from panel to avoid suffix conflicts
    conflict_cols = [c for c in ne_q.columns if c != "quarter" and c in panel.columns]
    panel_clean = panel.drop(columns=conflict_cols)
    merged = panel_clean.merge(ne_q, on="quarter", how="left")
    for band in ["16_24", "25_49", "50plus"]:
        rc_col = f"acc_{band}_pct"
        ne_col = f"ne_acc_{band}_pct"
        idx_col = f"rc_vs_ne_{band}_idx"
        merged[idx_col] = (merged[rc_col] / merged[ne_col]).round(3)

    # Only keep the new columns to merge back
    idx_cols = [c for c in merged.columns if c.startswith("rc_vs_ne_")]
    return merged[["quarter"] + idx_cols + list(ne_q.columns[1:])].drop_duplicates("quarter")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("fetch_redcar_extra.py")
    print("=" * 50)

    ne_acc_q  = fetch_ne_acc_by_age()
    beao_q    = fetch_beao()

    # Load panel
    print(f"\nLoading panel ({PANEL_PATH.name}) ...")
    panel = pd.read_csv(PANEL_PATH)
    print(f"  {len(panel)} rows x {len(panel.columns)} columns")

    ne_acc_q = ne_acc_q.merge(beao_q, on="quarter", how="left")

    # Build RC vs NE indices (needs existing acc_* columns from Stat-Xplore)
    acc_cols_present = [c for c in panel.columns if c.startswith("acc_")]
    if acc_cols_present:
        print("\nComputing RC vs NE comparison indices ...")
        idx_df = build_rc_vs_ne_index(panel, ne_acc_q)
        ne_acc_q = ne_acc_q.merge(
            idx_df[["quarter"] + [c for c in idx_df.columns if c.startswith("rc_vs_ne_")]],
            on="quarter", how="left"
        )
        print(f"  Indices: {[c for c in ne_acc_q.columns if c.startswith('rc_vs_ne_')]}")
    else:
        print("\n  (acc_* columns not found — run integrate_statxplore.py first for RC vs NE indices)")

    # Drop any existing columns from a previous run
    new_cols = [c for c in ne_acc_q.columns if c != "quarter"]
    existing = [c for c in new_cols if c in panel.columns]
    if existing:
        panel = panel.drop(columns=existing)
        print(f"\n  Dropped {len(existing)} existing columns for clean re-run")

    panel = panel.merge(ne_acc_q, on="quarter", how="left")
    print(f"  After merge: {len(panel)} rows x {len(panel.columns)} columns")
    added = [c for c in panel.columns if c in new_cols]
    print(f"  New columns: {added}")

    panel.to_csv(PANEL_PATH, index=False)
    print(f"\nSaved: {PANEL_PATH}")

    # Summary
    print("\n--- NE ACC age shares vs RC LA ACC age shares ---")
    print("  (values from all/all row type only)")
    summary_cols = ["quarter",
                    "ne_acc_16_24_pct", "ne_acc_25_49_pct", "ne_acc_50plus_pct",
                    "rc_vs_ne_16_24_idx", "rc_vs_ne_25_49_idx", "rc_vs_ne_50plus_idx"]
    available = [c for c in summary_cols if c in ne_acc_q.columns]
    print(ne_acc_q[available].to_string(index=False))


if __name__ == "__main__":
    main()
