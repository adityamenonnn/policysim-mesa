"""
integrate_esa.py
================
Parses DWP Stat-Xplore ESA (Employment Support Allowance) downloads for
Redcar & Cleveland LA and merges quarterly ESA caseload into
math_testing_panel.csv.

How to get the data (Stat-Xplore)
-----------------------------------
DWP split ESA data into two datasets at the UC transition boundary.
You need BOTH downloads to cover 2015Q4-2019Q4.

1. Go to: https://stat-xplore.dwp.gov.uk/ and log in
2. Open Data Explorer → Employment and Support Allowance

Download 1 — "ESA - Data to February 2018":
   - Rows: Quarter, Columns: Phase of Claim
   - Filter geography: Redcar and Cleveland
   - Date: 2015 Q4 → 2018 Q1
   - Download as Excel → save as:
     /Users/adityamenon/Downloads/esa_redcar_to_feb2018.xlsx

Download 2 — "ESA - Data from May 2018":
   - Same setup
   - Date: 2018 Q2 → 2019 Q4
   - Download as Excel → save as:
     /Users/adityamenon/Downloads/esa_redcar_from_may2018.xlsx

Then run:
   cd /Users/adityamenon/Documents/PolicySim/policysim-mesa
   python3 research/redcar/integrate_esa.py

What this adds
--------------
ESA caseload data represents workers who transitioned to health-related
benefits — the "health exit" component of the model's exit bucket.
Comparing ESA levels before and after the SSI closure isolates the
shock-attributable increase in health-benefit dependency.

  esa_total_q_mean      mean monthly ESA claimants, RC LA (all phases)
  esa_support_q_mean    support group (most severely ill, not expected to work)
  esa_wrag_q_mean       work-related activity group
  esa_assess_q_mean     assessment phase (new/continuing claims being assessed)
  esa_baseline          2019 annual mean (post-shock baseline)
  esa_excess_q_mean     excess over baseline (shock-attributable, clamped >= 0)
  esa_excess_pct        excess as % of total SSI workers (3,500 approximation)
  esa_data_quality      HIGH (DWP administrative data)

Note on timing: ESA is quarterly in Stat-Xplore, not monthly. Quarterly
values are repeated for the three months of each quarter in the panel.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE       = Path(__file__).parent
PANEL_IN   = BASE / "math_testing_panel.csv"
PANEL_OUT  = BASE / "math_testing_panel.csv"

# Default download locations
XLSX_PRE  = Path("/Users/adityamenon/Downloads/esa_redcar_to_feb2018.xlsx")
XLSX_POST = Path("/Users/adityamenon/Downloads/esa_redcar_from_may2018.xlsx")

# SSI total affected workforce (for excess% calculation)
SSI_WORKERS = 3_500

# 2019 quarters used as post-shock baseline
BASELINE_QUARTERS = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]

# Quarter label mapping: Stat-Xplore exports quarters in various formats.
# We normalise them to match our panel's "quarter" column (e.g. "2016Q1").
QUARTER_ALIASES = {
    # "Feb 2016" style (monthly export) handled separately
    "jan": "Q1", "feb": "Q1", "mar": "Q1",
    "apr": "Q2", "may": "Q2", "jun": "Q2",
    "jul": "Q3", "aug": "Q3", "sep": "Q3",
    "oct": "Q4", "nov": "Q4", "dec": "Q4",
}

# Known column label fragments for each ESA phase
PHASE_MAP = {
    "esa_total":   ["total", "all"],
    "esa_support": ["support group", "support"],
    "esa_wrag":    ["work-related", "wrag"],
    "esa_assess":  ["assessment", "assess"],
}


def normalise_quarter_label(raw: str) -> str | None:
    """
    Convert Stat-Xplore quarter labels to our format.
    Handles: '2016 Q1', '2016Q1', 'Q1 2016', 'January 2016', etc.
    Returns None if not parseable.
    """
    s = str(raw).strip()
    # Already in our format
    if len(s) == 6 and s[4] == "Q" and s[:4].isdigit():
        return s

    import re
    # '2016 Q1' or '2016Q1'
    m = re.match(r"(\d{4})\s*Q(\d)", s, re.I)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"

    # 'Q1 2016'
    m = re.match(r"Q(\d)\s+(\d{4})", s, re.I)
    if m:
        return f"{m.group(2)}Q{m.group(1)}"

    # 'January 2016' → Q1 2016
    m = re.match(r"([a-z]+)\s+(\d{4})", s, re.I)
    if m:
        mon = m.group(1)[:3].lower()
        yr  = m.group(2)
        q   = QUARTER_ALIASES.get(mon)
        if q:
            return f"{yr}{q}"

    return None


def _parse_transposed(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Stat-Xplore ESA xlsx where Quarter spans columns and Phase of Claim
    are sub-columns within each quarter (the format produced when Quarter is
    placed as a Wafer and Phase of Claim as a column dimension).

    Row 10: Quarter labels (e.g. 'Nov-15') — each spanning 5 sub-columns
    Row 11: Phase labels repeating (Assessment phase, WRAG, Support group, Unknown, Total)
    Row 12: Data values for RC LA
    """
    MONTH_TO_Q = {"Nov": "Q4", "Feb": "Q1", "May": "Q2", "Aug": "Q3"}
    PHASE_LABELS = {
        "assessment": "esa_assess",
        "work related": "esa_wrag",
        "wrag": "esa_wrag",
        "support group": "esa_support",
        "support": "esa_support",
        "total": "esa_total",
    }

    row10 = raw.iloc[10].tolist()
    row11 = raw.iloc[11].tolist()

    records = []
    current_q = None

    for ci in range(1, len(row10)):
        q_raw = str(row10[ci]).strip()
        if q_raw not in ("nan", ""):
            parts = q_raw.split("-")
            if len(parts) == 2:
                mon, yr_short = parts[0].strip(), parts[1].strip()
                try:
                    yr = int("20" + yr_short) if int(yr_short) < 50 else int("19" + yr_short)
                except ValueError:
                    continue
                q_label = MONTH_TO_Q.get(mon)
                if q_label:
                    current_q = f"{yr}{q_label}"

        if current_q is None:
            continue

        phase_raw = str(row11[ci]).strip().lower()
        phase_key = None
        for label, key in PHASE_LABELS.items():
            if label in phase_raw:
                phase_key = key
                break
        if phase_key is None:
            continue

        val_raw = raw.iloc[12, ci]
        try:
            val = float(str(val_raw).replace(",", "").strip())
            val = val if not np.isnan(val) else np.nan
        except (ValueError, TypeError):
            val = np.nan

        records.append({"quarter": current_q, "phase": phase_key, "value": val})

    if not records:
        raise ValueError("Transposed parser found no data rows — check sheet structure")

    df = pd.DataFrame(records)
    wide = (df.pivot_table(index="quarter", columns="phase", values="value", aggfunc="first")
              .reset_index())
    wide.columns.name = None
    return wide


def parse_esa_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """
    Parse the Stat-Xplore ESA download.
    Handles two layouts:
      - Row-based: Quarter in first column, phases as column headers
      - Transposed: Quarter spanning columns, phases as sub-column headers (row 10/11)
    Returns a wide DataFrame: quarter + one column per ESA phase.
    """
    xl = pd.ExcelFile(xlsx_path)
    print(f"  Sheets: {xl.sheet_names}")

    raw = pd.read_excel(xlsx_path, sheet_name=xl.sheet_names[0], header=None)
    print(f"  Raw shape: {raw.shape}")

    # Detect layout: if row 10 col 0 == "Quarter" and row 11 col 0 contains "Phase"
    # → transposed format (quarters as column wafers)
    cell_r10 = str(raw.iloc[10, 0]).strip().lower()
    cell_r11 = str(raw.iloc[11, 0]).strip().lower()
    if cell_r10 == "quarter" and "phase" in cell_r11:
        print("  Detected transposed layout (quarters as columns)")
        df = _parse_transposed(raw)
        print(f"  Parsed {len(df)} quarter rows: {sorted(df.quarter.unique())}")
        return df

    # ---- Row-based layout (original path) ----
    # Find the header row (contains "Total" or "Support Group" etc.)
    header_row = None
    for i in range(min(20, len(raw))):
        row_vals = raw.iloc[i].astype(str).str.lower().tolist()
        if any("total" in v or "support" in v or "phase" in v for v in row_vals):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Cannot find header row in ESA file — check sheet structure")

    headers = raw.iloc[header_row].astype(str).tolist()
    print(f"  Header row {header_row}: {headers[:8]}")

    phase_cols = {}
    for phase_key, search_terms in PHASE_MAP.items():
        for ci, h in enumerate(headers):
            if any(t in h.lower() for t in search_terms):
                phase_cols[phase_key] = ci
                break
    print(f"  Phase columns: {phase_cols}")

    if not phase_cols:
        raise ValueError("No recognisable phase columns found. Check PHASE_MAP search terms.")

    records = []
    for row_i in range(header_row + 1, len(raw)):
        row = raw.iloc[row_i]
        quarter_raw = str(row.iloc[0]).strip()
        if quarter_raw in ["nan", "", "None"]:
            continue

        quarter = normalise_quarter_label(quarter_raw)
        if quarter is None:
            continue

        rec = {"quarter": quarter}
        for phase_key, ci in phase_cols.items():
            val_raw = row.iloc[ci]
            try:
                val = float(str(val_raw).replace(",", "").strip())
                rec[phase_key] = val if not np.isnan(val) else np.nan
            except (ValueError, TypeError):
                rec[phase_key] = np.nan
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df)} quarter rows: {sorted(df.quarter.unique())}")
    return df


def build_lookup(esa_q: pd.DataFrame) -> pd.DataFrame:
    """
    Compute baseline, excess, and percentage for each quarter.
    """
    # Ensure total column exists
    if "esa_total" not in esa_q.columns:
        # Sum available phases as total
        phase_cols = [c for c in esa_q.columns if c.startswith("esa_") and c != "esa_total"]
        if phase_cols:
            esa_q["esa_total"] = esa_q[phase_cols].sum(axis=1, min_count=1)
        else:
            esa_q["esa_total"] = np.nan

    # Baseline: mean over 2019 quarters
    baseline_rows = esa_q[esa_q["quarter"].isin(BASELINE_QUARTERS)]
    baseline = baseline_rows["esa_total"].mean() if len(baseline_rows) > 0 else np.nan
    print(f"\n  2019 baseline (mean ESA total): {baseline:.0f}" if not np.isnan(baseline) else "\n  Baseline: N/A")

    # Rename to _q_mean columns
    rename = {}
    for col in esa_q.columns:
        if col.startswith("esa_") and col != "quarter":
            rename[col] = f"{col}_q_mean"
    esa_q = esa_q.rename(columns=rename)

    esa_q["esa_baseline"]    = round(baseline, 0) if not np.isnan(baseline) else np.nan
    esa_q["esa_excess_q_mean"] = (esa_q["esa_total_q_mean"] - baseline).clip(lower=0).round(0)
    esa_q["esa_excess_pct"]  = (esa_q["esa_excess_q_mean"] / SSI_WORKERS * 100).round(1)
    esa_q["esa_data_quality"] = "HIGH"

    return esa_q


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre",  default=str(XLSX_PRE),
                        help="ESA - Data to February 2018 xlsx")
    parser.add_argument("--post", default=str(XLSX_POST),
                        help="ESA - Data from May 2018 xlsx")
    args = parser.parse_args()

    pre_path  = Path(args.pre)
    post_path = Path(args.post)

    print("integrate_esa.py")
    print("=" * 50)

    missing = []
    if not pre_path.exists():
        missing.append(f"  {pre_path}  (ESA - Data to February 2018)")
    if not post_path.exists():
        missing.append(f"  {post_path}  (ESA - Data from May 2018)")

    if missing:
        print("\nMissing files:")
        for m in missing:
            print(m)
        print("\nSee docstring at top of this file for download instructions.")
        return

    print(f"\nParsing pre-2018: {pre_path.name}")
    esa_pre  = parse_esa_xlsx(pre_path)

    print(f"\nParsing post-2018: {post_path.name}")
    esa_post = parse_esa_xlsx(post_path)

    # Combine: pre covers up to 2018Q1, post covers 2018Q2+
    # Drop any overlapping quarters from pre (keep post as authoritative for 2018+)
    post_quarters = set(esa_post["quarter"].tolist())
    esa_pre_trim = esa_pre[~esa_pre["quarter"].isin(post_quarters)]
    esa_tidy = pd.concat([esa_pre_trim, esa_post], ignore_index=True)
    esa_tidy = esa_tidy.sort_values("quarter").reset_index(drop=True)
    print(f"\nCombined: {len(esa_tidy)} quarters: {sorted(esa_tidy.quarter.tolist())}")

    lookup = build_lookup(esa_tidy)

    print(f"\nLookup columns: {lookup.columns.tolist()}")
    print(lookup[["quarter", "esa_total_q_mean", "esa_excess_q_mean",
                  "esa_excess_pct"]].to_string(index=False))

    print(f"\nLoading panel ({PANEL_IN.name}) ...")
    panel = pd.read_csv(PANEL_IN)
    print(f"  {len(panel)} rows x {len(panel.columns)} columns")

    esa_cols = [c for c in lookup.columns if c != "quarter"]
    existing = [c for c in esa_cols if c in panel.columns]
    if existing:
        panel = panel.drop(columns=existing)
        print(f"  Dropped {len(existing)} existing esa_ columns")

    panel = panel.merge(lookup, on="quarter", how="left")
    print(f"  After merge: {len(panel)} rows x {len(panel.columns)} columns")

    panel.to_csv(PANEL_OUT, index=False)
    print(f"\nSaved: {PANEL_OUT}")


if __name__ == "__main__":
    main()
