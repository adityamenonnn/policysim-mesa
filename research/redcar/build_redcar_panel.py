"""
build_redcar_panel.py
=====================
Builds the Redcar 2015 quarterly demographic panel dataset.

Output: redcar_quarterly_panel.csv

Column structure:
  - All 22 columns from displacement_evidence.csv (same schema, same field names)
  - Plus extra columns for quarterly breakdown, gender, ONS claimant data,
    wage context, and composite outcome anchors

This means the output rows can be appended directly to displacement_evidence.csv
if needed, and all extra columns simply extend the schema.

Data sources combined:
  - displacement_evidence.csv        original research rows (13 Redcar rows)
  - ons_data/claimant_count_raw.csv  North East ITL1 monthly claimant count
  - ons_data/claimant_age_raw.csv    North East ITL1 claimant count by age band
  - ons_data/ashe_raw.csv            North East annual wages (ASHE)
  - ons_data/aps_raw.csv             North East APS employment/unemployment rates
  - Nomis API (fetched here)         Gender x age splits (NE ITL1)
                                     Redcar & Cleveland LA claimant count
                                     Redcar & Cleveland LA claimant count by age

Key data limitations:
  - All Nomis data is AREA-LEVEL. It is not individual SSI worker tracking.
  - outcome_retrain_%, outcome_share_similar_%, outcome_underemployed_%, outcome_exit_%
    cannot be split from public data. Only anchors are embedded (see OUTCOME_ANCHORS).
  - DWP microdata (ADRUK restricted access) needed for a clean 4-bucket split.
  - SSI age/gender shares are priors from evidence, not microdata.
  - Nomis age band "25-49" does not match model band "25-44" — flagged in notes.

Run: python3 build_redcar_panel.py
"""

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = Path(__file__).parent   # research/redcar/
RESEARCH = BASE.parent             # research/
ONS      = RESEARCH / "ons_data"
EVIDENCE = RESEARCH / "displacement_evidence.csv"
OUT      = BASE / "redcar_quarterly_panel.csv"

NOMIS = "https://www.nomisweb.co.uk/api/v01/dataset"

# Geography codes
REDCAR_LA  = "E06000003"   # Redcar & Cleveland unitary authority
NE_ITL1    = "E12000001"   # North East England

# ---------------------------------------------------------------------------
# SSI workforce priors
# ---------------------------------------------------------------------------
SSI_N = 2070
# Age band shares. Source: DWP/Task Force "50%+ seeking work aged 45+".
# 16-24 and 25-49 estimated from UK steel sector age profile — LOW confidence.
# Nomis returns "Aged 25-49" not "Aged 25-44"; documented in notes per row.
AGE_SHARE = {"16-24": 0.10, "25-49": 0.40, "50+": 0.50}
# Gender share. Source: UK Steel industry workforce stats (~95% male).
GENDER_SHARE = {"male": 0.95, "female": 0.05}

# ---------------------------------------------------------------------------
# Income bands — SSI workforce distribution and outcome adjustments
# ---------------------------------------------------------------------------
# Source for shares: ASHE NE 2015 percentiles (p10=£288/wk, p25=£359/wk,
# median=£486/wk, p75=£679/wk, p90=£882/wk) + UK Steel sector wage profile.
# Outcome adjustments are relative to the £25k-£35k aggregate baseline at
# Q4 2016 (retrain=6, share_similar=13, underemployed=52, exit=26, seeking=3).
# Cross-episode basis: China Shock ADH 2021 — "low-wage → exit/underemployed;
# high-wage → share_similar". Hartz: 80%+ re-employment to lower-paying employer.
# All outcome_adj values are percentage-point shifts from the aggregate mid.
INCOME_BANDS = {
    "<£20k": dict(
        share       = 0.05,   # ~5% — semi-skilled, admin, cleaners
        skill_tier  = "low_skilled",
        description = "Semi-skilled process support, admin, cleaners. "
                      "ASHE NE p10 (£288/wk=£15k/yr). Lowest-paid SSI cohort. "
                      "Estimated 5% of workforce — LOW confidence.",
        outcome_adj = dict(retrain=-2, share_similar=-6, underemployed=+3, exit=+6, still_seeking=-1),
        # Adj rationale: China Shock: low-wage workers → exit during mass layoffs.
        # Less savings buffer → faster benefit exit into any available work (underemployed).
        # Lower pension entitlement → some exit to ESA/disability rather than retirement.
    ),
    "£20k-£25k": dict(
        share       = 0.15,  # ~15% — semi-skilled operators, junior technicians
        skill_tier  = "skilled_manual",
        description = "Semi-skilled process operators, junior technicians. "
                      "ASHE NE p25-median range (£359-£486/wk). "
                      "Estimated 15% of workforce — LOW confidence.",
        outcome_adj = dict(retrain=-1, share_similar=-4, underemployed=+2, exit=+4, still_seeking=-1),
    ),
    "£25k-£35k": dict(
        share       = 0.55,  # ~55% — the majority cohort, skilled process operators
        skill_tier  = "skilled_manual",
        description = "Skilled process operators, blast furnace operators. "
                      "ASHE NE median-p75 range (£486-£679/wk). "
                      "PRIMARY ESTIMATE — majority of SSI workforce. "
                      "All aggregate anchors (25% still-seeking, 93% off benefits) apply directly.",
        outcome_adj = dict(retrain=0, share_similar=0, underemployed=0, exit=0, still_seeking=0),
        # This is the baseline — no adjustment applied.
    ),
    "£35k-£50k": dict(
        share       = 0.20,  # ~20% — senior operators, supervisors, maintenance engineers
        skill_tier  = "skilled_manual",
        description = "Senior operators, shift supervisors, maintenance engineers, technicians. "
                      "ASHE NE p75-p90 range (£679-£882/wk). "
                      "Estimated 20% of workforce — LOW confidence.",
        outcome_adj = dict(retrain=+1, share_similar=+7, underemployed=-6, exit=-4, still_seeking=+2),
        # Adj rationale: China Shock: high-wage → share_similar (comparable employers).
        # More transferable skills. Can afford longer job search (higher still_seeking).
    ),
    "£50k+": dict(
        share       = 0.05,  # ~5% — management, senior engineers, plant managers
        skill_tier  = "professional",
        description = "Plant management, senior engineers, operations managers. "
                      "Above ASHE NE p90 (£882/wk=£45.9k/yr). "
                      "Estimated 5% of workforce — LOW confidence. "
                      "Likely had enhanced redundancy packages separate from main SSI terms.",
        outcome_adj = dict(retrain=-1, share_similar=+15, underemployed=-14, exit=-6, still_seeking=+6),
        # Adj rationale: Management skills highly transferable across sectors.
        # High still_seeking — can afford long search; high share_similar — comparable employers exist.
    ),
}

# ---------------------------------------------------------------------------
# Subsidy by age band (employer hiring subsidy was age-tiered)
# ---------------------------------------------------------------------------
# Source: DWP SSI JSA Statistics / Gov.uk 6-month press release
SUBSIDY_BY_AGE = {
    "16-24": (
        "£11k employer hiring subsidy (under-45 tier); "
        "Routes to Work pilot £7.5m (Tees Valley-wide); "
        "Employer Ownership of Skills pilot; "
        "£46m total government package"
    ),
    "25-49": (
        "£11k employer hiring subsidy (under-45 tier); "
        "Routes to Work pilot £7.5m (Tees Valley-wide); "
        "£46m total government package"
    ),
    "50+": (
        "£15k employer hiring subsidy (45+ tier — highest rate, recognising harder-to-place status); "
        "Routes to Work pilot £7.5m (Tees Valley-wide); "
        "£46m total government package"
    ),
    "all": (
        "£11k-£15k employer hiring subsidy (age-tiered: £11k under-45, £15k aged 45+); "
        "Routes to Work pilot £7.5m (Tees Valley-wide); "
        "£46m total government package (announced Oct 2015)"
    ),
}

# ---------------------------------------------------------------------------
# UBI level — no UBI operational in UK 2015, but document the actual baseline
# ---------------------------------------------------------------------------
UBI_LEVEL = (
    "N/A — no UBI operational in UK Oct 2015. "
    "Actual baseline safety net: JSA £73.10/wk (adult rate) or £57.90/wk (under-25); "
    "ESA (support group) £109.65/wk; UC standard allowance £317.82/month. "
    "For model calibration: ubi_level = 0. "
    "UBI counterfactuals are simulation scenarios, not historical data."
)

# Constant fields that are the same for every Redcar row
REDCAR_CONSTANTS = dict(
    episode_id  = "redcar_2015",
    year_window = "2015-2017",
    geography   = "Redcar / Teesside",
    region      = "North East England",
    sector      = "primary_metals / steel_manufacturing",
)

# ---------------------------------------------------------------------------
# Outcome anchors from evidence CSV and literature
# Keyed by quarter. Only hard data points — nothing estimated.
# ---------------------------------------------------------------------------
ANCHORS = {
    "2015Q4": dict(
        # Closure: 2 Oct 2015. End of Q4 = Dec 2015.
        never_claimed_jsa_pct   = 35.0,
        # 720/2070 never claimed JSA — DWP lists possible reasons: retired,
        # immediately re-employed, on ESA/UC, left UK, on partner's claim.
        # Cannot be split into model buckets without microdata.
        anchor_source = "DWP SSI JSA Statistics March 2016",
        anchor_url    = "https://assets.publishing.service.gov.uk/media/5a757a2ae5274a1622e22197/sahaviriya-steel-ind-ad-hoc-jsa-stats.pdf",
        data_quality  = "MEDIUM",
    ),
    "2016Q1": dict(
        # ~5 months post-closure. DWP reporting window ends Feb 2016.
        outcome_still_seeking_pct = 25.0,   # 540/2070 still on JSA
        never_claimed_jsa_pct     = 35.0,   # 720/2070 never claimed JSA
        # 40% left JSA by Feb 2016 but DWP warns ≠ re-employed (could be ESA/UC/retired)
        anchor_source = "DWP SSI JSA Statistics March 2016",
        anchor_url    = "https://assets.publishing.service.gov.uk/media/5a757a2ae5274a1622e22197/sahaviriya-steel-ind-ad-hoc-jsa-stats.pdf",
        data_quality  = "HIGH",
    ),
    "2016Q2": dict(
        # ~6 months. Gov.uk 6-month press release.
        training_courses_delivered_n = 4300,   # course counts NOT worker counts
        job_advice_recipients_n      = 2988,   # near-universal takeup
        startup_advice_n             = 418,    # 20% of 2070
        anchor_source = "Gov.uk 6-month press release",
        anchor_url    = "https://www.gov.uk/government/news/business-minister-returns-to-redcar-6-months-after-steel-plant-closure",
        data_quality  = "MEDIUM",
    ),
    "2016Q4": dict(
        # ~12 months. Task Force One Year On report, Sept 2016.
        # Task Force base = 2150 (all-benefit claimants, larger than DWP 2070).
        off_all_benefits_pct        = 93.0,   # 1990/2150 off ALL benefits
        still_on_benefits_pct       =  7.0,   # ~160 workers — likely older/health-affected
        self_employment_n           =  172,   # new businesses launched
        self_employment_pct         =  8.3,   # 172/2070
        training_courses_total_n    = 15510,  # course counts NOT worker counts
        anchor_source = "SSI Task Force One Year On Report Sept 2016",
        anchor_url    = "https://teesvalley-ca.gov.uk/news/ssi-task-force-publishes-one-year-report/",
        data_quality  = "HIGH",
    ),
    "2017Q4": dict(
        # ~24 months. Teesside Live / Evidently Substack (Tees Valley CA data).
        self_employment_n           =  315,
        self_employment_pct         =  15.2,   # 315/2070
        area_wage_change_pct        =  -0.5,   # real terms 2015-2017
        ne_wage_change_pct          =   1.6,   # rest of NE for comparison (2.1pp gap)
        manual_jobs_share_pre_pct   =  10.0,   # % of local employment Dec 2014
        manual_jobs_share_post_pct  =   8.0,   # % of local employment Dec 2016
        claimant_count_recovered    = True,    # returned to baseline ~Oct 2017
        anchor_source = "Teesside Live Sept 2025; Evidently Substack (Tees Valley CA data)",
        anchor_url    = "https://evidently.substack.com/p/3-lessons-from-redcar-steelworks",
        data_quality  = "MEDIUM",
    ),
}

# ---------------------------------------------------------------------------
# Month name -> quarter
# ---------------------------------------------------------------------------
_M = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
      "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}

def to_q(s):
    p = s.strip().split(); m = _M[p[0]]; q = (m-1)//3+1
    return f"{p[1]}Q{q}"

# ---------------------------------------------------------------------------
# Nomis helpers
# ---------------------------------------------------------------------------
def nomis_csv(url):
    r = requests.get(url, timeout=120); r.raise_for_status()
    time.sleep(1.2)
    return pd.read_csv(StringIO(r.text))

def fetch_and_cache(path, url, cols):
    if path.exists():
        print(f"  cached: {path.name}")
        return pd.read_csv(path)
    df = nomis_csv(url); df.columns = cols
    df.to_csv(path, index=False)
    print(f"  fetched: {path.name} ({len(df)} rows)")
    return df

# ---------------------------------------------------------------------------
# Load ONS data
# ---------------------------------------------------------------------------
def load_ons():
    cc = pd.read_csv(ONS/"claimant_count_raw.csv")
    cc.columns = ["month","region","region_code","claimant_count"]

    ca = pd.read_csv(ONS/"claimant_age_raw.csv")
    ca.columns = ["month","region","age_band","claimants"]

    ashe = pd.read_csv(ONS/"ashe_raw.csv")
    ashe.columns = ["year","region","measure","weekly_pay_gbp"]

    aps = pd.read_csv(ONS/"aps_raw.csv")
    aps.columns = ["period","region","variable","value"]

    return cc, ca, ashe, aps

# ---------------------------------------------------------------------------
# Fetch extra Nomis data
# ---------------------------------------------------------------------------
def fetch_extra():
    ga = fetch_and_cache(
        ONS/"claimant_gender_age_ne.csv",
        (f"{NOMIS}/NM_162_1.data.csv?geography={NE_ITL1}&date=2014-01...2019-12"
         f"&gender=1,2&age=1,2,3,4,5,6,7,8&measure=1&measures=20100"
         f"&select=date_name,geography_name,gender_name,age_name,obs_value"),
        ["month","geography","gender","age_band","claimants"],
    )
    rc_tot = fetch_and_cache(
        ONS/"claimant_redcar_cleveland.csv",
        (f"{NOMIS}/NM_162_1.data.csv?geography={REDCAR_LA}&date=2014-01...2019-12"
         f"&gender=0&age=0&measure=1&measures=20100"
         f"&select=date_name,geography_name,geography_code,obs_value"),
        ["month","geography","geography_code","claimant_count"],
    )
    rc_age = fetch_and_cache(
        ONS/"claimant_redcar_cleveland_age.csv",
        (f"{NOMIS}/NM_162_1.data.csv?geography={REDCAR_LA}&date=2014-01...2019-12"
         f"&gender=0&age=1,2,3,4,5,6,7,8&measure=1&measures=20100"
         f"&select=date_name,geography_name,age_name,obs_value"),
        ["month","geography","age_band","claimants"],
    )
    return ga, rc_tot, rc_age

# ---------------------------------------------------------------------------
# Aggregate monthly -> quarterly mean
# ---------------------------------------------------------------------------
def to_quarterly(df, group_cols, val_col):
    df = df.copy()
    df["quarter"] = df["month"].apply(to_q)
    return df.groupby(["quarter"]+group_cols, as_index=False)[val_col].mean().round(1)

def baseline(q_df, val_col, before="2015Q4"):
    return q_df[q_df["quarter"] < before][val_col].mean()

# ---------------------------------------------------------------------------
# Build lookup for APS (best-match rolling period for a given quarter)
# ---------------------------------------------------------------------------
def build_aps_lookup(ne_aps):
    """
    APS uses rolling 12-month periods like "Oct 2015-Sep 2016".
    For each calendar quarter, pick the period whose centre is closest.
    Returns dict: quarter -> {employment_rate, unemployment_rate, activity_rate}
    """
    unemp = ne_aps[ne_aps["variable"] == "Unemployment rate - aged 16+"]
    emp   = ne_aps[ne_aps["variable"] == "Employment rate - aged 16-64"]
    act   = ne_aps[ne_aps["variable"] == "Economic activity rate - aged 16-64"]

    # APS periods use 3-letter abbreviations: "Apr 2015-Mar 2016"
    _M3 = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
            "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

    def parse_start(s):
        part = s.split("-")[0].strip()
        bits = part.split()
        return int(bits[1]), _M3[bits[0]]

    quarters = [
        "2015Q4","2016Q1","2016Q2","2016Q3",
        "2016Q4","2017Q1","2017Q2","2017Q3","2017Q4",
        "2018Q1","2018Q2","2018Q3","2018Q4",
        "2019Q1","2019Q2","2019Q3","2019Q4",
    ]
    q_to_month = {
        "2015Q4":(2015,11),"2016Q1":(2016,2),"2016Q2":(2016,5),
        "2016Q3":(2016,8),"2016Q4":(2016,11),"2017Q1":(2017,2),
        "2017Q2":(2017,5),"2017Q3":(2017,8),"2017Q4":(2017,11),
        "2018Q1":(2018,2),"2018Q2":(2018,5),"2018Q3":(2018,8),"2018Q4":(2018,11),
        "2019Q1":(2019,2),"2019Q2":(2019,5),"2019Q3":(2019,8),"2019Q4":(2019,11),
    }
    result = {}
    for q, (ty, tm) in q_to_month.items():
        best_period, best_dist = None, 999
        for _, row in unemp.iterrows():
            sy, sm = parse_start(row["period"])
            # centre of 12-month window = start + 6 months
            cy, cm = sy + (sm+5)//12, (sm+5)%12 + 1
            dist = abs((ty*12+tm) - (cy*12+cm))
            if dist < best_dist:
                best_dist = dist
                best_period = row["period"]
        def get_val(df, period):
            r = df[df["period"]==period]
            return round(r["value"].values[0], 1) if len(r) else None
        result[q] = dict(
            aps_period      = best_period,
            ne_unemp_rate   = get_val(unemp, best_period),
            ne_emp_rate     = get_val(emp,   best_period),
            ne_activity_rate= get_val(act,   best_period),
        )
    return result

# ---------------------------------------------------------------------------
# Build ASHE lookup: quarter -> ne_median_weekly_pay_gbp
# (ASHE is annual; use the year that covers the quarter)
# ---------------------------------------------------------------------------
def build_ashe_lookup(ne_ashe):
    medians = ne_ashe[ne_ashe["measure"]=="Value"][["year","weekly_pay_gbp"]].copy()
    medians["year"] = medians["year"].astype(int)
    result = {}
    for q in ["2015Q4","2016Q1","2016Q2","2016Q3",
               "2016Q4","2017Q1","2017Q2","2017Q3","2017Q4",
               "2018Q1","2018Q2","2018Q3","2018Q4",
               "2019Q1","2019Q2","2019Q3","2019Q4"]:
        y = int(q[:4])
        row = medians[medians["year"]==y]
        result[q] = round(row["weekly_pay_gbp"].values[0], 1) if len(row) else None
    return result

# ---------------------------------------------------------------------------
# Build one panel row (matches evidence CSV columns + extras)
# ---------------------------------------------------------------------------
def make_row(quarter, months_post, gender, age_band,
             anchors_q, ssi_n_cell,
             rc_la_q_mean, rc_la_excess, rc_la_base,
             ne_cc_q_mean, ne_cc_excess, ne_cc_base,
             rc_age_claimants, rc_age_pct,
             ne_gender_claimants, ne_gender_pct_within_age,
             ne_unemp_rate, ne_emp_rate, ne_activity_rate, aps_period,
             ne_median_weekly_pay,
             source_citation, source_url, data_quality,
             income_band="£25k-£35k est.", skill_tier="skilled_manual",
             income_band_note=None):

    a = anchors_q

    # ---- outcome_still_seeking_% ----
    # Q1 2016: 25% still on JSA (DWP). Carry this forward as a ceiling for
    # still_seeking in that quarter. Other quarters: None (not observed).
    still_seeking = a.get("outcome_still_seeking_pct")

    # ---- outcome_exit_% ----
    # Q4 2016: 7% lower bound (still on all benefits). Not a clean exit rate.
    # Cannot distinguish early retirement from long-term unemployment.
    exit_pct = a.get("still_on_benefits_pct")   # floor only

    # ---- Build the notes field ----
    # Rich, model-usable description of what this row represents.
    note_parts = []

    if age_band != "all":
        note_parts.append(
            f"AGE BAND: {age_band}. "
            f"Nomis band used: {'Aged '+age_band} "
            f"{'(NOTE: Nomis uses 25-49, not 25-44 — document mismatch)' if age_band=='25-49' else ''}. "
            f"Estimated SSI workers in this cell: {ssi_n_cell} "
            f"(prior: {AGE_SHARE.get(age_band,'?')*100:.0f}% of 2070). "
        )
    if gender != "all":
        note_parts.append(
            f"GENDER: {gender}. Prior: {GENDER_SHARE.get(gender,0)*100:.0f}% of SSI workforce "
            f"(UK steel industry statistic, not SSI-specific microdata). "
        )

    note_parts.append(
        f"QUARTER: {quarter} ({months_post} months post-closure, Oct 2 2015). "
    )

    # ONS claimant data
    if rc_la_q_mean is not None:
        note_parts.append(
            f"REDCAR & CLEVELAND LA (E06000003) claimant count: {rc_la_q_mean:.0f} quarterly avg "
            f"(baseline {rc_la_base:.0f}; excess vs baseline: {rc_la_excess:+.0f}). "
        )
    if rc_age_claimants is not None:
        note_parts.append(
            f"RC LA age band claimants: {rc_age_claimants:.0f} ({rc_age_pct:.1f}% of LA total). "
        )
    if ne_gender_claimants is not None:
        note_parts.append(
            f"NE ITL1 gender claimants in this age band: {ne_gender_claimants:.0f} "
            f"({ne_gender_pct_within_age:.1f}% of NE age-band total). "
            f"NOTE: NE gender split mixes steel and non-steel workers. "
            f"SSI gender prior (95% male) used for ssi_est_workers_this_cell instead. "
        )

    # APS
    if ne_unemp_rate is not None:
        note_parts.append(
            f"NE LABOUR MARKET (APS {aps_period}): unemployment {ne_unemp_rate}%, "
            f"employment {ne_emp_rate}%, activity {ne_activity_rate}%. "
        )

    # ASHE wages
    if ne_median_weekly_pay is not None:
        note_parts.append(
            f"NE MEDIAN WAGE (ASHE {quarter[:4]}): £{ne_median_weekly_pay}/week "
            f"(£{ne_median_weekly_pay*52:.0f}/yr). "
            f"SSI workers earned above NE median (skilled manual, union rates, est. £25k-£35k). "
        )

    # Outcome anchors for this quarter
    if "outcome_still_seeking_pct" in a:
        note_parts.append(
            f"OUTCOME ANCHOR — still_seeking: {a['outcome_still_seeking_pct']}% (540/2070 still on JSA, DWP Feb 2016). "
            f"35% (720/2070) never claimed JSA — cannot split into model buckets "
            f"(could be retired, immediately re-employed, on ESA/UC, left UK). "
            f"40% left JSA by Feb 2016 but DWP warns ≠ re-employed. "
        )
    if "off_all_benefits_pct" in a:
        note_parts.append(
            f"OUTCOME ANCHOR — 12 months (Task Force, n=2150): "
            f"{a['off_all_benefits_pct']}% off ALL benefits; "
            f"{a['still_on_benefits_pct']}% (~160 workers) still on benefits — likely older/health-affected. "
            f"CRITICAL: cannot split {a['off_all_benefits_pct']}% into retrain/share_similar/underemployed/exit "
            f"without DWP microdata. "
            f"Self-employment: {a['self_employment_n']} new businesses ({a['self_employment_pct']}% of 2070). "
        )
    if "training_courses_delivered_n" in a:
        note_parts.append(
            f"6-month data: {a['training_courses_delivered_n']} training courses delivered "
            f"(NOT worker counts — one worker may take multiple courses). "
            f"{a['job_advice_recipients_n']} received job advice (near-universal). "
        )
    if "area_wage_change_pct" in a:
        note_parts.append(
            f"UNDEREMPLOYMENT PROXY: area wages {a['area_wage_change_pct']:+.1f}% real 2015-2017 "
            f"vs {a['ne_wage_change_pct']:+.1f}% rest of NE (2.1pp gap). "
            f"Manual jobs share: {a['manual_jobs_share_pre_pct']}% → {a['manual_jobs_share_post_pct']}% "
            f"of local employment (occupational downshift into lower-paid service jobs). "
            f"Claimant count returned to pre-closure baseline ~Oct 2017 (2-year recovery). "
        )
    if "never_claimed_jsa_pct" in a and "outcome_still_seeking_pct" not in a:
        note_parts.append(
            f"35% (720/2070) never claimed JSA at closure — split unknown "
            f"(early retired / immediately re-employed / ESA/UC / left UK). "
        )
    if not a:
        note_parts.append(
            "No direct outcome data for this quarter. "
            "Use RC LA claimant count trajectory as proxy for still_seeking rate. "
        )

    note_parts.append(
        "outcome_retrain_% and outcome_share_similar_% remain UNKNOWN — "
        "DWP microdata required for clean bucket split. "
        "outcome_underemployed_% proxy only (area wage data). "
        "All area-level data is NOT individual SSI worker tracking."
    )

    comparability = (
        "Nomis claimant count is AREA-LEVEL (Redcar & Cleveland LA or NE ITL1) — "
        "not individual SSI worker tracking. "
        "SSI dominated local economy so LA-level excess is largely attributable to closure, "
        "but non-SSI workers are included. "
        "JSA exit ≠ re-employed (workers may move to ESA/UC). "
        "Nomis age band 25-49 ≠ model band 25-44. "
        "UK training course = 2-day refresher to full vocational qual — "
        "not comparable to German Umschulung (1-2yr full requalification). "
        "Steel sector ~95% male; gender-disaggregated outcome rates unavailable from public sources."
    )

    # Add income band note to notes if present
    if income_band_note:
        note_parts.insert(0, f"INCOME BAND: {income_band}. {income_band_note} ")

    notes = " ".join(note_parts)

    return {
        # ---- Original evidence CSV columns ----
        "episode_id":               REDCAR_CONSTANTS["episode_id"],
        "source_citation":          source_citation,
        "source_url":               source_url,
        "year_window":              REDCAR_CONSTANTS["year_window"],
        "geography":                REDCAR_CONSTANTS["geography"],
        "region":                   REDCAR_CONSTANTS["region"],
        "age_band":                 age_band,
        "skill_tier":               skill_tier,
        "sector":                   REDCAR_CONSTANTS["sector"],
        "prior_income_band":        income_band,
        "subsidy_level":            SUBSIDY_BY_AGE.get(age_band, SUBSIDY_BY_AGE["all"]),
        "ubi_level":                UBI_LEVEL,
        "population_n":             ssi_n_cell if ssi_n_cell else SSI_N,
        "time_horizon":             f"{quarter} ({months_post} months post-closure)",
        "outcome_retrain_%":        None,
        "outcome_share_similar_%":  a.get("self_employment_pct"),   # proxy: self-employment
        "outcome_underemployed_%":  None,
        "outcome_exit_%":           exit_pct,
        "outcome_still_seeking_%":  still_seeking,
        "data_quality":             data_quality,
        "notes":                    notes,
        "comparability_notes":      comparability,
        # ---- Extra columns ----
        "quarter":                  quarter,
        "months_post_closure":      months_post,
        "gender":                   gender,
        # Redcar & Cleveland LA claimant data (closest geographic proxy)
        "rc_la_claimant_count_q_mean":    round(rc_la_q_mean)   if rc_la_q_mean   is not None else None,
        "rc_la_claimant_baseline":        round(rc_la_base)     if rc_la_base     is not None else None,
        "rc_la_excess_vs_baseline":       round(rc_la_excess,1) if rc_la_excess   is not None else None,
        "rc_la_age_claimants":            round(rc_age_claimants) if rc_age_claimants is not None else None,
        "rc_la_age_pct_of_la_total":      rc_age_pct,
        # North East ITL1 data (broader region)
        "ne_claimant_count_q_mean":       round(ne_cc_q_mean)  if ne_cc_q_mean   is not None else None,
        "ne_claimant_baseline":           round(ne_cc_base)    if ne_cc_base     is not None else None,
        "ne_claimant_excess_vs_baseline": round(ne_cc_excess,1)if ne_cc_excess   is not None else None,
        "ne_gender_claimants":            round(ne_gender_claimants) if ne_gender_claimants is not None else None,
        "ne_gender_pct_within_age_band":  ne_gender_pct_within_age,
        # Labour market context (APS)
        "ne_unemployment_rate_%":         ne_unemp_rate,
        "ne_employment_rate_%":           ne_emp_rate,
        "ne_activity_rate_%":             ne_activity_rate,
        "aps_period":                     aps_period,
        # Wage context (ASHE)
        "ne_median_weekly_pay_gbp":       ne_median_weekly_pay,
        "ne_median_annual_pay_gbp":       round(ne_median_weekly_pay*52) if ne_median_weekly_pay else None,
        # SSI workforce estimates for this cell
        "ssi_est_workers_this_cell":      ssi_n_cell,
        "ssi_age_share_prior":            AGE_SHARE.get(age_band),
        "ssi_gender_share_prior":         GENDER_SHARE.get(gender),
        # Composite outcome anchors (not in 4-bucket format but still useful)
        "off_all_benefits_%":             a.get("off_all_benefits_pct"),
        "still_on_benefits_%":            a.get("still_on_benefits_pct"),
        "never_claimed_jsa_%":            a.get("never_claimed_jsa_pct"),
        "self_employment_%":              a.get("self_employment_pct"),
        "self_employment_n":              a.get("self_employment_n"),
        "training_courses_n":             a.get("training_courses_total_n") or a.get("training_courses_delivered_n"),
        "area_wage_change_%":             a.get("area_wage_change_pct"),
        "ne_wage_change_%":               a.get("ne_wage_change_pct"),
        "manual_jobs_share_pre_%":        a.get("manual_jobs_share_pre_pct"),
        "manual_jobs_share_post_%":       a.get("manual_jobs_share_post_pct"),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
QUARTERS = ["2015Q4","2016Q1","2016Q2","2016Q3",
            "2016Q4","2017Q1","2017Q2","2017Q3","2017Q4",
            "2018Q1","2018Q2","2018Q3","2018Q4",
            "2019Q1","2019Q2","2019Q3","2019Q4"]
MONTHS_POST = {
    "2015Q4":1,"2016Q1":4,"2016Q2":7,"2016Q3":10,
    "2016Q4":13,"2017Q1":16,"2017Q2":19,"2017Q3":22,"2017Q4":25,
    "2018Q1":28,"2018Q2":31,"2018Q3":34,"2018Q4":37,
    "2019Q1":40,"2019Q2":43,"2019Q3":46,"2019Q4":49,
}
AGE_BANDS   = ["16-24","25-49","50+"]
AGE_NOMIS   = {"16-24":"Aged 16-24","25-49":"Aged 25-49","50+":"Aged 50+"}
GENDERS     = ["male","female"]
GENDER_NOMIS= {"male":"Male","female":"Female"}


def main():
    print("Loading ONS data...")
    cc_raw, ca_raw, ashe_raw, aps_raw = load_ons()

    ne_cc = cc_raw[cc_raw["region"]=="North East"]
    ne_ca = ca_raw[ca_raw["region"]=="North East"]
    ne_ashe = ashe_raw[ashe_raw["region"]=="North East"]
    ne_aps  = aps_raw[aps_raw["region"]=="North East"]

    print("Fetching / loading extra Nomis data...")
    ne_ga, rc_tot, rc_age_raw = fetch_extra()

    print("Aggregating to quarterly...")
    ne_cc_q  = to_quarterly(ne_cc,  [], "claimant_count")
    ne_cc_bl = baseline(ne_cc_q, "claimant_count")

    ne_ca_filt = ne_ca[ne_ca["age_band"].isin(AGE_NOMIS.values())]
    ne_ca_q    = to_quarterly(ne_ca_filt, ["age_band"], "claimants")

    ne_ga_filt = ne_ga[
        ne_ga["age_band"].isin(AGE_NOMIS.values()) &
        ne_ga["gender"].isin(GENDER_NOMIS.values())
    ]
    ne_ga_q = to_quarterly(ne_ga_filt, ["gender","age_band"], "claimants")

    rc_q    = to_quarterly(rc_tot, [], "claimant_count")
    rc_bl   = baseline(rc_q, "claimant_count")

    rc_age_filt = rc_age_raw[rc_age_raw["age_band"].isin(AGE_NOMIS.values())]
    rc_age_q    = to_quarterly(rc_age_filt, ["age_band"], "claimants")

    aps_lookup  = build_aps_lookup(ne_aps)
    ashe_lookup = build_ashe_lookup(ne_ashe)

    print(f"NE claimant baseline: {ne_cc_bl:.0f} | RC LA baseline: {rc_bl:.0f}")

    # ----------------------------------------------------------------
    # Generate rows
    # ----------------------------------------------------------------
    rows = []

    for q in QUARTERS:
        mp   = MONTHS_POST[q]
        anch = ANCHORS.get(q, {})
        aps  = aps_lookup.get(q, {})
        wage = ashe_lookup.get(q)

        # NE and RC quarterly totals
        ne_cc_row = ne_cc_q[ne_cc_q["quarter"]==q]
        ne_total  = ne_cc_row["claimant_count"].values[0] if len(ne_cc_row) else None
        ne_excess = (ne_total - ne_cc_bl) if ne_total is not None else None

        rc_row   = rc_q[rc_q["quarter"]==q]
        rc_total = rc_row["claimant_count"].values[0] if len(rc_row) else None
        rc_excess= (rc_total - rc_bl) if rc_total is not None else None

        src = anch.get("anchor_source","ONS Nomis NM_162_1 / DWP JSA Stats / SSI Task Force")
        url = anch.get("anchor_url","https://www.nomisweb.co.uk/")
        dq  = anch.get("data_quality","LOW")

        common = dict(
            ne_cc_q_mean=ne_total, ne_cc_excess=ne_excess, ne_cc_base=ne_cc_bl,
            rc_la_q_mean=rc_total, rc_la_excess=rc_excess, rc_la_base=rc_bl,
            ne_unemp_rate=aps.get("ne_unemp_rate"), ne_emp_rate=aps.get("ne_emp_rate"),
            ne_activity_rate=aps.get("ne_activity_rate"), aps_period=aps.get("aps_period"),
            ne_median_weekly_pay=wage,
            source_citation=src, source_url=url, data_quality=dq,
        )

        # --- Row A: all genders, all ages ---
        rows.append(make_row(
            quarter=q, months_post=mp, gender="all", age_band="all",
            anchors_q=anch, ssi_n_cell=SSI_N,
            rc_age_claimants=None, rc_age_pct=None,
            ne_gender_claimants=None, ne_gender_pct_within_age=None,
            **common,
        ))

        # --- Row B: all genders, per age band (using RC LA data) ---
        for ab in AGE_BANDS:
            ab_nom  = AGE_NOMIS[ab]
            ab_share= AGE_SHARE[ab]
            est_n   = round(SSI_N * ab_share)

            # RC LA age claimants
            rc_ab_row = rc_age_q[
                (rc_age_q["quarter"]==q) & (rc_age_q["age_band"]==ab_nom)
            ]
            rc_ab_n   = rc_ab_row["claimants"].values[0] if len(rc_ab_row) else None

            # RC LA total for this quarter (for pct)
            rc_ab_total = rc_age_q[rc_age_q["quarter"]==q]["claimants"].sum()
            rc_ab_pct   = round(100*rc_ab_n/rc_ab_total,1) if (rc_ab_n and rc_ab_total) else None

            rows.append(make_row(
                quarter=q, months_post=mp, gender="all", age_band=ab,
                anchors_q=anch, ssi_n_cell=est_n,
                rc_age_claimants=rc_ab_n, rc_age_pct=rc_ab_pct,
                ne_gender_claimants=None, ne_gender_pct_within_age=None,
                **common,
            ))

        # --- Row C: per gender x age band (NE ITL1 gender data) ---
        for ab in AGE_BANDS:
            ab_nom  = AGE_NOMIS[ab]
            ab_share= AGE_SHARE[ab]

            # RC LA age (same as above — no gender split at LA level from Nomis)
            rc_ab_row = rc_age_q[
                (rc_age_q["quarter"]==q) & (rc_age_q["age_band"]==ab_nom)
            ]
            rc_ab_n   = rc_ab_row["claimants"].values[0] if len(rc_ab_row) else None
            rc_ab_total = rc_age_q[rc_age_q["quarter"]==q]["claimants"].sum()
            rc_ab_pct   = round(100*rc_ab_n/rc_ab_total,1) if (rc_ab_n and rc_ab_total) else None

            # NE gender x age block
            ab_block = ne_ga_q[
                (ne_ga_q["quarter"]==q) & (ne_ga_q["age_band"]==ab_nom)
            ]
            ab_total_ne = ab_block["claimants"].sum()

            for g in GENDERS:
                g_nom  = GENDER_NOMIS[g]
                g_share= GENDER_SHARE[g]
                est_n  = round(SSI_N * ab_share * g_share)

                g_row = ab_block[ab_block["gender"]==g_nom]
                ne_g_n   = g_row["claimants"].values[0] if len(g_row) else None
                ne_g_pct = round(100*ne_g_n/ab_total_ne,1) if (ne_g_n and ab_total_ne) else None

                rows.append(make_row(
                    quarter=q, months_post=mp, gender=g, age_band=ab,
                    anchors_q=anch, ssi_n_cell=est_n,
                    rc_age_claimants=rc_ab_n, rc_age_pct=rc_ab_pct,
                    ne_gender_claimants=ne_g_n, ne_gender_pct_within_age=ne_g_pct,
                    **common,
                ))

        # --- Row D: per income band (all genders, all ages) ---
        # Aggregate mid outcome estimates from estimate_redcar_outcomes.py
        # applied at the quarter closest to each estimate time horizon.
        # Baseline (£25k-£35k) = aggregate; other bands adjusted per INCOME_BANDS.
        BASELINE_OUTCOMES = {
            "2016Q1": dict(retrain=2,  share_similar=10, underemployed=43, exit=20, still_seeking=25),
            "2016Q4": dict(retrain=6,  share_similar=13, underemployed=52, exit=26, still_seeking=3),
            "2017Q4": dict(retrain=8,  share_similar=15, underemployed=45, exit=30, still_seeking=2),
        }
        # For quarters without a direct estimate, use the nearest
        NEAREST = {
            "2015Q4": "2016Q1", "2016Q1": "2016Q1", "2016Q2": "2016Q1",
            "2016Q3": "2016Q4", "2016Q4": "2016Q4", "2017Q1": "2016Q4",
            "2017Q2": "2017Q4", "2017Q3": "2017Q4", "2017Q4": "2017Q4",
            "2018Q1": "2017Q4", "2018Q2": "2017Q4", "2018Q3": "2017Q4", "2018Q4": "2017Q4",
            "2019Q1": "2017Q4", "2019Q2": "2017Q4", "2019Q3": "2017Q4", "2019Q4": "2017Q4",
        }
        base = BASELINE_OUTCOMES[NEAREST[q]]

        for band_label, band in INCOME_BANDS.items():
            adj     = band["outcome_adj"]
            band_n  = round(SSI_N * band["share"])

            # Apply adjustments — clamp to [0, 100] and renormalise to 100
            raw = {
                "retrain":       base["retrain"]       + adj["retrain"],
                "share_similar": base["share_similar"] + adj["share_similar"],
                "underemployed": base["underemployed"] + adj["underemployed"],
                "exit":          base["exit"]          + adj["exit"],
                "still_seeking": base["still_seeking"] + adj["still_seeking"],
            }
            raw = {k: max(0, v) for k, v in raw.items()}
            total = sum(raw.values())
            normed = {k: round(v / total * 100, 1) for k, v in raw.items()}

            # For Q1 2016 still_seeking — keep the hard anchor for baseline band
            still_seeking_val = (
                anch.get("outcome_still_seeking_pct")
                if band_label == "£25k-£35k" else normed["still_seeking"]
            )

            income_note = (
                f"{band['description']} "
                f"Outcome adjustments vs £25k-£35k baseline: "
                f"retrain {adj['retrain']:+d}pp, share_similar {adj['share_similar']:+d}pp, "
                f"underemployed {adj['underemployed']:+d}pp, exit {adj['exit']:+d}pp, "
                f"still_seeking {adj['still_seeking']:+d}pp. "
                f"Source: China Shock ADH (low-wage→exit/underemployed; high-wage→share_similar) "
                f"+ Hartz re-employment wage analysis. All income-band outcomes ESTIMATED_LOW."
            )

            row = make_row(
                quarter=q, months_post=mp, gender="all", age_band="all",
                anchors_q=anch, ssi_n_cell=band_n,
                rc_age_claimants=None, rc_age_pct=None,
                ne_gender_claimants=None, ne_gender_pct_within_age=None,
                income_band=band_label,
                skill_tier=band["skill_tier"],
                income_band_note=income_note,
                **common,
            )
            # Overwrite outcome columns with income-adjusted estimates
            row["outcome_retrain_%"]       = normed["retrain"]
            row["outcome_share_similar_%"] = normed["share_similar"]
            row["outcome_underemployed_%"] = normed["underemployed"]
            row["outcome_exit_%"]          = normed["exit"]
            row["outcome_still_seeking_%"] = still_seeking_val
            row["data_quality"]            = "ESTIMATED_LOW" if band_label != "£25k-£35k" else row["data_quality"]
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel.to_csv(OUT, index=False)

    print(f"\nSaved: {OUT}")
    print(f"  {len(panel)} rows x {len(panel.columns)} columns")
    print(f"  Columns: {panel.columns.tolist()}")
    print()

    # Summary table
    summary = panel[panel["gender"]=="all"][panel["age_band"]=="all"][[
        "quarter","months_post_closure",
        "rc_la_claimant_count_q_mean","rc_la_excess_vs_baseline",
        "ne_unemployment_rate_%","ne_median_weekly_pay_gbp",
        "outcome_still_seeking_%","off_all_benefits_%","outcome_exit_%",
        "self_employment_%","area_wage_change_%",
        "data_quality",
    ]]
    print("SUMMARY (all genders, all ages):")
    print(summary.to_string(index=False))
    print()
    print("DATA GAPS (outcome_% columns that remain None):")
    print("  outcome_retrain_%       — UNKNOWN. DWP microdata required.")
    print("  outcome_share_similar_% — proxied only by self_employment_% at Q4 2016 and Q4 2017.")
    print("  outcome_underemployed_% — UNKNOWN. area_wage_change_% is a proxy, not a rate.")
    print("  outcome_exit_%          — PARTIAL. still_on_benefits_% at Q4 2016 is a lower bound only.")
    print("  Recommendation: constrain retrain+share_similar+underemployed+exit to sum to")
    print("  off_all_benefits_% (93%) at Q4 2016, using Hartz/China shock priors for the split.")


if __name__ == "__main__":
    main()
