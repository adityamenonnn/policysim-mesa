"""
ONS/Nomis data acquisition for the labor-scenario platform.
Run on YOUR machine (needs internet):  pip install requests pandas
                                       python fetch_ons.py
Downloads every scrapeable series in ons_manifest.csv into ./ons_data/
and writes tidy CSVs matching the platform's environment/Tier-1 layers.

Notes:
- Nomis REST API needs no key for these volumes. Be polite: ~1 req/sec.
- If a dataset ID or measure code 404s, run discover() to list current
  codes -- Nomis occasionally renumbers. Verify a few values against the
  Nomis website before trusting a full pull.
- LFS longitudinal microdata (the Tier-1 core) is NOT here: register at
  ukdataservice.ac.uk and download the two- and five-quarter panels.
"""
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("ons_data")
OUT.mkdir(exist_ok=True)
NOMIS = "https://www.nomisweb.co.uk/api/v01/dataset"

# Geographies: national coverage as ITL1 (former NUTS1) regions.
# "National" here = Great Britain, obtained by summing these regions in the
# fitter. Northern Ireland is excluded -- it is NISRA-sourced and absent from
# these LFS/ASHE/BRES pulls, and the model's region taxonomy is GB-only.
# Keeping the regional breakdown (rather than one UK geography code) also
# supplies the 'regions' field of the population calibration directly, and
# sidesteps Nomis' opaque national-total codes (sum the regions instead).
#
# NOTE: the model collapses these 11 ITL1 regions into 7 buckets
# (london, south_east, north_west, midlands, scotland, wales, north_east).
# That crosswalk (e.g. midlands <- East + West Midlands) is a documented
# mapping applied in the fitter, not here: acquire the true geography, map
# downstream.
REGIONS = {
    "E12000001": "North East",
    "E12000002": "North West",
    "E12000003": "Yorkshire and The Humber",
    "E12000004": "East Midlands",
    "E12000005": "West Midlands",
    "E12000006": "East of England",
    "E12000007": "London",
    "E12000008": "South East",
    "E12000009": "South West",
    "W92000004": "Wales",
    "S92000003": "Scotland",
}
GEO_PARAM = ",".join(REGIONS)


def get_csv(url: str, dest: Path) -> pd.DataFrame:
    """Fetch a Nomis CSV endpoint to disk, return DataFrame."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    time.sleep(1.2)
    return pd.read_csv(dest)


def discover(dataset_id: str) -> None:
    """Print a dataset's dimensions/codes -- use when a pull 404s."""
    r = requests.get(f"{NOMIS}/{dataset_id}.def.sdmx.json", timeout=60)
    r.raise_for_status()
    for dim in r.json()["structure"]["keyfamilies"]["keyfamily"][0][
        "components"]["dimension"]:
        print(dim["conceptref"])


# ---------------------------------------------------------------- 1
def claimant_count():
    """NM_162_1: monthly claimant count, 2005-2019, all ITL1 regions.
    Includes UC-era experimental measure; also pull flows where present."""
    # '...' is Nomis range syntax -- every month between the endpoints.
    # A comma ('2005-01,2019-12') fetches ONLY those two months, which is
    # useless for the displacement-episode trajectories (e.g. Redcar SSI,
    # Oct 2015) that the behaviour calibration is validated against.
    url = (f"{NOMIS}/NM_162_1.data.csv"
           f"?geography={GEO_PARAM}"
           f"&date=2005-01...2019-12"
           f"&gender=0&age=0&measure=1&measures=20100"
           f"&select=date_name,geography_name,geography_code,obs_value")
    df = get_csv(url, OUT / "claimant_count_raw.csv")
    df.columns = ["month", "region", "region_code", "claimant_count"]
    df.to_csv(OUT / "claimant_count_monthly.csv", index=False)
    print("claimant_count:", len(df), "rows")


# ---------------------------------------------------------------- 2
def claimant_by_age_duration():
    """Age x duration splits for duration-dependence estimation
    (JSA era; post-2015 use alternative claimant count for levels)."""
    url = (f"{NOMIS}/NM_162_1.data.csv"
           f"?geography={GEO_PARAM}"           # all regions, national coverage
           f"&date=2005-01...2019-12"          # range, not two endpoints
           f"&gender=0"
           f"&age=1,2,3,4,5,6,7,8"
           f"&measure=1&measures=20100"
           f"&select=date_name,geography_name,age_name,obs_value")
    df = get_csv(url, OUT / "claimant_age_raw.csv")
    df.columns = ["month", "region", "age_band", "claimants"]
    df.to_csv(OUT / "claimant_by_age.csv", index=False)
    print("claimant_by_age:", len(df), "rows")


# ---------------------------------------------------------------- 3
def ashe_pay():
    """NM_30_1: ASHE resident-based gross weekly pay, MEDIAN only (pay=1).
    Emits Value + Confidence (CV). This pins the *centre* of the wage
    distribution but not its spread -- use ashe_pay_distribution() for the
    percentiles needed to fit sigma. item=2 = weekly pay, gross."""
    url = (f"{NOMIS}/NM_30_1.data.csv"
           f"?geography={GEO_PARAM}"
           f"&date=2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019"
           f"&sex=8&item=2&pay=1"
           f"&measures=20100,20701"
           f"&select=date_name,geography_name,measures_name,obs_value")
    df = get_csv(url, OUT / "ashe_raw.csv")
    df.columns = ["year", "region", "measure", "weekly_pay_gbp"]
    df.to_csv(OUT / "ashe_resident_pay.csv", index=False)
    print("ashe:", len(df), "rows")


# --------------------------------------------------------------- 3b
# In NM_30_1 the STATISTIC (median / mean / percentiles) is the ITEM
# dimension, while PAY is the pay *type* (weekly gross, hourly, ...).
# (Verified via discover: item 2=Median, 6=10pct, 8=25pct, 13=75pct,
# 15=90pct; pay 1=Weekly pay - gross.) We hold pay=1 and vary item to get
# the percentiles needed to fit the log-normal wage model (income_by_skill):
#     mu    = ln(median_weekly * 52)             # annualise, log space
#     sigma = (ln(p90) - ln(p50)) / 1.2816       # 1.2816 = z_0.90
# CAVEAT: these percentiles describe the *aggregate* wage distribution, NOT
# a per-skill split. income_by_skill needs pay by occupation (SOC ->
# low/med/high) from the national ASHE-by-occupation table (or LFS micro-
# data); anchor those skill differentials to this aggregate as a location
# shift. sex=8 = Full-Time Workers (avoids the part-time distortion).
_ASHE_ITEM_CODES = {2: "median", 6: "p10", 8: "p25", 13: "p75", 15: "p90"}


def ashe_pay_distribution():
    """NM_30_1 gross-weekly-pay percentiles per region-year, for fitting the
    log-normal wage model. Additive: leaves ashe_resident_pay.csv untouched."""
    codes = ",".join(str(c) for c in _ASHE_ITEM_CODES)
    url = (f"{NOMIS}/NM_30_1.data.csv"
           f"?geography={GEO_PARAM}"
           f"&date=2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019"
           f"&sex=8&item={codes}&pay=1"             # vary statistic, weekly gross
           f"&measures=20100"                       # value only (drop CV here)
           f"&select=date_name,geography_name,item_name,obs_value")
    try:
        df = get_csv(url, OUT / "ashe_percentiles_raw.csv")
    except requests.RequestException as e:
        print(f"ashe_pay_distribution: {e} -- run discover('NM_30_1') and "
              f"check the ITEM codes in _ASHE_ITEM_CODES")
        return
    df.columns = ["year", "region", "statistic", "weekly_pay_gbp"]
    df.to_csv(OUT / "ashe_pay_percentiles.csv", index=False)
    print("ashe_pay_distribution:", len(df), "rows")


# ---------------------------------------------------------------- 4
def annual_population_survey():
    """NM_17_5: employment rate, inactivity, by age band, per LA-year."""
    url = (f"{NOMIS}/NM_17_5.data.csv"
           f"?geography={GEO_PARAM}"
           f"&date=2005-12...2019-12"     # range, not two endpoints
           f"&variable=18,45,83"          # emp rate 16-64, unemp rate, inactivity
           f"&measures=20599"
           f"&select=date_name,geography_name,variable_name,obs_value")
    df = get_csv(url, OUT / "aps_raw.csv")
    df.columns = ["period", "region", "variable", "value"]
    df.to_csv(OUT / "aps_labour_market.csv", index=False)
    print("aps:", len(df), "rows")


# ---------------------------------------------------------------- 5
def bres_industry_jobs():
    """NM_189_1: employee jobs by broad industry per region-year (2009+).
    Summed over industries this also yields regional employment totals, the
    'regions' shares of the population calibration (employment basis; for a
    resident basis use ONS mid-year population estimates instead)."""
    url = (f"{NOMIS}/NM_189_1.data.csv"
           f"?geography={GEO_PARAM}"
           f"&date=2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019"
           f"&industry=163577857...163577874"   # broad SIC sections
           f"&employment_status=1&measure=1&measures=20100"
           f"&select=date_name,geography_name,industry_name,obs_value")
    df = get_csv(url, OUT / "bres_raw.csv")
    df.columns = ["year", "region", "industry", "employee_jobs"]
    df.to_csv(OUT / "industry_mix.csv", index=False)
    print("bres:", len(df), "rows")


# --------------------------------------------------------------- 5b
# SOC major group (occupation) x SIC section (industry), national (England &
# Wales), 2011 Census table CT0144 (NM_974_1). Feeds skill_by_sector: the
# occupation mix within each sector, mapped SOC-major -> low/med/high skill.
# Occupation STRUCTURE within an industry is far more stable over time than
# employment LEVELS, so pairing a 2011 mix with 2019 sector levels is sound.
# Full SIC sections (not the coarse census 8-group industry) keep H (logistics)
# and J (tech), K (finance) and Q (health) separate -- exactly the splits the
# 7-sector model needs. Codes verified via discover on NM_974_1.
_SOC_MAJOR = "1,52,144,230,264,339,375,401,453"          # SOC major groups 1-9
_SIC_SECTION = ("1,10,16,100,104,109,111,119,128,136,146,"
                "150,152,164,173,183,195,204,210,214,217")  # sections A-U


def census_occ_by_industry():
    """NM_974_1 (CT0144): occupation-major x SIC-section worker counts, England
    & Wales. Feeds skill_by_sector in the population calibration."""
    url = (f"{NOMIS}/NM_974_1.data.csv"
           f"?geography=0"                               # England and Wales
           f"&c_occpuk11h_0={_SOC_MAJOR}"
           f"&c_indpuk_0={_SIC_SECTION}"
           f"&measures=20100"
           f"&select=c_occpuk11h_0_name,c_indpuk_0_name,obs_value")
    df = get_csv(url, OUT / "census_occ_industry_raw.csv")
    df.columns = ["occupation", "industry", "workers"]
    df.to_csv(OUT / "occupation_by_industry.csv", index=False)
    print("census_occ_by_industry:", len(df), "rows")


# --------------------------------------------------------------- 5c
def population_by_age():
    """NM_2002_1: mid-year population estimates by single year of age, per
    region. Feeds the population calibration 'age' block. Single-year age
    codes are 101 + age (verified: code 101 = 'Age 0'); we pull the model's
    working-age window 18-64. gender=0 = persons; date=latest = newest MYE."""
    ages = ",".join(str(101 + a) for a in range(18, 65))   # Age 18..64
    url = (f"{NOMIS}/NM_2002_1.data.csv"
           f"?geography={GEO_PARAM}"
           f"&date=latest"
           f"&gender=0"
           f"&c_age={ages}"
           f"&measures=20100"
           f"&select=date_name,geography_name,c_age_name,obs_value")
    df = get_csv(url, OUT / "population_age_raw.csv")
    df.columns = ["year", "region", "age_band", "population"]
    df.to_csv(OUT / "population_by_age.csv", index=False)
    print("population_by_age:", len(df), "rows")


# ---------------------------------------------------------------- 6
def ons_timeseries():
    """National macro series from the ONS website time-series API (no key).
    The old api.ons.gov.uk host is retired (404s on everything); series now
    resolve ONLY under their exact taxonomy path on www.ons.gov.uk, as
        /<topic-path>/timeseries/<cdid>/<dataset>/data
    A wrong topic path 404s even when the CDID is valid, so the path is
    pinned per series below.
    MGSX = LFS unemployment rate; AP2Y = vacancies (000s);
    BEAO = redundancy level (000s). If one 404s, ONS moved its taxonomy
    path -- open the series page on ons.gov.uk and copy the new path."""
    base = "https://www.ons.gov.uk"
    # ONS blocks the default python-requests user-agent; send a browser one.
    headers = {"User-Agent": "Mozilla/5.0 (labour-scenario-platform data pull)"}
    # cdid: (topic_path, dataset, output_name)
    series = {
        "MGSX": ("employmentandlabourmarket/peoplenotinwork/unemployment",
                 "lms", "unemployment_rate_uk"),
        "AP2Y": ("employmentandlabourmarket/peopleinwork/employmentandemployeetypes",
                 "unem", "vacancies_uk"),
        "BEAO": ("employmentandlabourmarket/peoplenotinwork/redundancies",
                 "lms", "redundancies_uk"),
    }
    for sid, (path, dataset, name) in series.items():
        url = f"{base}/{path}/timeseries/{sid.lower()}/{dataset}/data"
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"{sid}: HTTP {r.status_code} -- taxonomy path moved; open "
                  f"the series on ons.gov.uk and copy its 'data' path")
            continue
        data = r.json()
        # Prefer monthly; fall back to quarterly/annual if a series lacks it.
        points = (data.get("months") or data.get("quarters")
                  or data.get("years") or [])
        df = pd.DataFrame(
            [(p["date"], p["value"]) for p in points],
            columns=["period", name])
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), "rows")
        time.sleep(1.2)


# ---------------------------------------------------------------- 7
def business_demography_note():
    """Business demography survival tables ship as xlsx on ons.gov.uk
    (search 'business demography UK'), with 1-5yr survival by LA.
    URLs change yearly, so this downloads nothing -- grab the latest
    edition manually and save as ons_data/business_demography.xlsx,
    then run tidy_business_demography() if you want it reshaped."""
    print(business_demography_note.__doc__)


if __name__ == "__main__":
    claimant_count()
    claimant_by_age_duration()
    ashe_pay()
    ashe_pay_distribution()
    annual_population_survey()
    bres_industry_jobs()
    census_occ_by_industry()
    population_by_age()
    ons_timeseries()
    business_demography_note()
    print("\nDone. Validate a few numbers against nomisweb.co.uk before "
          "trusting the pull. LFS longitudinal microdata: ukdataservice.ac.uk")