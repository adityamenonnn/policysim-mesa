# Bibliography: Redcar SSI 2015 panel dataset

## Primary data sources

### DWP administrative statistics

Department for Work and Pensions (2016a). *Sahaviriya Steel Industries (SSI) Redcar Closure: Jobseeker's Allowance Statistics*. Ad-hoc statistical release, March 2016. Available at: https://assets.publishing.service.gov.uk/media/5a757a2ae5274a1622e22197/sahaviriya-steel-ind-ad-hoc-jsa-stats.pdf

> Used for: 25% still-seeking anchor (540/2,070 SSI employees on JSA at end of February 2016). Data derived from DWP bespoke tracking database populated from Insolvency Service employee records and JSA/ESA administrative systems.

Department for Work and Pensions (2016b). *Sahaviriya Steel Industries (SSI) Redcar Closure: Updated Statistics*. Ad-hoc statistical release, September 2016. Available at: https://assets.publishing.service.gov.uk/media/5a80843be5274a2e87dba425/ssi-redcar-ad-hoc-statistics-updated-september-2016.pdf

> Used for: 93% off-benefits anchor (1,990/2,150 SSI and supply-chain claimants ended benefit claim by 31 August 2016). Reference date is 31 August 2016 (2016Q3); placed at 2016Q4 in panel for model alignment.

### SSI Task Force

SSI Task Force (2016). *One Year On Report*. September 2016. Tees Valley Combined Authority. Available at: https://teesvalley-ca.gov.uk/news/ssi-task-force-publishes-one-year-report/

> Used for: self-employment count (172 new businesses), training courses (15,510 approved), general contextual figures on post-closure support.

### Nomis (ONS/DWP administrative data via API)

All Nomis series accessed via the Nomis REST API (https://www.nomisweb.co.uk/api/v01/). Fetched by scripts in this directory. Cache stored in `_nomis_cache/`.

| Series | Description | Used for | Script |
|---|---|---|---|
| NM_162_1 | Claimant Count (Alternative Claimant Count) | RC LA and NE monthly claimant counts by age/gender | build_redcar_panel.py, fetch_redcar_extra.py |
| NM_17_5 | Annual Population Survey | NE unemployment, employment, activity rates (all ages and by age band) | build_redcar_panel.py |
| NM_30_1 | Annual Survey of Hours and Earnings (aggregate) | NE median and percentile weekly pay | build_redcar_panel.py, parse_ashe_age.py |
| NM_189_1 | Business Register and Employment Survey (BRES) | RC LA sector employment (manufacturing, mining/utilities, total) | build_redcar_panel.py |
| NM_4_1 | Jobseeker's Allowance by Age and Duration | NE JSA stock by duration band (2015Q4-2019Q4) | fetch_redcar_extra.py |

**Nomis citation:** Office for National Statistics. *Nomis: Official Labour Market Statistics*. Available at: https://www.nomisweb.co.uk

**Note on NM_4_1 geography:** JSA duration stock is suppressed at local authority level by DWP. North East regional level (E12000001) is available and used here as a regional benchmark.

**Note on UC migration:** Alternative Claimant Count (NM_162_1) includes Universal Credit claimants from the point of UC rollout. Redcar and Cleveland UC rollout began mid-2016. Claimant counts from 2017 onward include UC claimants not previously captured in JSA figures, creating a partial discontinuity with pre-UC counts. All 2017+ rows in the panel carry a UC migration caveat in the `notes` column.

### ONS Annual Survey of Hours and Earnings — Table 6 (age group)

Office for National Statistics. *Annual Survey of Hours and Earnings: Age Group, Table 6 — Gross Weekly Pay*. Annual revised releases, 2016-2019. Available at: https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/agegroupashetable6

> Used for: age-specific wage distribution estimates (median, p10, p25, p75, p90) for 16-24, 25-49, and 50+ bands. ONS does not publish ASHE by age band at regional level. National age-band ratios (band median / all-workers median) are applied to the NE aggregate median (NM_30_1) to estimate NE age-specific wages. Columns prefixed `ashe_age_*` carry data quality flag `HIGH_SCALED`. Fetched and cached by `parse_ashe_age.py` in `_ashe_cache/`.

### ONS BEAO timeseries

Office for National Statistics. *Redundancies: BEAO — Redundancies (Thousands), Seasonally Adjusted*. Labour Market Statistics timeseries. Available at: https://www.ons.gov.uk/employmentandlabourmarket/peoplenotinwork/redundancies/timeseries/beao/lms

> Used for: national monthly redundancy notifications (GB, 000s) as macroeconomic context for the 2015-2016 shock period. Fetched by `fetch_redcar_extra.py`.

### DWP Stat-Xplore — manual downloads

The following files were downloaded manually from DWP Stat-Xplore (https://stat-xplore.dwp.gov.uk) and are stored in `_source_data/`. They cannot be fetched programmatically as Stat-Xplore requires a login.

| File | Dataset | Geography | Period |
|---|---|---|---|
| acc4_age_gender_redcar_cleveland.xlsx | ACC4 — Alternative Claimant Count by Age and Gender | Redcar and Cleveland LA | Oct 2015 - Dec 2019 |
| esa_redcar_to_feb2018.xlsx | ESA Caseload — Data to February 2018 | Redcar and Cleveland LA | Nov 2008 - Feb 2018 |
| esa_redcar_from_may2018.xlsx | ESA Caseload — Data from May 2018 | Redcar and Cleveland LA | May 2018 - Nov 2025 |

**DWP Stat-Xplore citation:** Department for Work and Pensions. *Stat-Xplore*. Available at: https://stat-xplore.dwp.gov.uk

### Nomis series investigated but not used

The following Nomis datasets were queried during data collection but found to be unavailable for the 2015-2019 period at relevant geographies. They are referenced in script code comments for transparency.

| Series | Description | Why not used |
|---|---|---|
| NM_5_1 | Annual Business Inquiry (misidentified as vacancy survey) | Not a vacancy dataset; series ends before 2012 |
| NM_19_1 to NM_24_1 | Vacancy survey (various) | All series end before 2012; retired when UC rolled out |
| NM_89_1 | Vacancy survey (alternative) | Ends before 2012 |
| NM_2_1 | JSA duration stock (old series) | Ends October 1998 |

For 2015-2019 vacancy data at sub-national geography, the ONS LFS regional tables or DWP Stat-Xplore are required. These were not included in the panel due to manual download requirements and time constraints.

---

## How to reproduce the panel

Run scripts in this order:

```bash
cd /Users/adityamenon/Documents/PolicySim/policysim-mesa

# 1. Build base panel (ACC, APS, ASHE aggregate, BRES, Tees Valley)
python3 research/redcar/build_redcar_panel.py

# 2. Add real RC LA ACC by age and gender (requires Stat-Xplore download)
python3 research/redcar/integrate_statxplore.py

# 3. Add NE ACC by age, BEAO redundancy notifications, NE JSA duration (all live Nomis/ONS)
python3 research/redcar/fetch_redcar_extra.py

# 4. Add ASHE age-specific wage estimates (fetches ONS Table 6 ZIPs)
python3 research/redcar/parse_ashe_age.py

# 5. Add ESA caseload (requires two Stat-Xplore downloads)
python3 research/redcar/integrate_esa.py \
  --pre research/redcar/_source_data/esa_redcar_to_feb2018.xlsx \
  --post research/redcar/_source_data/esa_redcar_from_may2018.xlsx

# 6. Run statistical estimation (Methods 1, 2, 2b, 3)
python3 research/redcar/estimate_statistical.py
```

Steps 3 and 4 use local caches (`_nomis_cache/`, `_ashe_cache/`) so subsequent runs do not re-fetch from the API. Delete the cache directories to force a fresh pull.
