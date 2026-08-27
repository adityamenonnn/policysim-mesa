# Panel column reference: math_testing_panel.csv

Redcar SSI closure episode (Oct 2015). 255 rows x 150 columns.
17 quarters: 2015Q4 to 2019Q4. 15 row types per quarter (age band x gender combinations).

Data quality flags used throughout:
- HIGH: DWP administrative data or ONS official survey — treat as ground truth
- HIGH_SCALED: Derived from HIGH-quality source via a documented scaling step
- ESTIMATED_MED: Modelled estimate with meaningful empirical basis
- ESTIMATED_LOW: Weakly evidenced estimate, treat as indicative only
- UNAVAILABLE: Series not available for this geography/period

---

## Identifiers and metadata

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| episode_id | Always `redcar_2015` | — | — |
| source_citation | Primary citation for the row's outcome anchors | — | DWP ad-hoc statistical releases |
| source_url | URL of primary source document | — | gov.uk / Tees Valley CA |
| year_window | Year range of the episode panel | — | — |
| geography | Human-readable geography label | — | — |
| region | ONS region label | — | — |
| quarter | Quarter in YYYYQN format (e.g. 2016Q1) | — | — |
| months_post_closure | Months since 2 Oct 2015 (closure date) | — | Calculated |
| data_quality | Row-level data quality flag | — | — |
| notes | Free-text notes: labour market context, data caveats, anchor sources | — | Various |
| comparability_notes | Notes on cross-episode comparability | — | — |

---

## Row-type dimensions

Each quarter has 15 rows covering all combinations of the dimensions below.

| Column | Values | Description |
|---|---|---|
| age_band | all, 16-24, 25-49, 50+ | Model age band for the row |
| gender | all, male, female | Gender for the row |
| skill_tier | skilled_manual, low_skilled, professional | Skill segment |
| sector | primary_metals / steel_manufacturing | Sector (constant) |
| prior_income_band | <£20k, £20k-£25k, £25k-£35k, £35k-£50k, £25k-£35k est. | Pre-displacement income segment |
| subsidy_level | Text description of government support available | Policy context |
| ubi_level | N/A — no UBI in UK Oct 2015. Model counterfactual field. | Policy context |
| population_n | Estimated workers in this row's cell | ESTIMATED_LOW |
| time_horizon | Panel time horizon label | — |

---

## Outcome variables (row-level, cross-episode or observed)

These are the outcome percentages from the cross-episode dataset or (where available) directly observed for the SSI closure. They are the dependent variables the model tries to reproduce.

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| outcome_retrain_% | % who retrained for a new sector | ESTIMATED_LOW — cross-episode prior | Cross-episode literature |
| outcome_share_similar_% | % who found similar-sector work | ESTIMATED_LOW — cross-episode prior | Cross-episode literature |
| outcome_underemployed_% | % who took lower-paid work | ESTIMATED_MED — area wage proxy | NE ASHE / Nomis |
| outcome_exit_% | % who left the labour market | ESTIMATED_LOW — cross-episode prior | Cross-episode literature |
| outcome_still_seeking_% | % still seeking work at this quarter | REAL (2016Q1 anchor) / ESTIMATED otherwise | DWP March 2016 (Q1 anchor) |

**Anchors:**
- `outcome_still_seeking_%` at 2016Q1 = **25.0%** (540/2,070 SSI direct employees still on JSA at end of Feb 2016). Source: DWP SSI JSA Statistics March 2016. URL: https://assets.publishing.service.gov.uk/media/5a757a2ae5274a1622e22197/sahaviriya-steel-ind-ad-hoc-jsa-stats.pdf
- `off_all_benefits_%` at 2016Q4 = **93%** (1,990/2,150 SSI+supply-chain claimants had ended benefit claims). Note: DWP reference date is 31 Aug 2016 (2016Q3). Placed at Q4 for model alignment. Source: DWP SSI Updated Statistics September 2016. URL: https://assets.publishing.service.gov.uk/media/5a80843be5274a2e87dba425/ssi-redcar-ad-hoc-statistics-updated-september-2016.pdf
- "Off benefits" does not mean employed. DWP explicitly lists other outcomes: ESA, State Pension, UC, retiring, leaving UK, being a dependent partner.

---

## SSI workforce priors

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| ssi_est_workers_this_cell | Estimated SSI workers in this age/gender cell | ESTIMATED_LOW | Scaled from 3,500 total workforce |
| ssi_age_share_prior | Prior age-band share of SSI workforce | ESTIMATED_LOW (updated by ACC data) | Industry norms; now superseded by acc_excess_*_pct |
| ssi_gender_share_prior | Prior gender share of SSI workforce | ESTIMATED_LOW (updated by ACC data) | Steel-sector norms; now superseded by acc_excess_male_pct |

---

## Redcar LA (RC LA) claimant count — from Nomis NM_162_1

RC LA = Redcar and Cleveland Local Authority (E06000003).
These are monthly Alternative Claimant Count (ACC) figures from DWP administrative records, aggregated to quarterly means.

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| rc_la_claimant_count_q_mean | Mean monthly ACC claimants in RC LA | HIGH | Nomis NM_162_1 |
| rc_la_claimant_baseline | 2019 annual mean claimant count (post-shock stable level) | HIGH | Nomis NM_162_1 |
| rc_la_excess_vs_baseline | Excess over baseline (shock-attributable claimants, clamped >= 0) | HIGH | Derived |
| rc_la_age_claimants | ACC claimants in RC LA for this row's age band | HIGH | Nomis NM_162_1 |
| rc_la_age_pct_of_la_total | This age band as % of RC LA total ACC | HIGH | Derived |

---

## North East regional claimant count — from Nomis NM_162_1

NE = North East England (E12000001 ITL1 region).

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| ne_claimant_count_q_mean | Mean monthly ACC claimants in North East (all ages) | HIGH | Nomis NM_162_1 |
| ne_claimant_baseline | 2019 annual mean NE claimant count | HIGH | Nomis NM_162_1 |
| ne_claimant_excess_vs_baseline | Excess over 2019 baseline (NE region) | HIGH | Derived |
| ne_gender_claimants | ACC claimants in NE for this row's gender | HIGH | Nomis NM_162_1 |
| ne_gender_pct_within_age_band | Gender as % within age band for NE | HIGH | Derived |

---

## North East APS labour market rates — from Nomis NM_17_1

APS = Annual Population Survey. Rolling 12-month periods, mapped to the nearest quarter.

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| ne_unemployment_rate_% | NE unemployment rate (all ages, all genders) | HIGH | Nomis APS NM_17_1 |
| ne_employment_rate_% | NE employment rate | HIGH | Nomis APS NM_17_1 |
| ne_activity_rate_% | NE economic activity rate | HIGH | Nomis APS NM_17_1 |
| aps_period | Rolling 12-month period this APS estimate covers (e.g. "Apr 2015-Mar 2016") | — | — |
| ne_age_unemployment_rate_% | NE unemployment rate for this row's age band | HIGH | Nomis APS NM_17_1 |
| ne_age_employment_rate_% | NE employment rate for this row's age band | HIGH | Nomis APS NM_17_1 |
| ne_age_activity_rate_% | NE economic activity rate for this row's age band | HIGH | Nomis APS NM_17_1 |
| ne_age_aps_period | APS rolling period for age-specific rates | — | — |

---

## North East wage distribution — from Nomis ASHE NM_30_1

ASHE = Annual Survey of Hours and Earnings. Annual data; same value repeated for all quarters in a year.
These are NE aggregate wages across all age groups.

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| ne_median_weekly_pay_gbp | NE median gross weekly pay (all workers, full-time) | HIGH | Nomis ASHE NM_30_1 |
| ne_median_annual_pay_gbp | NE median gross annual pay (derived: weekly x 52) | HIGH | Derived |
| ne_wage_p10_gbp | NE 10th percentile gross weekly pay | HIGH | Nomis ASHE NM_30_1 |
| ne_wage_p25_gbp | NE 25th percentile gross weekly pay | HIGH | Nomis ASHE NM_30_1 |
| ne_wage_p75_gbp | NE 75th percentile gross weekly pay | HIGH | Nomis ASHE NM_30_1 |
| ne_wage_p90_gbp | NE 90th percentile gross weekly pay | HIGH | Nomis ASHE NM_30_1 |

---

## Tees Valley CA claimant count — from Nomis NM_162_1

Tees Valley CA = Combined Authority (E47000006): Redcar, Middlesbrough, Stockton, Hartlepool, Darlington. This is the SSI Task Force geography.

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| tv_ca_claimant_count_q_mean | Mean monthly ACC claimants in Tees Valley CA | HIGH | Nomis NM_162_1 |
| tv_ca_claimant_baseline | 2019 annual mean TV CA claimant count | HIGH | Nomis NM_162_1 |
| tv_ca_excess_vs_baseline | Excess over 2019 baseline (TV CA) | HIGH | Derived |

---

## RC LA sector employment — from Nomis BRES NM_189_1

BRES = Business Register and Employment Survey. Annual data; same value repeated for all quarters in a year.

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| rc_la_manufacturing_jobs | Total manufacturing jobs in RC LA | HIGH | Nomis BRES NM_189_1 |
| rc_la_mining_util_jobs | Mining and utilities jobs in RC LA | HIGH | Nomis BRES NM_189_1 |
| rc_la_total_jobs | Total jobs in RC LA (all sectors) | HIGH | Nomis BRES NM_189_1 |
| rc_la_manufacturing_pct | Manufacturing as % of total RC LA jobs | HIGH | Derived |

---

## Contextual outcome variables (row-level)

These columns capture aggregate/reported outcomes from DWP and Task Force sources. They apply to specific quarters only (NaN elsewhere).

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| off_all_benefits_% | % of SSI+supply-chain claimants who ended benefit claim | REAL (Q3/Q4 2016 only) | DWP September 2016 report |
| still_on_benefits_% | % still on benefits (complement of above) | REAL (Q3/Q4 2016 only) | DWP September 2016 report |
| never_claimed_jsa_% | % who never made any benefit claim after closure | REAL | DWP March 2016 report (~25% of 2,070) |
| self_employment_% | % who started a business | REAL (approximate) | SSI Task Force reports (172 new businesses / 2,070) |
| self_employment_n | Count of new businesses started | REAL | SSI Task Force One Year On Report |
| training_courses_n | Number of training courses approved for SSI workers | REAL | SSI Task Force reports (15,510 total) |
| area_wage_change_% | Change in RC LA / Teesside area wages (pre vs post) | ESTIMATED_LOW | ONS area earnings estimates |
| ne_wage_change_% | Change in NE median wage (year on year from ASHE) | HIGH | Nomis ASHE NM_30_1 |
| manual_jobs_share_pre_% | Manual/blue-collar jobs as % of RC LA employment pre-closure | ESTIMATED_MED | BRES-derived |
| manual_jobs_share_post_% | Same share post-closure | ESTIMATED_MED | BRES-derived |

---

## ACC by age and gender (Stat-Xplore, RC LA) — acc_* columns

Real DWP administrative data. Parsed from Stat-Xplore ACC4 (Age by Gender, Redcar and Cleveland LA). Monthly counts aggregated to quarterly means. Baseline = mean of 2019 months. Excess = actual minus baseline, clamped at zero.

Data quality: HIGH for all acc_* columns.

| Column | Description |
|---|---|
| acc_16_24_male_q_mean | Mean monthly male ACC claimants aged 16-24 in RC LA |
| acc_16_24_female_q_mean | Mean monthly female ACC claimants aged 16-24 in RC LA |
| acc_25_49_male_q_mean | Mean monthly male ACC claimants aged 25-49 in RC LA |
| acc_25_49_female_q_mean | Mean monthly female ACC claimants aged 25-49 in RC LA |
| acc_50plus_male_q_mean | Mean monthly male ACC claimants aged 50+ in RC LA |
| acc_50plus_female_q_mean | Mean monthly female ACC claimants aged 50+ in RC LA |
| acc_total_q_mean | Total ACC claimants (all ages, both genders) in RC LA |
| acc_16_24_pct | 16-24 share of total RC LA ACC claimants (%) |
| acc_25_49_pct | 25-49 share |
| acc_50plus_pct | 50+ share |
| acc_male_pct | Male share of total RC LA ACC claimants (%) |
| acc_female_pct | Female share |
| acc_excess_16_24_pct | 16-24 share of EXCESS claimants (SSI shock proxy — proportion of shock-attributable claimants who are 16-24) |
| acc_excess_25_49_pct | 25-49 share of excess claimants |
| acc_excess_50plus_pct | 50+ share of excess claimants |
| acc_excess_male_pct | Male share of excess claimants |
| acc_data_quality | Always HIGH |

**Key finding:** RC LA 50+ share ~27% (prior assumption was 50%); male share ~62% (prior from steel-sector norms was 95%). The excess shares are a better proxy for SSI worker demographics than the total shares.

**UC caveat:** From mid-2016 onward, UC migration means some claimants who would previously appear in ACC/JSA do not appear. ACC counts from 2017 onward undercount true labour market activity relative to the pre-UC baseline.

---

## NE ACC by age band — ne_acc_* columns

North East regional ACC by model age band, from Nomis NM_162_1. Used as a regional benchmark to assess whether RC LA's age composition was typical of the broader NE claimant population.

Data quality: HIGH.

| Column | Description |
|---|---|
| ne_acc_16_24_q_mean | Mean monthly NE ACC claimants aged 16-24 |
| ne_acc_25_49_q_mean | Mean monthly NE ACC claimants aged 25-49 |
| ne_acc_50plus_q_mean | Mean monthly NE ACC claimants aged 50+ |
| ne_acc_total_q_mean | Total NE ACC claimants (all ages) |
| ne_acc_16_24_pct | 16-24 share of NE total ACC (%) |
| ne_acc_25_49_pct | 25-49 share |
| ne_acc_50plus_pct | 50+ share |
| ne_acc_data | Always HIGH |

---

## RC vs NE comparison indices — rc_vs_ne_* columns

Ratio of RC LA age share to NE age share for each model band. Index > 1 means RC LA is more heavily weighted toward that band than the North East average. Derived from acc_*_pct and ne_acc_*_pct.

| Column | Description |
|---|---|
| rc_vs_ne_16_24_idx | RC LA 16-24 ACC share / NE 16-24 ACC share |
| rc_vs_ne_25_49_idx | RC LA 25-49 share / NE 25-49 share |
| rc_vs_ne_50plus_idx | RC LA 50+ share / NE 50+ share |

---

## BEAO redundancy notifications — beao_* columns

ONS BEAO timeseries: national monthly redundancy notifications (GB, 000s). 3-month rolling average published by ONS. Provides macroeconomic context for the shock period.

| Column | Description | Real or estimated | Source |
|---|---|---|---|
| beao_redundancies_q_mean | Mean monthly redundancy notifications, GB (000s) | HIGH | ONS BEAO timeseries |
| beao_redundancies_data | HIGH if fetched successfully, UNAVAILABLE if ONS API failed | — | — |

---

## ASHE age-specific wage estimates — ashe_age_* columns

ONS does not publish regional ASHE by age band. These are estimated by taking national ASHE Table 6 (age group, weekly gross pay) band ratios (band median / all-workers median) and scaling them against the NE aggregate median (ne_median_weekly_pay_gbp).

Method: NE_band_est = NE_aggregate_median x (national_band_median / national_all_median)

Data quality: HIGH_SCALED — based on HIGH-quality national ASHE data but involves a scaling approximation. The national age ratio is applied uniformly; actual NE age-wage structure may differ.

Annual data (from ONS ASHE Table 6 annual releases). Same value repeated for all quarters in a year.

| Column | Description |
|---|---|
| ashe_age_data | Always HIGH_SCALED |
| ashe_age_16_24_median_est | Estimated NE median weekly pay for 16-24 age band (GBP) |
| ashe_age_16_24_p10_est | Estimated NE p10 weekly pay for 16-24 |
| ashe_age_16_24_p25_est | Estimated NE p25 weekly pay for 16-24 |
| ashe_age_16_24_p75_est | Estimated NE p75 weekly pay for 16-24 |
| ashe_age_16_24_p90_est | Estimated NE p90 weekly pay for 16-24 |
| ashe_age_16_24_sigma | Log-normal sigma for 16-24 wage distribution (derived from p25/p75) |
| ashe_age_nat_ratio_16_24 | National ratio: 16-24 median / all-workers median (used in scaling) |
| ashe_age_25_49_median_est | Estimated NE median weekly pay for 25-49 |
| ashe_age_25_49_p10_est | Estimated NE p10 for 25-49 |
| ashe_age_25_49_p25_est | Estimated NE p25 for 25-49 |
| ashe_age_25_49_p75_est | Estimated NE p75 for 25-49 |
| ashe_age_25_49_p90_est | Estimated NE p90 for 25-49 |
| ashe_age_25_49_sigma | Log-normal sigma for 25-49 |
| ashe_age_nat_ratio_25_49 | National ratio: 25-49 median / all-workers median |
| ashe_age_50plus_median_est | Estimated NE median weekly pay for 50+ |
| ashe_age_50plus_p10_est | Estimated NE p10 for 50+ |
| ashe_age_50plus_p25_est | Estimated NE p25 for 50+ |
| ashe_age_50plus_p75_est | Estimated NE p75 for 50+ |
| ashe_age_50plus_p90_est | Estimated NE p90 for 50+ |
| ashe_age_50plus_sigma | Log-normal sigma for 50+ |
| ashe_age_nat_ratio_50plus | National ratio: 50+ median / all-workers median |

**Typical values (2016):** 16-24: ~£411/wk, 25-49: ~£545/wk, 50+: ~£521/wk.
**National ratios (stable 2016-2019):** 16-24 ~0.84, 25-49 ~1.11, 50+ ~1.05.

---

## Statistical estimates — stat_* columns

All stat_* columns are model outputs, not observed data. They are applied uniformly across all row types in a quarter (the same values repeat across the 15 rows of a given quarter). Confidence intervals are 5th/95th percentile from bootstrap or Monte Carlo sampling.

See `stat_method` for the full method chain.

### Method 1: Exponential survival (still-seeking)

Calibrated to two DWP anchors: 25% still-seeking at 2016Q1, ~3% at 2016Q4. Analytically solved lambda = 0.265/month. Bootstrap n=10,000.

| Column | Description |
|---|---|
| stat_still_seeking_mid_% | % of original workforce still actively job-seeking (mid estimate) |
| stat_still_seeking_lo_% | 5th percentile CI |
| stat_still_seeking_hi_% | 95th percentile CI |

### Method 2: Aggregate log-normal underemployment

P(new wage < 90% of SSI pre-wage) using NE aggregate wage distribution (ne_median_weekly_pay_gbp, ne_wage_p10-p90). Wide CI reflects genuine NE wage dispersion.

| Column | Description |
|---|---|
| stat_underemployed_mid_% | % of resolved workers earning below 90% of pre-displacement wage (mid) |
| stat_underemployed_lo_% | 5th percentile CI |
| stat_underemployed_hi_% | 95th percentile CI |

### Method 2b: Age-specific log-normal underemployment

Same approach as Method 2 but uses ashe_age_*_median_est and ashe_age_*_sigma for each age band. Applies AGE_WAGE_RATIOS (16-24: 0.840, 25-49: 1.109, 50+: 1.049) to shift the pre-displacement wage benchmark.

| Column | Description |
|---|---|
| stat_underemployed_16_24_mid_% | Age-specific underemployment estimate for 16-24 (mid) |
| stat_underemployed_16_24_lo_% | 5th percentile CI |
| stat_underemployed_16_24_hi_% | 95th percentile CI |
| stat_underemployed_25_49_mid_% | 25-49 (mid) |
| stat_underemployed_25_49_lo_% | 5th percentile |
| stat_underemployed_25_49_hi_% | 95th percentile |
| stat_underemployed_50plus_mid_% | 50+ (mid) |
| stat_underemployed_50plus_lo_% | 5th percentile |
| stat_underemployed_50plus_hi_% | 95th percentile |

### Method 3: Bayesian Dirichlet-Multinomial outcome split

Cross-episode prior: retrain 25%, share_similar 40%, exit 35% (concentration kappa=20, alpha=[5,8,7]).
RC LA evidence: ACC-derived outcome proportions from panel outcome columns, weighted at 0.3.
Posterior updated over 17 quarters. 10,000 Dirichlet samples for CIs.

These are absolute shares of the original workforce (not of the resolved pool only).

| Column | Description |
|---|---|
| stat_retrain_mid_% | % of original workforce who retrained (mid) |
| stat_retrain_lo_% | 5th percentile CI |
| stat_retrain_hi_% | 95th percentile CI |
| stat_share_similar_mid_% | % who found similar-sector work (mid) |
| stat_share_similar_lo_% | 5th percentile |
| stat_share_similar_hi_% | 95th percentile |
| stat_exit_mid_% | % who exited the labour market (mid) |
| stat_exit_lo_% | 5th percentile |
| stat_exit_hi_% | 95th percentile |
| stat_dirichlet_kappa | Prior concentration (kappa=20, effective prior sample size) |
| stat_dirichlet_pi_retrain | Posterior mean proportion for retrain (~0.121) |
| stat_dirichlet_pi_similar | Posterior mean proportion for share_similar (~0.326) |
| stat_dirichlet_pi_exit | Posterior mean proportion for exit (~0.553) |
| stat_method | Full method chain string |

**Posterior mean split:** retrain 12.1%, share_similar 32.6%, exit 55.3%.
The prior said 25/40/35. RC LA ACC data pulled exit up substantially, reflecting an older workforce in a depressed local labour market with few comparable job openings.

---

## Known gaps

- **ESA caseload:** Health-benefit exit data not yet integrated. integrate_esa.py is ready; requires two Stat-Xplore downloads (ESA Caseload 4, RC LA, split at Feb 2018 boundary).
- **JSA duration:** Suppressed at LA level on Nomis. North East regional availability not yet verified.
- **Vacancy data:** Nomis regional vacancy series (NM_5_1, NM_19_1-24_1, NM_89_1) all end before 2012. Not available for 2015-2019.
- **UC migration adjustment:** ACC counts from 2017 onward are affected by Universal Credit rollout. No formal adjustment applied; rows from 2017Q1 onward carry a caveat in the notes column.
- **ASHE age x region:** ONS does not publish ASHE by age band at regional level. The ashe_age_* columns use national age ratios scaled to NE aggregate — an approximation.

---

## Sources

| Source | Series | Data quality | Used for |
|---|---|---|---|
| DWP SSI JSA Statistics, March 2016 | Bespoke DWP tracking DB | HIGH | 25% still-seeking anchor (2016Q1) |
| DWP SSI Updated Statistics, September 2016 | Bespoke DWP tracking DB | HIGH | 93% off-benefits anchor (ref date 31 Aug 2016) |
| Nomis NM_162_1 | Alternative Claimant Count | HIGH | RC LA and NE claimant counts |
| Nomis NM_17_1 | Annual Population Survey | HIGH | NE unemployment/employment/activity rates |
| Nomis NM_30_1 | ASHE aggregate | HIGH | NE wage distribution |
| Nomis NM_189_1 | BRES | HIGH | RC LA sector employment |
| ONS ASHE Table 6 | Age group weekly pay | HIGH (national), HIGH_SCALED (NE est.) | Age-specific wage distributions |
| ONS BEAO timeseries | Redundancy notifications | HIGH | National macroeconomic context |
| Stat-Xplore ACC4 | Age by gender, RC LA | HIGH | Real RC LA ACC by age and gender |
