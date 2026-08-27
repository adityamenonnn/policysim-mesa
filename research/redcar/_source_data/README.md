# Source data files

Raw downloads from DWP Stat-Xplore. Do not edit. Re-run the integration scripts to regenerate panel columns from these files.

| File | Dataset | Geography | Period | Used by |
|---|---|---|---|---|
| acc4_age_gender_redcar_cleveland.xlsx | ACC4 - Alternative Claimant Count by Age and Gender | Redcar and Cleveland LA | Oct 2015 - Dec 2019 | integrate_statxplore.py |
| esa_redcar_to_feb2018.xlsx | ESA - Data to February 2018 | Redcar and Cleveland LA | Nov 2008 - Feb 2018 | integrate_esa.py |
| esa_redcar_from_may2018.xlsx | ESA - Data from May 2018 | Redcar and Cleveland LA | May 2018 - Nov 2025 | integrate_esa.py |

## To regenerate panel columns from these files

```bash
cd /Users/adityamenon/Documents/PolicySim/policysim-mesa

# ACC age/gender claimant columns
python3 research/redcar/integrate_statxplore.py

# ESA caseload columns
python3 research/redcar/integrate_esa.py \
  --pre research/redcar/_source_data/esa_redcar_to_feb2018.xlsx \
  --post research/redcar/_source_data/esa_redcar_from_may2018.xlsx

# Nomis/ONS pulls (fetches live — uses cache in _nomis_cache/)
python3 research/redcar/fetch_redcar_extra.py

# ASHE age-specific wage estimates (fetches live — uses cache in _ashe_cache/)
python3 research/redcar/parse_ashe_age.py

# Statistical estimates (Methods 1, 2, 2b, 3)
python3 research/redcar/estimate_statistical.py
```
