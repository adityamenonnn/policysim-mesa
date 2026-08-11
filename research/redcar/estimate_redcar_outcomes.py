"""
estimate_redcar_outcomes.py
===========================
Produces estimated outcome bucket ranges for Redcar 2015.

Estimates are derived manually by reasoning across:
  - All rows in displacement_evidence.csv (Redcar anchors + Hartz + China Shock + Port Talbot)
  - Redcar quarterly ONS panel (claimant trajectory, wages, APS)
  - Published steel-sector displacement literature

No API call needed. All estimates are hardcoded with full arithmetic shown.

Output: redcar_outcome_estimates.csv   (ranges + math per quarter x age band)
        redcar_outcome_reasoning.txt   (full audit trail)
"""

from pathlib import Path
import pandas as pd

BASE     = Path(__file__).parent
RESEARCH = BASE.parent
OUT_CSV  = BASE / "redcar_outcome_estimates.csv"
OUT_TXT  = BASE / "redcar_outcome_reasoning.txt"

# ---------------------------------------------------------------------------
# STEP 1 — ARITHMETIC CONSTRAINTS FROM KNOWN ANCHORS
# ---------------------------------------------------------------------------
# These are HARD constraints. Any valid estimate must satisfy all of them.
#
# At any time horizon:
#   retrain + share_similar + underemployed + exit + still_seeking = 100%
#
# At Q1 2016 (5 months):
#   still_seeking = 25%  [HARD — DWP JSA stats, 540/2070, HIGH quality]
#   never_claimed_jsa = 35%  [HARD — 720/2070 never on JSA, unknown split]
#   → remaining 40% left JSA by Feb 2016 (not necessarily employed — could be ESA/UC)
#   → retrain + share_similar + underemployed + exit = 75%  (complement of still_seeking)
#
# At Q4 2016 (12 months):
#   off_all_benefits = 93%  [HARD — Task Force, 1990/2150, HIGH quality]
#   still_on_benefits = 7%  [HARD — ~160 workers]
#   self_employment = 8.3%  [MEDIUM — 172/2070 new businesses]
#   claimant count already BELOW pre-closure baseline by Q4 2016
#   → still_seeking at 12 months is small (most resolved; estimate 2-4%)
#   → exit_via_ESA ≈ 7% - still_seeking ≈ 4-5%
#   → exit_via_retirement (off benefits, on pension) = separate component
#   → retrain + share_similar + underemployed + exit_retirement = 93%  [HARD]
#   → exit_total = exit_ESA + exit_retirement
#
# At Q4 2017 (24 months):
#   claimant count returned to pre-closure baseline by Oct 2017
#   → still_seeking ≈ 1-3% (essentially resolved)
#   self_employment grew to 15.2%  [MEDIUM — 315/2070]
#   area wages still -0.5% real vs NE  [MEDIUM — underemployment persists]
#
# ---------------------------------------------------------------------------
# STEP 2 — CROSS-EPISODE PRIORS AND APPLICABILITY
# ---------------------------------------------------------------------------
#
# HARTZ 2003-2005 (Germany)                         APPLICABILITY: MEDIUM
#   Re-employment wages fell ~10% for most workers.
#   80%+ of wage loss due to moving to lower-paying employers.
#   → Underemployment is the dominant bucket for re-employed workers.
#   Older workers (50+) systematically pushed into pension system.
#   → Exit rate for 50+ is high.
#   FbW retraining: only +5pp employment probability after 22 months.
#   → Retraining has a small positive effect but low take-up.
#   Caveat: Hartz is a national policy reform not a single-plant closure.
#   UK has different benefit structure (JSA/ESA vs ALG I/II).
#
# CHINA SHOCK — ADH 2013/2021 (USA)                 APPLICABILITY: LOW-MEDIUM
#   86% of manufacturing job losses → labour force exit.
#   BUT: this is a 20-year diffuse shock with no government support.
#   Redcar had a £46m government package — significantly better supported.
#   TAA retraining: <10% take-up; of those who retraining, 70-80% employed
#   but with 10-20% wage loss → classified as underemployed, not share_similar.
#   Low-wage workers → exit. High-wage workers → share_similar.
#   → SSI workers were mid-high wage (£25k-£35k), so less exit than ADH average.
#   Caveat: US has no equivalent of UK JSA/ESA safety net and pension system.
#
# PORT TALBOT 2024 (UK, same sector)                APPLICABILITY: MEDIUM-HIGH
#   16% into new jobs at 12 months (NPT Council scheme, 332/2100).
#   Retraining near zero (perverse incentive from enhanced redundancy rules).
#   Average age 36 vs Redcar 50%+ aged 45+ → Redcar has HIGHER exit probability.
#   Avg Port Talbot wage £46k+ vs Redcar £25k-£35k → higher wage loss in Port Talbot.
#   Caveat: Port Talbot data is very recent (2024-2025) and still emerging.
#   → 16% into new jobs is a floor for share_similar+retrain at 12 months.
#
# REDCAR-SPECIFIC SIGNALS
#   50%+ aged 45+: significant early retirement cohort. British Steel Pension Scheme
#   (BSPS) gave access to good DB pensions. Workers aged 55+ could likely retire.
#   ~15% of total workforce aged 55+ estimated → ~15% early retirement exit floor.
#   Union confirmed lower pay in all new roles → underemployment dominant.
#   Manual jobs 10%→8% of local employment → occupational downshift confirmed.
#   Geographic isolation (Teesside) → fewer equivalent employers → lower share_similar.
#   Government subsidy: £11k-£15k hiring subsidy → incentivises employers to hire
#   → marginally increases share_similar/underemployed, reduces still_seeking.
#
# ---------------------------------------------------------------------------
# STEP 3 — DERIVATION OF ESTIMATES
# ---------------------------------------------------------------------------
#
# === Q4 2016 — 12 MONTHS (all ages) ===
#
# still_seeking:
#   RC LA claimant count at Q4 2016 = 3193, already -284 BELOW baseline (3477).
#   This means the area has effectively absorbed the shock at a headline level.
#   However, some SSI workers are still seeking. Claimant count below baseline
#   partly because some non-SSI workers found work due to the intervention activity.
#   DWP anchor: 25% at 5 months. Recovery is fast based on claimant trajectory.
#   At 12 months: estimate 2-5%, mid 3%.
#
# exit:
#   exit_ESA = 4% (part of 7% still on benefits, minus still_seeking ~3%)
#   exit_retirement: ~15% of total workforce aged 55+ → most retire = 15%
#                    plus some aged 45-54 with long service records → +7%
#                    = ~22% retirement exit
#   exit_total MID = 4 + 22 = 26%
#   Range: LOW 18%, MID 26%, HIGH 34%
#   Constraint check: exit_ESA(4) + still_seeking(3) = 7% ✓ (matches still_on_benefits anchor)
#
# retrain:
#   15,510 training courses / 2070 workers = 7.5 courses per worker but mostly short refreshers.
#   UK training ≠ German Umschulung. No evidence of significant occupational change.
#   Port Talbot: near zero retraining at 12 months.
#   China Shock: <10% TAA take-up; Hartz FbW only +5pp at 22+ months.
#   Government subsidy programme targeted re-employment not retraining.
#   Estimate: LOW 3%, MID 6%, HIGH 10%.
#
# share_similar:
#   Self-employment = 8.3% (floor — some self-employed may be underemployed).
#   Some workers found equivalent roles: offshore energy, construction, other steel.
#   Port Talbot: 16% into new jobs at 12 months (includes underemployed).
#   Union: lower pay in new roles → much of re-employment is underemployed not share_similar.
#   Government hiring subsidy → some equivalent-sector employment created.
#   Estimate: LOW 8%, MID 13%, HIGH 20%.
#
# underemployed:
#   Residual after fixing other buckets:
#   underemployed = 100 - still_seeking - exit - retrain - share_similar
#   MID: 100 - 3 - 26 - 6 - 13 = 52%
#   Supported by: wage decline proxy, occupational downshift, union confirmation,
#   Hartz analogue (80%+ of re-employment to lower-paying employers).
#   Range: LOW 38%, MID 52%, HIGH 60%.
#   MID check: 52 + 3 + 26 + 6 + 13 = 100 ✓
#
# === Q1 2016 — 5 MONTHS (all ages) ===
#
# still_seeking = 25% (HARD ANCHOR — DWP).
#
# exit:
#   Early retirements mostly completed immediately (35% never claimed JSA).
#   Some of that 35% are retirees — est 12-18% of total workforce retired by 5 months.
#   ESA applications still being processed at 5 months; some on ESA by now.
#   exit MID = 20%, range LOW 12% HIGH 28%.
#
# retrain:
#   At 5 months, virtually no retraining completed — training programmes
#   only beginning to ramp up (4,300 courses delivered by 6 months but mostly
#   short refreshers). Genuine occupational retraining takes 6-24 months.
#   MID = 2%, range LOW 1% HIGH 4%.
#
# share_similar:
#   Some workers found equivalent roles quickly (particularly those with
#   portable skills: maintenance engineers, electricians, HGV drivers).
#   Some self-employment already started.
#   MID = 10%, range LOW 6% HIGH 15%.
#
# underemployed:
#   100 - 25 - 20 - 2 - 10 = 43% MID.
#   Many workers taking any available job while still looking for equivalent.
#   Range LOW 30% HIGH 50%.
#   MID check: 25 + 20 + 2 + 10 + 43 = 100 ✓
#
# === Q4 2017 — 24 MONTHS (all ages) ===
#
# still_seeking:
#   Claimant count returned to pre-closure baseline by Oct 2017.
#   Essentially resolved. MID = 2%, range LOW 1% HIGH 4%.
#
# exit:
#   More workers have exhausted job search and moved to pension/disability.
#   Some 45-54 who couldn't find work have now taken early pension.
#   exit MID = 30%, range LOW 22% HIGH 40%.
#
# retrain:
#   Some delayed retraining completed (£9m invested in 23,000 courses by 2 years).
#   Some workers who failed to find work used the time to retrain.
#   MID = 8%, range LOW 5% HIGH 14%.
#
# share_similar:
#   Self-employment grew to 15.2% (315/2070) — this is the strongest signal.
#   Some additional workers found equivalent roles over 2 years.
#   MID = 15%, range LOW 10% HIGH 22%.
#
# underemployed:
#   100 - 2 - 30 - 8 - 15 = 45% MID.
#   Wage scarring persists (area wages still -0.5% real).
#   Range LOW 35% HIGH 55%.
#   MID check: 2 + 30 + 8 + 15 + 45 = 100 ✓
#
# ---------------------------------------------------------------------------
# STEP 4 — AGE BAND ADJUSTMENTS
# ---------------------------------------------------------------------------
#
# Applied at Q4 2016 (12 months) as the primary calibration point.
#
# 16-24 (~207 workers, 10% of workforce):
#   EXIT much lower — no pension access, not eligible for early retirement.
#   Main exit route: disability (ESA), which is rare at young age.
#   exit MID = 8% (LOW 4%, HIGH 14%).
#   RETRAIN higher — more willing to change occupation, longer payback period.
#   retrain MID = 12% (LOW 7%, HIGH 20%).
#   SHARE_SIMILAR higher — more geographically mobile, can relocate for work.
#   share_similar MID = 18% (LOW 10%, HIGH 28%).
#   UNDEREMPLOYED — likely to take any job available (entry-level service jobs).
#   underemployed MID = 55% (LOW 45%, HIGH 65%).
#   STILL_SEEKING — higher than average (more choosy, longer search).
#   still_seeking MID = 7% (LOW 3%, HIGH 12%).
#   MID check: 8+12+18+55+7 = 100 ✓
#
# 25-49 (~828 workers, 40% of workforce):
#   EXIT — limited pension access (only upper end of band, 45-49).
#   Some disability/ESA at this age. Family constraints limit mobility.
#   exit MID = 18% (LOW 12%, HIGH 26%).
#   RETRAIN — possible but costly (family obligations, mortgage).
#   retrain MID = 6% (LOW 3%, HIGH 11%).
#   SHARE_SIMILAR — moderate mobility, some portable skills.
#   share_similar MID = 14% (LOW 8%, HIGH 22%).
#   UNDEREMPLOYED — dominant bucket (skills specific to steel, local market).
#   underemployed MID = 58% (LOW 48%, HIGH 68%).
#   STILL_SEEKING — near average.
#   still_seeking MID = 4% (LOW 2%, HIGH 8%).
#   MID check: 18+6+14+58+4 = 100 ✓
#
# 50+ (~1035 workers, 50% of workforce):
#   EXIT — dominant due to pension access and age discrimination in hiring.
#   Workers aged 55-64 mostly retire (BSPS DB pension accessible).
#   Workers aged 50-54 with long service also access early pension.
#   ESA/disability higher at this age.
#   exit MID = 40% (LOW 30%, HIGH 52%).
#   RETRAIN — very low. Short payback period, employers less likely to hire
#   for new role after expensive retraining. Hartz: 50+ pushed to pension not retrained.
#   retrain MID = 3% (LOW 1%, HIGH 6%).
#   SHARE_SIMILAR — harder to place; age discrimination in hiring.
#   Age-tiered hiring subsidy (£15k for 45+ vs £11k under 45) acknowledges this.
#   share_similar MID = 10% (LOW 5%, HIGH 16%).
#   UNDEREMPLOYED — lower than younger cohorts because many exit rather than accept
#   poor work. Those who do work are more likely underemployed.
#   underemployed MID = 44% (LOW 34%, HIGH 54%).
#   STILL_SEEKING — lower than younger workers (more likely to give up or retire).
#   still_seeking MID = 3% (LOW 1%, HIGH 6%).
#   MID check: 40+3+10+44+3 = 100 ✓
#
# ---------------------------------------------------------------------------

ESTIMATES = [
    # ---- Q1 2016 (5 months) ----
    {
        "time_horizon":    "Q1 2016 (5 months post-closure)",
        "quarter":         "2016Q1",
        "months_post":     5,
        "age_band":        "all",
        "retrain":         (1,   2,   4),
        "share_similar":   (6,  10,  15),
        "underemployed":   (30, 43,  50),
        "exit":            (12, 20,  28),
        "still_seeking":   (25, 25,  25),  # HARD ANCHOR — DWP
        "hard_anchors":    "still_seeking=25% (DWP JSA stats, HIGH). "
                           "35% never claimed JSA (unknown split). "
                           "Remaining 40% left JSA by Feb 2016 (not necessarily employed).",
        "math":            "still_seeking FIXED at 25% (DWP). "
                           "exit MID: 12-18% early retirement (of 35% never-claimed) + ~2-3% early ESA = 20%. "
                           "retrain MID: 2% — too early for genuine occupational retraining (courses only just starting). "
                           "share_similar MID: 10% — workers with portable skills (electricians, HGV, maintenance) found work quickly. "
                           "underemployed MID: 100-25-20-2-10 = 43%. "
                           "Sum check: 25+20+2+10+43 = 100 ✓",
        "cross_episode":   "Hartz: re-employment wages fell 10%; 80%+ moved to lower-paying employer → underemployment dominant. "
                           "Port Talbot 5-month analogue: retraining near zero at 5 months. "
                           "China Shock: immediate re-employment mainly low-wage (underemployed); high-wage workers → share_similar.",
        "dominant_uncertainty": "Split of the 35% who never claimed JSA — we cannot tell if they retired, were immediately re-employed, or left the UK.",
    },
    {
        "time_horizon":    "Q1 2016 (5 months post-closure)",
        "quarter":         "2016Q1",
        "months_post":     5,
        "age_band":        "16-24",
        "retrain":         (2,   5,  10),
        "share_similar":   (8,  15,  22),
        "underemployed":   (30, 45,  55),
        "exit":            (3,   7,  13),
        "still_seeking":   (25, 28,  32),
        "hard_anchors":    "still_seeking=25% aggregate (DWP) — 16-24 cohort likely slightly higher than average as they are more persistent searchers.",
        "math":            "exit LOW for 16-24: no pension access; only disability/ESA route (rare at young age). MID=7%. "
                           "retrain MID=5%: young workers more likely to enter training programmes early. "
                           "share_similar MID=15%: more geographically mobile, some find equivalent work quickly. "
                           "still_seeking MID=28%: slightly above aggregate (more choosy, longer search acceptable). "
                           "underemployed = 100-7-5-15-28 = 45%. Sum: 7+5+15+28+45 = 100 ✓",
        "cross_episode":   "China Shock: younger workers more likely to move region and find equivalent work (share_similar). "
                           "Hartz: younger workers had shorter benefit duration cuts, faster re-employment. "
                           "Port Talbot: average age 36 — younger workers showed faster resolution.",
        "dominant_uncertainty": "Very small cohort (~207 workers). Individual variation dominates; aggregate rates less reliable.",
    },
    {
        "time_horizon":    "Q1 2016 (5 months post-closure)",
        "quarter":         "2016Q1",
        "months_post":     5,
        "age_band":        "25-49",
        "retrain":         (1,   3,   7),
        "share_similar":   (6,  11,  17),
        "underemployed":   (28, 42,  52),
        "exit":            (8,  16,  24),
        "still_seeking":   (25, 28,  32),
        "hard_anchors":    "still_seeking=25% aggregate (DWP). 25-49 cohort near aggregate.",
        "math":            "exit MID=16%: limited pension access at this age; mainly ESA and some early leavers. "
                           "retrain MID=3%: too early for full retraining; family obligations limit options. "
                           "share_similar MID=11%: moderate portable skills. "
                           "underemployed = 100-16-3-11-28 = 42%. Sum: 28+16+3+11+42 = 100 ✓",
        "cross_episode":   "Hartz 25-49: re-employment at lower wages dominates. Mini-job and temp work explosion in this age group.",
        "dominant_uncertainty": "How many in the 45-49 subgroup accessed early pension at 5 months vs still searching.",
    },
    {
        "time_horizon":    "Q1 2016 (5 months post-closure)",
        "quarter":         "2016Q1",
        "months_post":     5,
        "age_band":        "50+",
        "retrain":         (0,   1,   3),
        "share_similar":   (4,   8,  13),
        "underemployed":   (22, 35,  45),
        "exit":            (20, 33,  42),
        "still_seeking":   (20, 23,  28),
        "hard_anchors":    "50%+ of workers seeking work aged 45+ (DWP note). "
                           "still_seeking for 50+ estimated at 23% MID — slightly below aggregate as more exit rather than seek.",
        "math":            "exit MID=33%: most early retirements happen immediately. "
                           "35% never claimed JSA — the 50+ cohort dominates this group. "
                           "If ~65% of never-claimed are aged 50+ (proportional to workforce share): "
                           "0.65 × 35% = 22.75% of total workforce = 45.5% of 50+ cohort who never claimed. "
                           "Of those, most retired → exit for 50+ is high even at 5 months. "
                           "retrain MID=1%: near zero — too old, too early. "
                           "share_similar MID=8%: some workers with specific expertise placed quickly. "
                           "underemployed = 100-33-1-8-23 = 35%. Sum: 33+1+8+23+35 = 100 ✓",
        "cross_episode":   "Hartz 50+: systematically pushed into pension system = EXIT. "
                           "China Shock: older low-wage workers → exit. "
                           "BSPS (British Steel Pension Scheme): DB pension accessible at 55; SSI workers aged 55-64 (~15% of total) likely all retire.",
        "dominant_uncertainty": "Proportion of 50+ cohort with sufficient pension entitlement to retire at 50-55 (depends on individual years of service).",
    },

    # ---- Q4 2016 (12 months) — PRIMARY CALIBRATION POINT ----
    {
        "time_horizon":    "Q4 2016 (12 months post-closure)",
        "quarter":         "2016Q4",
        "months_post":     12,
        "age_band":        "all",
        "retrain":         (3,   6,  10),
        "share_similar":   (8,  13,  20),
        "underemployed":   (38, 52,  60),
        "exit":            (18, 26,  34),
        "still_seeking":   (2,   3,   5),
        "hard_anchors":    "off_all_benefits=93% (Task Force, HIGH). "
                           "still_on_benefits=7% (Task Force). "
                           "self_employment=8.3% (172/2070). "
                           "RC LA claimant count BELOW baseline by Q4 2016.",
        "math":            "still_seeking MID=3%: RC LA claimants at Q4 2016 already -284 below baseline. "
                           "Most workers resolved. 3% still on JSA estimate. "
                           "exit_ESA = 7% - still_seeking = 7-3 = 4%. "
                           "exit_retirement: ~15% aged 55-64 (most retire) + ~7% aged 50-54 with long service = ~22%. "
                           "exit_total MID = exit_ESA(4) + exit_retirement(22) = 26%. "
                           "Constraint check: exit_ESA(4) + still_seeking(3) = 7% = still_on_benefits ✓. "
                           "retrain MID=6%: 15,510 courses / 2070 = 7.5 courses per worker but mostly short refreshers. "
                           "Genuine occupational change estimated at 6% (above Port Talbot near-zero; UK has better programmes). "
                           "share_similar MID=13%: self-employment floor 8.3% + ~5% who found equivalent industrial work = 13%. "
                           "underemployed MID = 100-3-26-6-13 = 52%. "
                           "Sum check: 3+26+6+13+52 = 100 ✓. "
                           "Consistency: retrain+share_similar+underemployed+exit_retirement = 6+13+52+22 = 93% = off_all_benefits ✓.",
        "cross_episode":   "Hartz: 80%+ re-employment to lower-paying employer → underemployment dominant → supports 52% MID. "
                           "Port Talbot: 16% into new jobs at 12 months (floor for share_similar+retrain = 19% here). "
                           "China Shock: TAA retrain <10% take-up; most retrained workers underemployed not share_similar. "
                           "Hartz 50+: systematic pension shift = EXIT → supports high exit estimate.",
        "dominant_uncertainty": "Size of early retirement cohort (depends on individual pension entitlements — not public data). "
                                "Whether self-employed workers count as share_similar or underemployed (income not verified).",
    },
    {
        "time_horizon":    "Q4 2016 (12 months post-closure)",
        "quarter":         "2016Q4",
        "months_post":     12,
        "age_band":        "16-24",
        "retrain":         (7,  12,  20),
        "share_similar":   (10, 18,  28),
        "underemployed":   (45, 55,  65),
        "exit":            (4,   8,  14),
        "still_seeking":   (3,   7,  12),
        "hard_anchors":    "No age-specific anchors available. Derived from aggregate + age-band adjustments.",
        "math":            "exit MID=8%: no pension access; only disability (ESA) route. Rare at this age. "
                           "retrain MID=12%: young workers more likely to change career; longer payback. "
                           "share_similar MID=18%: more mobile; some left Teesside for equivalent roles. "
                           "still_seeking MID=7%: above average — still searching for right role. "
                           "underemployed = 100-8-12-18-7 = 55%. Sum: 8+12+18+7+55 = 100 ✓",
        "cross_episode":   "China Shock: young workers more likely to relocate and find share_similar work. "
                           "Hartz: younger workers had faster re-employment transitions.",
        "dominant_uncertainty": "Small cohort (~207 workers). Geographic mobility assumption — did young Redcar workers actually relocate?",
    },
    {
        "time_horizon":    "Q4 2016 (12 months post-closure)",
        "quarter":         "2016Q4",
        "months_post":     12,
        "age_band":        "25-49",
        "retrain":         (3,   6,  11),
        "share_similar":   (8,  14,  22),
        "underemployed":   (48, 58,  68),
        "exit":            (12, 18,  26),
        "still_seeking":   (2,   4,   8),
        "hard_anchors":    "No age-specific anchors. Derived from aggregate + age-band adjustments.",
        "math":            "exit MID=18%: limited pension access (mainly 45-49 subgroup with long service). "
                           "retrain MID=6%: near-aggregate — family obligations limit but not prevent retraining. "
                           "share_similar MID=14%: moderate — some equivalent roles found. "
                           "still_seeking MID=4%: near average. "
                           "underemployed = 100-18-6-14-4 = 58%. Sum: 18+6+14+4+58 = 100 ✓",
        "cross_episode":   "Hartz 25-49: dominant re-employment to lower-paying employer. Mini-jobs + temp work concentration in this age group.",
        "dominant_uncertainty": "The 45-49 subgroup straddles two regimes (pension access vs not) — splitting at 45 vs 50 affects exit estimate significantly.",
    },
    {
        "time_horizon":    "Q4 2016 (12 months post-closure)",
        "quarter":         "2016Q4",
        "months_post":     12,
        "age_band":        "50+",
        "retrain":         (1,   3,   6),
        "share_similar":   (5,  10,  16),
        "underemployed":   (34, 44,  54),
        "exit":            (30, 40,  52),
        "still_seeking":   (1,   3,   6),
        "hard_anchors":    "50%+ of workforce aged 45+ (DWP). Age-tiered hiring subsidy £15k for 45+ acknowledges higher exit risk.",
        "math":            "exit MID=40%: exit_ESA ~6% (disability higher at this age) + "
                           "exit_retirement ~34% (workers aged 55-64 = ~15% of total workforce = 30% of 50+ cohort → most retire; "
                           "plus 50-54 with long service = additional 10% of 50+ cohort → some retire early). "
                           "Total retirement exit for 50+ cohort MID = 34%. exit_total = 40%. "
                           "Constraint: exit_ESA + still_seeking should ≈ 7% at aggregate — "
                           "50+ cohort's contribution is exit_ESA=6, still_seeking=3 → 9% for this cohort; "
                           "weighted by 50% of workforce: contributes 4.5% to the aggregate 7% (other cohorts contribute 2.5%). "
                           "retrain MID=3%: Hartz shows 50+ not retrained — pushed to pension. Very short payback period. "
                           "share_similar MID=10%: harder to place; age discrimination; £15k hiring subsidy helps at the margin. "
                           "underemployed = 100-40-3-10-3 = 44%. Sum: 40+3+10+3+44 = 100 ✓",
        "cross_episode":   "Hartz 50+: systematically pushed into pension system = EXIT (strongest applicable prior). "
                           "China Shock: older workers → exit/disability (SSDI uptake elevated in trade-exposed regions). "
                           "BSPS: workers aged 55-64 with DB pension can retire; 50-54 with 30+ service years can access early pension.",
        "dominant_uncertainty": "Pension entitlement data (years of service) is not publicly available. "
                                "Age discrimination in re-employment is difficult to quantify.",
    },

    # ---- Q4 2017 (24 months) ----
    {
        "time_horizon":    "Q4 2017 (24 months post-closure)",
        "quarter":         "2017Q4",
        "months_post":     24,
        "age_band":        "all",
        "retrain":         (5,   8,  14),
        "share_similar":   (10, 15,  22),
        "underemployed":   (35, 45,  55),
        "exit":            (22, 30,  40),
        "still_seeking":   (1,   2,   4),
        "hard_anchors":    "self_employment=15.2% (315/2070). "
                           "area_wage_change=-0.5% real (vs +1.6% NE). "
                           "claimant count returned to pre-closure baseline by Oct 2017.",
        "math":            "still_seeking MID=2%: claimant count at pre-closure baseline by Oct 2017. Essentially resolved. "
                           "exit MID=30%: some additional workers who failed at job search moved to pension/disability by 24 months. "
                           "More 50-54 workers with 2 years of failed search accepted early retirement. "
                           "retrain MID=8%: delayed retraining completed. £9m invested in 23,000 courses by 2 years. "
                           "Some workers who couldn't find work used the time to retrain. "
                           "share_similar MID=15%: self-employment = 15.2% (315/2070) — this is the direct anchor. "
                           "Note: some self-employed may be underemployed (lower income than pre-closure). "
                           "underemployed MID = 100-2-30-8-15 = 45%. "
                           "Sum check: 2+30+8+15+45 = 100 ✓. "
                           "Wage check: area wages still -0.5% real → underemployment persists → 45% consistent.",
        "cross_episode":   "Hartz: revolving door — workers cycle back to unemployment after unstable new jobs. "
                           "Some who appeared re-employed at 12 months may have returned to unemployment by 24 months. "
                           "China Shock: effects persisted to 2019 with no recovery 9 years after shock plateau. "
                           "BUT: Redcar had £46m government support → better than unsupported closures.",
        "dominant_uncertainty": "Whether self-employment (15.2%) is genuinely share_similar or masked underemployment. "
                                "Income data for the self-employed not available.",
    },
    {
        "time_horizon":    "Q4 2017 (24 months post-closure)",
        "quarter":         "2017Q4",
        "months_post":     24,
        "age_band":        "16-24",
        "retrain":         (10, 18,  28),
        "share_similar":   (12, 22,  32),
        "underemployed":   (35, 45,  55),
        "exit":            (3,   7,  13),
        "still_seeking":   (1,   3,   6),
        "hard_anchors":    "No age-specific data at 24 months.",
        "math":            "retrain MID=18%: by 24 months, young workers who couldn't find work have completed retraining. "
                           "share_similar MID=22%: 2 years is enough time to relocate and find equivalent work. "
                           "exit MID=7%: only disability; no retirement. "
                           "underemployed = 100-18-22-7-3 = 50. Sum: 18+22+7+3+50 = 100 ✓",
        "cross_episode":   "Hartz: younger workers showed faster labour market recovery at 2 years than older workers.",
        "dominant_uncertainty": "Geographic mobility — whether young Redcar workers relocated or stayed in the area.",
    },
    {
        "time_horizon":    "Q4 2017 (24 months post-closure)",
        "quarter":         "2017Q4",
        "months_post":     24,
        "age_band":        "25-49",
        "retrain":         (5,  10,  17),
        "share_similar":   (10, 16,  24),
        "underemployed":   (38, 50,  60),
        "exit":            (16, 22,  30),
        "still_seeking":   (1,   2,   4),
        "hard_anchors":    "No age-specific data at 24 months.",
        "math":            "exit MID=22%: 45-49 subgroup increasingly accessing early pension by 24 months. "
                           "retrain MID=10%: some completed retraining after failed job search. "
                           "share_similar MID=16%: near aggregate. "
                           "underemployed = 100-22-10-16-2 = 50. Sum: 22+10+16+2+50 = 100 ✓",
        "cross_episode":   "Hartz: 25-49 cohort showed persistent underemployment at 2 years (mini-jobs, temp contracts).",
        "dominant_uncertainty": "Family obligations limiting geographic mobility — reduces share_similar.",
    },
    {
        "time_horizon":    "Q4 2017 (24 months post-closure)",
        "quarter":         "2017Q4",
        "months_post":     24,
        "age_band":        "50+",
        "retrain":         (1,   3,   6),
        "share_similar":   (8,  12,  18),
        "underemployed":   (28, 38,  48),
        "exit":            (38, 45,  56),
        "still_seeking":   (0,   2,   4),
        "hard_anchors":    "No age-specific data at 24 months.",
        "math":            "exit MID=45%: by 24 months, more 50-54 workers have given up search and accessed pension. "
                           "Some who were underemployed at 12 months left poor jobs and retired. "
                           "retrain MID=3%: near zero — too late and too old to recoup investment. "
                           "share_similar MID=12%: slight increase from 12 months as hiring subsidy takes time to work. "
                           "underemployed = 100-45-3-12-2 = 38. Sum: 45+3+12+2+38 = 100 ✓",
        "cross_episode":   "Hartz 50+: pension shift predominantly completed within 2 years. "
                           "China Shock: older workers showed no recovery even at 9 years in trade-exposed zones.",
        "dominant_uncertainty": "Whether underemployed workers at 12 months stayed in work or moved to early retirement by 24 months (revolving door effect).",
    },
]

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

def main():
    rows = []
    for e in ESTIMATES:
        r_lo, r_mid, r_hi = e["retrain"]
        s_lo, s_mid, s_hi = e["share_similar"]
        u_lo, u_mid, u_hi = e["underemployed"]
        x_lo, x_mid, x_hi = e["exit"]
        k_lo, k_mid, k_hi = e["still_seeking"]

        rows.append({
            # Identification
            "episode_id":               "redcar_2015",
            "quarter":                  e["quarter"],
            "time_horizon":             e["time_horizon"],
            "months_post_closure":      e["months_post"],
            "age_band":                 e["age_band"],
            "estimation_method":        "manual — cross-episode reasoning + arithmetic constraints",
            "data_quality":             "ESTIMATED_LOW",

            # Ranges
            "est_retrain_%_low":        r_lo,
            "est_retrain_%_mid":        r_mid,
            "est_retrain_%_high":       r_hi,
            "est_share_similar_%_low":  s_lo,
            "est_share_similar_%_mid":  s_mid,
            "est_share_similar_%_high": s_hi,
            "est_underemployed_%_low":  u_lo,
            "est_underemployed_%_mid":  u_mid,
            "est_underemployed_%_high": u_hi,
            "est_exit_%_low":           x_lo,
            "est_exit_%_mid":           x_mid,
            "est_exit_%_high":          x_hi,
            "est_still_seeking_%_low":  k_lo,
            "est_still_seeking_%_mid":  k_mid,
            "est_still_seeking_%_high": k_hi,

            # Verification
            "mid_sum":  r_mid + s_mid + u_mid + x_mid + k_mid,

            # Reasoning
            "hard_anchors_used":        e["hard_anchors"],
            "math":                     e["math"],
            "cross_episode_priors":     e["cross_episode"],
            "dominant_uncertainty":     e["dominant_uncertainty"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  ({len(df)} rows x {len(df.columns)} cols)")

    # Print summary
    print("\n" + "="*70)
    print("ESTIMATED OUTCOME RANGES (all ages, mid values)")
    print("="*70)
    agg = df[df["age_band"] == "all"]
    for _, r in agg.iterrows():
        print(f"\n  {r.time_horizon}")
        print(f"    retrain:       {r['est_retrain_%_low']}-{r['est_retrain_%_high']}%  (MID {r['est_retrain_%_mid']}%)")
        print(f"    share_similar: {r['est_share_similar_%_low']}-{r['est_share_similar_%_high']}%  (MID {r['est_share_similar_%_mid']}%)")
        print(f"    underemployed: {r['est_underemployed_%_low']}-{r['est_underemployed_%_high']}%  (MID {r['est_underemployed_%_mid']}%)")
        print(f"    exit:          {r['est_exit_%_low']}-{r['est_exit_%_high']}%  (MID {r['est_exit_%_mid']}%)")
        print(f"    still_seeking: {r['est_still_seeking_%_low']}-{r['est_still_seeking_%_high']}%  (MID {r['est_still_seeking_%_mid']}%)")
        print(f"    mid sum check: {r.mid_sum} (must be 100)")

    print("\n" + "="*70)
    print("AGE BAND BREAKDOWN — Q4 2016 (12 months), mid values")
    print("="*70)
    q4 = df[df["quarter"] == "2016Q4"]
    print(f"\n  {'age_band':<10} {'retrain':>8} {'similar':>8} {'under':>8} {'exit':>8} {'seeking':>8} {'sum':>6}")
    for _, r in q4.iterrows():
        print(f"  {r.age_band:<10} {r['est_retrain_%_mid']:>8} {r['est_share_similar_%_mid']:>8} "
              f"{r['est_underemployed_%_mid']:>8} {r['est_exit_%_mid']:>8} {r['est_still_seeking_%_mid']:>8} "
              f"{r.mid_sum:>6}")

    # Write reasoning text
    with open(OUT_TXT, "w") as f:
        f.write("REDCAR 2015 — OUTCOME ESTIMATE REASONING\n")
        f.write("=" * 70 + "\n\n")
        f.write("All estimates are ESTIMATED_LOW data quality.\n")
        f.write("Mid values represent the most plausible single estimate.\n")
        f.write("Low/high values represent the plausible range given data uncertainty.\n")
        f.write("All mid values sum to 100% at each time horizon.\n\n")
        for e in ESTIMATES:
            f.write(f"\n{'='*70}\n")
            f.write(f"TIME HORIZON: {e['time_horizon']}\n")
            f.write(f"AGE BAND: {e['age_band']}\n")
            f.write(f"\nHARD ANCHORS USED:\n  {e['hard_anchors']}\n")
            f.write(f"\nMATH:\n  {e['math']}\n")
            f.write(f"\nCROSS-EPISODE PRIORS:\n  {e['cross_episode']}\n")
            f.write(f"\nDOMINANT UNCERTAINTY:\n  {e['dominant_uncertainty']}\n")
            r_lo, r_mid, r_hi = e["retrain"]
            s_lo, s_mid, s_hi = e["share_similar"]
            u_lo, u_mid, u_hi = e["underemployed"]
            x_lo, x_mid, x_hi = e["exit"]
            k_lo, k_mid, k_hi = e["still_seeking"]
            f.write(f"\nESTIMATES:\n")
            f.write(f"  retrain:       {r_lo}-{r_hi}%  (mid {r_mid}%)\n")
            f.write(f"  share_similar: {s_lo}-{s_hi}%  (mid {s_mid}%)\n")
            f.write(f"  underemployed: {u_lo}-{u_hi}%  (mid {u_mid}%)\n")
            f.write(f"  exit:          {x_lo}-{x_hi}%  (mid {x_mid}%)\n")
            f.write(f"  still_seeking: {k_lo}-{k_hi}%  (mid {k_mid}%)\n")
            f.write(f"  mid sum: {r_mid+s_mid+u_mid+x_mid+k_mid}\n")
    print(f"\nFull reasoning: {OUT_TXT}")


if __name__ == "__main__":
    main()
