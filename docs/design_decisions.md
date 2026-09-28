# Design decisions

Changes to `brief.md`, in the order they were made. Each entry: decision, reason, date.

## Data
- **Main dataset: SEC EDGAR Reg AB II CMBS asset-level filings (EX-102), not Freddie Mac.** 375 conduit deals, 16,437 loans, Dec 2016 to Sep 2026.
  Reason: FRED/ALFRED has no real-time (vintage) house-price data before 22 May 2008 (checked 875 series), so a 2002–07 training window would use revised data. CMBS is public, needs no login, and all the relevant macro series have vintages after 2014. (2026-09-16)
- **Accepted limits:** 2020 delinquencies are partly forbearance that later cured; 173 exits with a loss, so LGD is estimated for the whole portfolio, not per segment; H4 has about 5 property-type segments with ≥100 defaults. (2026-09-17)

## Macro information set
- **A forecast for loan-month t uses the ALFRED vintage in force on the last day of month t−1.** Daily and weekly series are averaged to months inside that vintage, and changes (12-month, 3-month) are computed within the vintage. Series list: `docs/macro_series.csv` (24 national, 3 × 51 state).
  Reason: prevents look-ahead through revisions. Revisions are material for the CRE price indices (BIS: mean absolute revision 0.42 × the series' SD; Fed Z.1: 0.27), commercial rent PPI (0.33) and financial-sector employment (0.30), and negligible for rates, CPI and unemployment. (2026-09-17)
- The Fed Z.1 CRE price index has vintages only from 2019-03; before then, only the BIS CRE series is available. Estimation is not affected: 19 defaults before 2020. (2026-09-17)

## PD models
- **Direct horizon models instead of a monthly hazard.** A pooled multinomial model per arm predicts default and payoff in year k (k = 1–3) given survival to its start, from information at the forecast date. Years 4–10 reuse the year-3 dummy with the loan aged forward.
  Reason: a monthly hazard conditioned on current delinquency puts almost all risk on loans already 60 days late, and compounding it understated 12-month PD for current loans by 5–10×. (2026-09-17)
- **Inputs clipped to the training range.** Without clipping, the PIT logit trained on 2017–19 predicted 9,174 defaults in 2020 against 366 actual. (2026-09-17)
- **Macro reverts to its training-sample mean from year 3** (reasonable-and-supportable period = 2 years). (2026-09-17)
- **At-maturity defaults are flagged** (status 5 or ≤1 month to maturity at default): 161 of 1,102 defaults, 34–46% of defaults in 2022–24. Of the 161: 45 later paid off, 35 performing again after extension or modification, 15 ended in a loss, 66 still non-performing (median 22 months observed after default). (2026-09-17)
- **Default split into two competing outcomes: in-term and at-maturity** (plus payoff). Horizon models are 4-class; LGD will be estimated per default type and shared across arms. The headline definition is unchanged.
  Reason: maturity defaults are 35–43% of maturity windows but mostly resolve without loss, and their share of defaults swings from 0% to 46% across years. With one pooled PD and one LGD, that swing would show up as spurious procyclicality. (2026-09-17)
- No maturity defaults appear in training data before the 2023Q1 refit, so earlier refits assign zero maturity-default risk. Early refits (before 2022Q3) are unstable in both arms, because only year-1 windows exist and they are COVID-dominated. (2026-09-17)

- **ML arm: scikit-learn HistGradientBoostingClassifier** (histogram GBDT, same algorithm family as LightGBM; LightGBM needs a system-wide OpenMP install). Same features, clipping and macro reversion as the PIT logit. Hyperparameters were tuned once on data reported before the 2022Q3 refit, then frozen (`docs/gbm_params.json`). Validation gap vs the logit: log loss 0.079 vs 0.143, AUC 0.92 vs 0.81.
  Out of sample (2022–25) the ranking gap is small (all loans AUC 0.80–0.82 vs 0.79–0.82; performing-loan in-term AUC +0.00–0.04), and GBM 12-month PDs sit at about half the realised rate. (2026-09-17)

- **Headline uses raw (uncalibrated) model PDs; dampened recalibration is a robustness row.** Amplitude is therefore measured in relative terms (proportional change from the starting level), so a constant level difference between arms does not drive it. Recalibration adds 12-25% to quarter-to-quarter PD volatility in every arm, which would otherwise contaminate H1; that effect becomes a result in its own right. (2026-09-17)
- **Recalibration mechanics.** Hold-out rule: at each refit, fit the same specification on data cut 12 months earlier, then shift log-odds so its mean 12-month PD on the latest fully reported year of origins matches realised (by default type, capped at ±2). A first version that back-tested earlier models' out-of-sample PDs hit the cap in most quarters, because those early models had near-zero PDs going into COVID and no maturity class.
  Raw shifts are dampened: credibility shrinkage on effective events (window = 12 overlapping monthly origins, so n_eff = events / 12; K = 50), exponential smoothing across refits (alpha 0.35), cap ±1. This halves quarter-to-quarter shift variation and removes all cap hits.
  2022–26 comparison (mean absolute level error, pp / SD of quarterly % change in mean PD): no recalibration TTC 0.71/0.16, PIT 0.34/0.51, GBM 0.66/0.25; dampened recalibration TTC 0.71/0.22, PIT 0.42/0.63, GBM 0.48/0.28. Recalibration fixes GBM's level but adds volatility to every arm.
  Hold-out result, 2022–25, all loans (realised 0.60/1.12/1.76/1.64%): GBM 1.05/0.52/1.11/2.07, PIT logit 2.17/1.28/0.60/2.53, TTC 1.66/0.46/1.79/1.51, against raw GBM 0.34/0.49/0.68/0.97, PIT 0.55/1.00/0.57/1.61, TTC 2.08/1.34/2.44/2.11. Recalibration imports last year's default experience with a lag: it raises PDs in 2022 from the COVID year, and hits the cap for TTC in 2022–23 and for maturity defaults in 2023. It does not uniformly improve levels. (2026-09-17)

## ECL construction
- **LGD by default type, common to all arms** (`docs/lgd_table.csv`): in-term 11.6%, at-maturity 0.9% (resolved defaults, zero-loss cures included; 75% of in-term defaults cost the trust nothing). Bound with open defaults imputed from the latest appraisal: 14.7% / 5.4%. Property type adds nothing once cures are included (11.6% vs 11.7%). LGD on CRE price growth is flat (corr -0.03), so the cyclical-LGD robustness row will be uninformative. (2026-09-17)
- **SICR benchmark: the current model's score of the loan's origination-month characteristics over the same remaining life.** Benchmarking against the loan's first observed PD instead put 45-87% of exposure in Stage 2, because it compares a 2019-vintage model with a 2025-vintage one. With the fix, Stage 2 is 5-7% of exposure for the logits and 11-22% for GBM. The 30+ DPD backstop accounts for 0.6% of Stage 2. (2026-09-17)
- **Evaluation window 2022-07 to 2026-08**: earlier refits are unstable, and the current month is only partly filed. (2026-09-17)

## Evaluation window and timing series
- **Window rule, decided from training data alone:** start at the first refit whose training data holds at least 5 events of every outcome class per estimated parameter (49 parameters in the widest arm, so 245 events), applied identically to all arms. That gives 2023-10; EPV 10 gives 2024-04. The binding class is at-maturity default, which only accrues from 2022. This replaces the hand-picked 2022-07 start, which was chosen after seeing when the models settled. (2026-09-17)
- **H2 reference series:** newly 90+ delinquent balances and new special-servicer transfers, each as a share of exposure. Realised charge-offs are too lumpy (a few liquidations a month) and are kept only for comparison. (2026-09-17)

## Robustness (see docs/robustness_results.csv)
- **Under the pre-committed window (2023-10) the ML result is not a null: GBM provisions are significantly LESS procyclical than the PIT logit.** Paired difference -0.26 (CI -0.44 to -0.09), and the same sign at SICR 1.5x (-0.22) and 3x (-0.28), under the LGD bound (-0.40), under the SS default definition (-0.67, window 2024-01) and at EPV 10 (-0.53); with recalibrated PDs it is -0.18 (CI -0.35 to 0.00), borderline.
- On the earlier hand-picked windows (2022-01, 2022-07) the same difference is an insignificant null (-0.15, -0.16). The window governs which trough the amplitude is measured from, so both must be reported: the pre-committed rule is the headline, the earlier window a sensitivity.
- PIT-vs-TTC is positive and significant in every configuration under the rule (+0.85 baseline, +0.70 to +1.21 elsewhere) and reverses only on the pre-2022-07 windows where the TTC arm is still unwinding its 2021 instability.
- Alternative default definition (special-servicer transfer, VARIANT=_ss: 1,519 defaults instead of 1,102, arriving earlier; LGD 7.8% in-term / 1.1% at maturity): ML-vs-logit +0.10 (CI -0.21 to 0.39, still null, sign flipped); PIT-vs-TTC +0.20 (CI 0.05 to 0.41, still significant); migration share of the rise 46-56%.
- Seeds: HistGradientBoosting is deterministic with early stopping off, so the seed only matters through which stay-rows are sampled into training. Across seeds, amplitude is 1.61/1.64/1.65 (GBM) and 1.77/1.69 (PIT logit), i.e. +/-0.04-0.08, well inside the ML-vs-logit gap. (2026-09-17)

## ECL regimes
- **Add a CECL arm alongside IFRS 9.** Same lifetime PDs, LGD and EAD in both; CECL takes lifetime ECL on every loan with no staging.
  Reason: a 2×2 (logit/ML PD × IFRS 9/CECL) separates the effect of staging (the H3 cliff) from the effect of PD volatility. It also answers the US-loans referee question and costs almost nothing once lifetime PDs exist. Compare the swing relative to the starting level, because CECL starts from a higher base. (2026-09-17)

## Default definition
- **Headline: first month a loan is 90+ days delinquent (payment status 3 or 5) or leaves the pool with a loss (liquidation code 3 or 6, or realised loss > 0).** About 1,120 events.
  Reason: this is closest to the IFRS 9 / Basel 90-days-past-due backstop, and loss exits alone (173) are too few for PD models.
  Robustness rows: special-servicer transfer (broader and earlier, ~1,480); credit exit only. (2026-09-17)
