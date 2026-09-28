# Paper 3 — Does machine learning make IFRS 9 provisioning more procyclical?

Implementation brief. Written to be handed to a coding session that has local network
and disk; the design decisions here are the ones that determine whether the result is
defensible, so change them deliberately rather than incidentally.

---

## 1. The claim to be tested

IFRS 9 made provisions forward-looking. A PD model that reacts faster to incoming
information produces provisions that move earlier and further over a cycle. Machine
learning models react faster than through-the-cycle logits. So:

- **H1 (amplitude).** Peak-to-trough amplitude of aggregate ECL is larger for ML-based
  PD than for a TTC logit, holding LGD and EAD identical.
- **H2 (timing).** The ECL path turns earlier relative to realised charge-offs — test on
  the lead/lag at which cross-correlation with realised losses is maximised.
- **H3 (stage cliff).** A larger share of the ECL swing comes from stage migration rather
  than from PD movement within Stage 1, because faster-moving PDs cross the SICR
  threshold in bulk.
- **H4 (heterogeneity).** The amplitude gap is larger in segments where the ML model's
  accuracy advantage is larger — which, if it holds, is the paper's sharpest sentence:
  the better the model, the less stable the provision.

A null on H1 is publishable. Supervisors and standard-setters assume the opposite, so
"no material difference under realistic implementation choices" is a finding, not a
failure. Write the pre-analysis intention down before you look.

## 2. Data

**Primary: Freddie Mac Single-Family Loan-Level Dataset.** Origination file plus monthly
performance file, 1999 onward, quarterly releases. Free account and a click-through
licence; not scriptable without credentials, so the download is manual or
cookie-authenticated. Critically it carries *realised* loss on disposition — net sale
proceeds and expenses — so LGD is measured, not assumed.

**Replication: Fannie Mae Single-Family Loan Performance Data.** Same structure, separate
universe. Use it as an out-of-sample replication of the headline table, not as pooled
extra rows.

**Macro: ALFRED, not FRED.** Every macro covariate must be the vintage available at the
forecast date — unemployment, HPI, rates. Using current revised series hands the ML arm
a look-ahead advantage and invalidates H2 specifically, because revisions are exactly
where the timing information lives.

**Start with the published stratified sample** to build and debug the pipeline. Move to
the full files only once the ECL path reproduces on the sample. The full performance
files will not fit in 16 GB of RAM: DuckDB over partitioned parquet, partitioned by
origination vintage-quarter.

**Do not hardcode field names.** Layouts differ across releases; read the layout
document shipped with the download and generate the schema from it. Fields you will
need: loan sequence number, credit score, original LTV and CLTV, DTI, occupancy, channel,
property state and MSA, first payment date, current actual UPB, current loan delinquency
status, modification flag, zero-balance code and date, net sale proceeds, and the
expense fields. Post-2020 releases carry payment-deferral and forbearance indicators —
find them explicitly rather than assuming their absence.

## 3. Design

**Training discipline.** Expanding window, monthly or quarterly refit, using only loans
and outcomes observable at the refit date. No crisis-era information in a pre-crisis
model. This is the single thing a referee will check first.

**PD arms.** All arms predict the same object — 12-month and lifetime marginal default
probabilities at loan-month level — and differ only in functional form:

1. TTC logit: few covariates, no macro conditioning. The industry baseline.
2. PIT logit: same covariates plus vintage macro. Isolates macro conditioning from
   functional flexibility, so the ML-versus-logit gap is not confounded with it.
3. Gradient boosting (LightGBM or XGBoost), tuned on a pre-crisis validation split only.
4. Optional: a shallow net or a survival model (discrete-time hazard), if H4 needs it.

**Definition of default.** Pick one and state it: 180+ days delinquent, or zero-balance
code indicating a credit event. The choice moves the level of everything and the timing
of nothing, so it is a robustness row, not a design risk.

**LGD and EAD are held fixed across arms.** Estimate LGD once from realised dispositions
and use the same LGD and EAD in every arm. If LGD also varies by arm you can no longer
attribute the amplitude difference to the PD model, and the paper loses its identification.

**ECL construction.** Three-stage IFRS 9: Stage 1 carries 12-month ECL, Stage 2 lifetime
ECL on significant increase in credit risk, Stage 3 credit-impaired. Implement SICR as a
relative PD-increase trigger — doubling of lifetime PD since origination is the common
choice — with a 30-DPD backstop. Discount at the effective interest rate.
**The SICR threshold is the main free parameter in the whole paper**: report the headline
at one value and sweep it, because a referee will assume the result is threshold-driven
until shown otherwise.

**Outcomes.**
- Amplitude: peak-to-trough of aggregate ECL / UPB, 2005–2012.
- Timing: lead/lag of max cross-correlation between ECL and realised charge-offs.
- Decomposition: share of the ECL change attributable to stage migration versus
  within-stage PD movement (fix one, move the other, in both orders, and report both —
  the decomposition is not unique).
- Cliff: distribution of the per-loan ECL jump at Stage 1 → 2 transfer.

**Inference.** The arms share the same loans, so compare them paired: bootstrap over
time blocks (12 months) and compute the paired difference in amplitude within each
resample. Paired differences remove the loan-sampling noise that would otherwise swamp
the comparison. Report the difference and its interval, not two separate amplitudes with
overlapping intervals.

## 4. Known threats

- **Look-ahead through feature engineering.** Any feature built from a future-dated
  series — regional HPI, unemployment — must use the vintage available at that month.
  This is the most common way this class of paper dies.
- **COVID is not a second crisis.** Forbearance and fiscal transfers suppressed defaults.
  Either stop the sample in 2019 or treat 2020–21 as a separate model-breakdown exercise.
  Do not pool it with 2008.
- **Survivorship through prepayment.** Prepayment is competing risk, not censoring at
  random; loans that prepay are systematically different. A discrete-time competing-risks
  hazard handles it; a naive default classifier does not.
- **LGD selection.** Realised severities only exist for dispositions, which are selected.
  State it; a Heckman-style correction is defensible but not required if LGD is held
  constant across arms.
- **External validity.** GSE conforming mortgages are prime US residential. The result
  does not transfer to corporate or SME portfolios, and the paper should say so in the
  abstract rather than in the conclusion.

## 5. Where the two sessions split

Local coding session: downloads, schema generation, the DuckDB/parquet store, PD model
fitting, and the loan-month ECL computation. Output a small aggregate panel — date ×
arm × segment, with ECL, staged balances, realised charge-offs — a few megabytes at most.

Here: bring that aggregate panel back for the amplitude and timing statistics, the block
bootstrap, the decomposition, and the figures, so the paper's exhibits sit alongside the
literature map and the coupling evidence that motivates the introduction.

## 6. Positioning

The gap this fills is visible in the coupling matrix: the ML credit-scoring community
couples to IFRS 9 expected credit loss at 0.13 of the size-implied expectation — above
the 0.044 cross-family baseline, so contact exists, but far below its 0.53 link to
climate-credit. The procyclicality literature (78 works on the ECL/IFRS 9 probe,
including Abad and Suárez's recursive cyclical model) and the ML-in-IFRS-9 literature
(39 works, all estimation-accuracy focused) do not meet. Cite both sides and state that
the composition is the contribution.
