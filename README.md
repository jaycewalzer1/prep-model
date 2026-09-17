# Injectable PrEP for people experiencing homelessness

A transmission microsimulation and economic evaluation of long-acting injectable
PrEP (lenacapavir, with cabotegravir as a comparator) delivered to adults
experiencing homelessness in one catchment.

**Nothing here is an Atlanta measurement.** Every calibration target and most
delivery inputs are flagged `hypothetical`: defensible placeholders with stated
ranges, standing in for local data that would need a data-use agreement. The
model is a structure for a decision, and the report says so on its first page.

## What question it answers

Three claims are kept apart, because they are different claims and only the
first two are usually earned:

1. **Health benefit** — infections averted and QALYs gained.
2. **Cost-effectiveness** — cost per QALY against a stated willingness to pay.
3. **Cost saving** — total spending falls. Ordinary shelter expenditure is a
   background service present in both arms, and the HIV-attributable housing
   saving is zero unless somebody supplies a sourced value for it.

Costs live in separate category ledgers and a *perspective* is a set of
categories, so "can the programme operator fund delivery?" and "does total
public spending fall?" are answered from the same run without one being
silently substituted for the other.

## Install and run

```sh
uv venv && uv pip install -e '.[dev]'

python -m prep_model.validate  --config config/dev.yaml --smoke
python -m prep_model.calibrate --config config/dev.yaml
python -m prep_model.run       --config config/dev.yaml --calibration outputs/dev/calibration.json
python -m prep_model.analyze   --run-dir outputs/dev
```

`config/dev.yaml` is 5,000 people over 5 years and finishes in a few minutes;
`config/base.yaml` is 20,000 over 10 years. The report lands at
`outputs/<run_name>/report.md` with its figures beside it.

The four commands are separate on purpose:

- **validate** audits the registry and smoke-runs every arm. It does not model.
- **calibrate** fits usual care only, keeps an ensemble rather than a point, and
  scores held-out targets that were not fitted.
- **run** does the policy experiments and writes JSON. Nothing is formatted here.
- **analyze** re-simulates the base case from the recorded seed and **refuses to
  write a report** if it disagrees with what `run` wrote, then builds the tables
  and figures. A long run is never repeated to change a table.

## The five arms

| id | arm | delivery | product |
|---|---|---|---|
| A | usual care | — | — |
| B | clinic lenacapavir | existing clinics | lenacapavir |
| C | low-barrier lenacapavir | walk-in, mobile, shelter-based, with navigation | lenacapavir |
| D | low-barrier cabotegravir | same outreach | cabotegravir |
| E | matched contact control | same outreach, no injectable uptake | — |

Arm E separates the effect of the *drug* from the effect of *showing up*: the
same outreach, testing and ART linkage without the injection. It is a modelling
counterfactual, not a proposal to withhold care.

## How the model is put together

- `rng.py` — counter-based draws keyed on `(seed, stream, step, person)`. No
  stream state, so two arms that should be identical are bit-identical.
- `population.py` — a pre-allocated cohort with a deterministic entry schedule.
  Arms cannot diverge in who exists or when.
- `transmission.py` — group-based mixing, infectiousness-weighted prevalence
  recomputed every step. Freezing it gives the direct-effects-only sensitivity.
- `natural_history.py` — stages, cascade, and death and out-migration as
  coherent competing hazards: one draw for whether a person leaves, a second to
  split the cause, so nobody both dies and migrates in one interval.
- `delivery.py` — the programme as a delivery system: capacity, refusal, failure
  to initiate after accepting, screening, late visits, discontinuation,
  re-engagement, and cost for contacts that produce no injection.
- `prep.py` — protection that wanes linearly past the due date rather than
  falling off a cliff, and is route-specific.
- `economics.py` — category ledgers, QALYs, and a terminal value that covers
  only time strictly beyond the horizon, at the same state-specific costs the
  simulation used.
- `experiments.py` — ICERs with simple *and* extended dominance removed, net
  benefit, one- and two-way sweeps, the scenario grid, structural sensitivity,
  and a PSA that draws calibrated parameters from the accepted ensemble.

Three things the code is built to refuse:

- **No efficacy cliff.** Protection does not drop to zero the day a dose is due.
- **No perfect protection.** Efficacy against the equipment-sharing route is set
  to zero as a conservative structural assumption and relaxed only in
  sensitivity; it is never silently borrowed from the sexual-route trial result.
- **No double counting.** A lifetime HIV cost is not added on top of simulated
  care, and longer survival keeps accruing non-HIV care and housing costs rather
  than being treated as free.

## Parameters

`data/parameters.csv` is the single source of every number. Fifteen columns,
including the distribution, an evidence grade A–D, and an assumption flag from
`observed_local | transported | calibrated | hypothetical`. An unknown parameter
name raises rather than resolving to zero, an override must name a real row, and
the registry records which rows a run actually read — so the report can list the
hypothetical inputs that were used and the rows that were not.

Most rows carry no source URL. That is the model's main evidential limitation
and the report states it as one.

## What the tests are for

`pytest` runs in about 25 seconds. The checks are the plan's validation list,
and they target failure modes that would change a decision:

- zero hazard gives zero infections; zero uptake reproduces usual care; zero
  efficacy leaves the testing effect intact;
- perfect sexual protection does **not** block the injection route;
- nobody is infected twice, prescribed prevention instead of treatment, or
  accrues life-years after death; the population reconciles;
- **two identical arms produce exactly zero incremental outcomes** — not "within
  stochastic error". If they differ at all, the random streams have slipped and
  every incremental result is contaminated by noise that looks like an effect;
- a shared background cost cancels in the difference; a higher price raises cost
  without touching the epidemic and lowers net benefit;
- the break-even price is *solved*, not searched: total cost is affine in the
  dose price, so the discounted dose count is the slope. The test re-runs the
  model at the returned price and requires the incremental cost to be zero;
- discounting matches its closed form, and the terminal value adds nothing that
  was already simulated.

One test is worth reading before trusting any single run.
`test_a_finer_step_and_a_larger_cohort_move_nothing_the_replicate_noise_does_not`
exists because halving the step changes every step index and is therefore a new
seed, not a more precise version of the same run. This epidemic feeds back
through a small, nine-times-infectious acute compartment, and the replicate
standard deviation of infections averted is roughly a fifth of its mean. A
single run is not a result. `stochastic_replicates` in the config is set low for
speed; raise it before quoting a number.

## Layout

```
config/     base.yaml, dev.yaml
data/       parameters.csv, sources.csv, price_index.csv
prep_model/ the model and the four commands
tests/      test_invariants.py, test_machinery.py
outputs/    written by run and analyze; not checked in
```
