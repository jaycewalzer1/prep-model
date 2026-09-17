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

## The reinforcement learning environment

`prep_model/env.py` wraps the same simulation as a sequential decision problem.
It is deliberately *behind* the cost-effectiveness analysis rather than under it:
the CEA does not import it, and nothing in the report depends on it.

```python
from prep_model.config import load_config
from prep_model.params import Registry
from prep_model.env import PrepEnv, compare, constant_policy, seed_bank

cfg = load_config("config/base.yaml")
params = Registry.load().resolve(cfg.price_year, overrides=cfg.overrides)
env = PrepEnv(cfg, params, arm_id="C", wtp=100_000.0)

train, held = seed_bank(cfg.seed, 32, held_out=16)
serve_highest_exposure_first = constant_policy((1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
print(compare(env, serve_highest_exposure_first, seeds=held))
```

A step is one week. The **action** is six numbers: the share of contact capacity
reserved for scheduled injection visits, and five weights over the features a
priority score may be built from — exposure group, sleeping outside, sharing
equipment, how overdue a visit is, and whether one was already missed. Every one
of those is something an outreach worker could know at the door. The latent
engagement propensity that actually drives retention in the model is *not*
available to the policy, because a policy that used it could not be run.

The **observation** is thirteen numbers a programme could genuinely see: its own
throughput and queue, coverage and lapse rates, the exposure mix of the people it
is in contact with. True prevalence and incidence are withheld.

The **reward** is the increment in net monetary benefit, `wtp * ΔQALYs − Δcost`,
per 1000 population, terminal value included in the final step. Summed over an
episode it equals exactly what the evaluation would report for that arm, which a
test asserts. Because the environment is seeded by counter-based RNG, two
policies run on the same seed share every draw they do not change, so the
difference in returns *is* the incremental net benefit — no control arm has to be
simulated to get it.

Only one thing about the world can be changed, and only where the model says a
policy could: the rationing of finite weekly capacity, via `DeliveryPlan` in
`delivery.py`. `DeliveryPlan()` with its defaults is the model's own hard-coded
policy, so the default action reproduces `simulate.run_arm` bit for bit. When
capacity is not binding, no action changes anything — also a test.

Three caveats, which are the reason this is a side door and not the front one:

1. **The noise floor is high.** Replicate standard deviation of infections
   averted is around a fifth of the mean. Any policy gradient computed from
   unpaired episodes is mostly estimating the seed. Use `compare`, which pairs on
   common random numbers; `test_pairing_removes_most_of_the_noise` requires the
   paired standard error to be under half the unpaired one, and in practice it is
   far smaller than that.
2. **The model is under-determined, and an optimizer will find that out.** Most
   parameters are transported or hypothetical. A policy tuned against them is
   tuned against assumptions, not against the world. `seed_bank(..., held_out=n)`
   exists so that a claimed improvement can at least be checked on seeds it was
   not fitted to; it cannot check it on parameters it was not fitted to.
3. **It answers a question nobody asked.** The commissioned question is whether
   the programme is worth funding, and that is what `run` and `analyze` report.
   Which of two people in a queue to serve first is a different question, worth
   asking only once the first one has been answered.

## Layout

```
config/     base.yaml, dev.yaml
data/       parameters.csv, sources.csv, price_index.csv
prep_model/ the model, the four commands, and env.py
tests/      test_invariants.py, test_machinery.py, test_env.py
outputs/    written by run and analyze; not checked in
```
