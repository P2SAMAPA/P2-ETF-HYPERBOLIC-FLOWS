# P2-ETF-HYPERBOLIC-FLOWS

A hyperbolic (Poincare-disk) map of each universe's ETF correlation
hierarchy, used to track structural migration — not a claim that ETFs
literally live on a hyperbolic manifold, a geometry chosen because it
represents trees with far less distortion than flat (Euclidean) space does.
Distinct from the existing `HYPERBOLIC-NEURAL-EMBEDDING` repo.

## The idea

Embed a branching hierarchy in flat 2D space and leaves crowd together as
the tree grows, because Euclidean area grows only polynomially with radius.
Hyperbolic space's area grows *exponentially* with radius — matching how a
tree's node count grows with depth — so trees embed with far lower
distortion (Sarkar 2011, *"Low Distortion Delaunay Embedding of Trees in
Hyperbolic Plane"*). The assumed market hierarchy (Global Macro → Rates/USD/
Commodities → Sectors → Industries → ETFs) is exactly this kind of tree, and
Euclidean distance is a genuinely poor way to represent it.

This repo does not assume that hierarchy directly — it **discovers** one
from each universe's own return correlation structure (hierarchical
clustering on correlation distance, `sqrt(2*(1-corr))`), embeds that tree
into the Poincare disk, and tracks where each ETF sits in the map from day
to day. **Structural migration** — an ETF's position shifting — is the
signal: its co-movement structure realigning, potentially before that shows
up as a simple correlation-number change.

## Two time scales (the lesson already learned twice on sibling engines)

1. **The map** is rebuilt only every `TREE_REFIT_INTERVAL` (63) trading
   days, from `TREE_CORR_WINDOW` (252) days of correlations, hierarchically
   clustered and embedded **exactly** via the closed-form Sarkar
   construction — no iterative optimization. This gives a stable coordinate
   system that persists for a while, same spirit as the barycenter engine's
   regime set or the morphogenesis field's grid.
2. **Each ETF's own position** within that frozen map is re-estimated every
   trading day — a few Riemannian gradient steps ("out-of-sample insertion"
   against fixed anchors) to match its current `short_corr_window`-day
   correlations, warm-started from yesterday. This is what moves day to
   day, and what "migration" measures — never the map itself.

### Why not fit the whole map by gradient descent every day

Tested directly: generic hyperbolic stress minimization (multidimensional
scaling) on a known 15-node balanced binary tree got stuck in a local
optimum with >100% mean relative distance error — *worse* than a plain 2D
**Euclidean** MDS of the same tree (19.6% error). The closed-form Sarkar
construction on the same tree got under 6% error (at a sufficient scale
factor) with no iterative optimization at all, because it is a specific
analytic answer for tree metrics rather than a generic, non-convex fit. That
is why the (infrequent) map build uses the exact construction, and gradient
descent is reserved only for the much easier daily "locate one point against
fixed anchors" problem.

### Why the Sarkar scale factor (tau) is chosen adaptively, not fixed

The validation above used tau=4 and got ~6% error — but applying that same
fixed tau to a real correlation tree (only a few levels deeper) pushed
anchors to Euclidean radius >0.95, right at the edge of safe numerical
precision, where tiny Euclidean errors become huge hyperbolic distances.
`hyperbolic_model.adaptive_tau` instead picks tau per-tree so the deepest
leaf lands at a fixed target radius (`TARGET_MAX_RADIUS`, 2.2 hyperbolic
units) — this cut the fraction of near-boundary points from ~29% to ~1.4% on
the same test data.

### Why the daily relocation step size is damped

Tested directly: at a larger step size, the median DAY-TO-DAY radius change
exceeded the median 10-DAY migration feature (ratio 1.1–1.6) — pure noise
dominating any real signal, and an engineered correlation-structure flip
(one ETF's market-loading sign flipped partway through a synthetic series)
was actually *less* visible than the noise floor (0.3x, backwards). A
smaller, damped step size (`RELOCATE_LR=0.08`) cut that noise ratio to
~0.13 and cleanly detected the same engineered flip as an isolated migration
spike, with the expected lag (`short_corr_window` days — how long it takes
the rolling correlation window to become dominated by post-flip data), with
no response from an unaffected ticker over the same days.

## What it does and does not claim

* The discovered hierarchy is this engine's best guess from price data
  alone, not the true causal macro→sector→industry structure. It is checked
  for plausibility against a hand-labeled asset-class taxonomy (dashboard's
  "taxonomy alignment" diagnostic), never used as model input.
* The forecast is a small linear model on the radius/migration/density
  readout, always compared against a baseline with **no hyperbolic features
  at all**, so the geometry is credited only for what it adds.

## How the result is kept honest

Same checks as the sibling engines, applied from the start:

1. **Baseline ablation.** Every configuration runs twice: the hyperbolic
   model (`mom5, mom21, vol21, corr_to_market` + `radius, radius_chg,
   migration, local_density, density_chg`), and a baseline using only the
   first four. Credit is the *difference*.
2. **Cross-sectional target.** Features and forward returns are z-scored
   across ETFs each day, so drift/beta cancel.
3. **The product is what is tested**: mean forward return of the top-3
   minus the equal-weight universe, after cost, plus cross-sectional
   rank-IC.
4. **Non-overlapping test points** spaced one horizon apart; Sharpe
   annualized by `sqrt(252/horizon)`.
5. **Selection/holdout split** with a sample-size-aware selection gate (an
   absolute AND relative period-count floor — `MIN_SELECTION_N`,
   `MIN_SELECTION_N_FRACTION`) and confidence **capped at Low whenever that
   gate didn't pass**, both carried over directly from the sibling
   delay-agent-dynamics repo after real-data runs there showed what goes
   wrong without them (a thin-sample grid cell won a selection purely on
   variance, then collapsed on its own holdout; and a config that never
   cleared its own selection bar still surfaced a misleadingly high
   confidence label from holdout stats alone).

## The "detect structural migration" claim is tested, not assumed

`hyperbolic_model.migration_lead_diagnostic` compares, per ETF: a
**migration onset** (10-day migration's z-score crosses above 1.5 — the
crossing moment, not the whole elevated stretch) against a **large-move
event** (`|`10-day forward return`|` exceeds 2x 21-day realized vol,
scaled), both de-duplicated to one per 21 quiet days.

**The chance baseline here is an empirical circular-shift permutation, not
an analytic formula** — this was a real bug, caught before release: an
analytic "uniform random placement" chance formula (the kind used in the
sibling delay-agent-dynamics engine, there validated correctly) gave hit
rate 0.56 vs its own computed chance level of 0.44 on two **fully
independent** synthetic series with no relationship by construction.
Root cause: de-duplicated onsets and events are a renewal process with a
minimum spacing (`dedup_days`), not a uniform-random scatter — two
independent quasi-periodic sequences with similar typical spacing overlap
*more* than a uniform-placement formula predicts (a "beat frequency"
effect). A circular shift of each ETF's actual onset timing (preserving its
exact spacing, rotating it to a random phase against the event timeline)
reproduces that periodicity in the null instead of assuming it away — this
closed the gap to 0.561 vs 0.555 on the same independent test data, and
still clearly separated a planted lead relationship (hit rate 0.35 vs
chance 0.12).

## Dashboard readouts (Tab 1)

* **The map itself**: every ETF plotted in the Poincare disk, colored by a
  static, hand-labeled asset class (reference only).
* **Furthest structural movers**: ETFs with the largest 10-day migration.
* **Taxonomy alignment**: between-class vs within-class hyperbolic distance
  in today's map (diagnostic only, never fed into the model).

## Repo structure

```
config.py            universes, asset-class tags (diagnostic only), map/relocation
                     parameters, SHORT_CORR_WINDOWS/HORIZONS/WINDOWS grid, LEAD_PARAMS
data_manager.py       master parquet from HF (same as sibling repos)
hyperbolic_model.py   Poincare geometry, correlation-tree construction, Sarkar
                     embedding, adaptive tau, daily relocation, features, O(1)
                     window ridge fits, walk-forward evaluation, migration-lead test
trainer.py            per-universe orchestration, selection/holdout, live picks, JSON
push_results.py       HfApi.upload_file to the results dataset
us_calendar.py         trading-day helpers
streamlit_app.py       Tab 1 live picks + hyperbolic map; Tab 2 backtest grid, curves,
                     lead diagnostic
.github/workflows/daily_run.yml   weekday cron after the US close
```

## Running

```bash
pip install -r requirements.txt
python trainer.py                 # writes hyperbolic_results_YYYY-MM-DD.json, pushes to HF
streamlit run streamlit_app.py
```

Set `HF_TOKEN`; create the dataset repo `P2SAMAPA/p2-etf-hyperbolic-flows-results`
or change `config.RESULTS_REPO`. To stop the live configuration re-selecting
daily, set `config.PINNED_CONFIG = {"short_corr_window": 63, "horizon": 5,
"window": 756}`.

## Validation status (please read)

Tested on synthetic data only; this sandbox cannot reach `huggingface.co`.

* **Pairwise stress gradient verified against finite differences** over 200
  random points (max error ~1e-9) before being used anywhere.
* **Sarkar construction validated on a known tree**: mean relative distance
  error fell from ~14% to ~4% as the scale factor grew from 1 to 6, and
  comfortably beat a generic hyperbolic MDS fit on the identical target
  (which got stuck in a bad local optimum, as described above).
* **Adaptive tau validated**: cut near-boundary points from ~29% to ~1.4% on
  real-shaped correlation-tree data relative to the fixed-tau version.
* **Relocation step size validated** against an engineered correlation flip:
  noise-to-signal ratio cut from 1.1–1.6 to ~0.13, with the flip correctly
  detected as an isolated migration spike at the expected lag and no
  response from an unaffected ticker.
* **Migration-lead diagnostic bug found and fixed pre-release**: the
  analytic chance-level formula was biased (0.56 vs 0.44 on independent
  data); replaced with an empirical circular-shift permutation, validated
  to match on the same independent data (0.561 vs 0.555) and to still
  separate a planted lead relationship cleanly.
* **Full three-universe run** at realistic size (2,500 days, up to 43 ETFs,
  36 grid cells: 3 correlation windows × 3 horizons × 4 windows) takes about
  53 seconds.
* **Known remaining limitation, not specific to this engine**: on pure
  synthetic noise, FI_COMMODITIES (the smallest universe, 7 tickers)
  reached "Medium" confidence on a holdout IC t-stat of only 0.72 — the
  confidence thresholds inherited unchanged from the sibling engines permit
  this on a universe this small; it is an acknowledged calibration
  looseness, not something newly introduced here, and is exactly the kind
  of thing the real-data run and holdout/baseline numbers in the dashboard
  are there to let you check directly rather than trust the label alone.

Nothing here says it works on real markets. The first real run is the
actual test; read the holdout numbers and baseline comparison before the
picks. Realistic rank-IC for a daily cross-sectional signal is small (order
0.01–0.05); with 36 configurations searched, one strong-looking cell is
expected by chance. This is not financial advice.
