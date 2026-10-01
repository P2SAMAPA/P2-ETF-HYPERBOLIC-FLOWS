"""
config.py  —  Configuration for P2 HYPERBOLIC-FLOWS
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-hyperbolic-flows-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

TOP_N = 3

# Hand-labeled, static asset-class tags — diagnostic only (dashboard coloring
# and the taxonomy-alignment check in hyperbolic_model.taxonomy_alignment),
# never fed into the model. These are just what each ETF IS (well-known,
# stable facts about the ticker), not a claim being tested.
ASSET_CLASS_MAP = {
    "TLT": "Fixed Income", "VCIT": "Fixed Income", "LQD": "Fixed Income", "HYG": "Fixed Income",
    "VNQ": "Real Estate", "XLRE": "Real Estate",
    "GLD": "Commodity", "SLV": "Commodity", "GDX": "Commodity", "XME": "Commodity", "URA": "Commodity",
    "SPY": "Equity-Broad", "QQQ": "Equity-Broad", "IWM": "Equity-Broad", "IWD": "Equity-Broad",
    "IWO": "Equity-Broad", "IWF": "Equity-Broad", "IWR": "Equity-Broad", "VO": "Equity-Broad",
    "VB": "Equity-Broad", "VIG": "Equity-Broad", "VEA": "Equity-Broad", "QUAL": "Equity-Broad",
    "SPYG": "Equity-Broad",
    "XLK": "Equity-Sector", "XLF": "Equity-Sector", "XLE": "Equity-Sector", "XLV": "Equity-Sector",
    "XLI": "Equity-Sector", "VGT": "Equity-Sector", "VDE": "Equity-Sector", "XLC": "Equity-Sector",
    "IBB": "Equity-Sector", "XLY": "Equity-Sector", "XLP": "Equity-Sector", "XLU": "Equity-Sector",
    "XSD": "Equity-Sector", "SOXX": "Equity-Sector", "SMH": "Equity-Sector", "XBI": "Equity-Sector",
    "XLB": "Equity-Sector", "VUG": "Equity-Sector", "VTV": "Equity-Sector",
}

# ---------------------------------------------------------------------------
# The hyperbolic map: built from a correlation-distance tree
# (sqrt(2*(1-corr))) over a rolling window, hierarchically clustered
# (average linkage), embedded exactly (no iterative optimization) into the
# Poincare disk via the Sarkar (2011) combinatorial construction. Re-built
# only every `tree_refit_interval` trading days from `tree_corr_window` days
# of history; between refits every ticker's own position is re-estimated
# daily (a few Riemannian gradient steps against the FROZEN map) to match
# its current `short_corr_window`-day correlations. See hyperbolic_model.py
# for why the map itself is never fit by gradient descent (it got stuck in a
# bad local optimum on a validation tree; the closed-form construction did
# not) and why the two time scales exist (the same why-not-refit-daily
# lesson as the sibling engines' persistence mechanisms).
# ---------------------------------------------------------------------------
TREE_CORR_WINDOW = 252
TREE_REFIT_INTERVAL = 63
# Target hyperbolic radius for the DEEPEST leaf at each map refit (adaptive_tau
# picks a per-tree scale factor to hit this, rather than a fixed scale factor
# — see hyperbolic_model.adaptive_tau's docstring for why: a fixed scale
# factor validated on one tree pushed anchors to Euclidean radius >0.95 (the
# edge of safe numerical precision) on a real, only slightly deeper
# correlation tree).
TARGET_MAX_RADIUS = 2.2
# Daily relocation step size/count: the target (today's short-window
# correlations) only changes slightly day to day, so a large step size here
# re-solves close to convergence every day instead of smoothly tracking —
# validated: lr=0.15 made the median 1-day radius change EXCEED the median
# 10-day migration (ratio 1.1-1.6, i.e. pure day-to-day noise swamping any
# real signal, and an engineered correlation-structure flip was actually
# LESS visible than the noise floor). lr=0.08 cut that ratio to ~0.13 and
# cleanly detected the same engineered flip as an isolated migration spike
# roughly `short_corr_window` days after the change (the expected lag for
# the rolling window to become dominated by post-change data), with no
# response from an unaffected ticker over the same days.
RELOCATE_STEPS = 8
RELOCATE_LR = 0.08
MIGRATION_LOOKBACK = 10
LOCAL_K = 5
LINKAGE_METHOD = "average"

# Grid axis: how fast a ticker's tracked position responds to changing
# correlations (the daily relocation target window). Analogous to
# local_window / trend_span in the sibling engines.
SHORT_CORR_WINDOWS = [21, 42, 63]

# ---------------------------------------------------------------------------
# Prediction: pooled ridge regression of the cross-sectionally z-scored
# forward H-day return on cross-sectionally z-scored features, refit at each
# evaluation date on the trailing WINDOW days. Two models share the same
# machinery:
#   baseline : mom5, mom21, vol21, corr_to_market — no hyperbolic geometry.
#   hyperbolic: baseline + radius, radius_chg, migration, local_density,
#              density_chg. Credited only for what it adds beyond baseline.
# ---------------------------------------------------------------------------
HORIZONS = [1, 5, 20]
WINDOWS = [252, 504, 756, 1008]
RIDGE_LAMBDA = 0.01
MIN_TRAIN_DAYS = 60

SELECTION_FRACTION = 0.70
MIN_HOLDOUT_PERIODS = 30
TRADING_COST_BPS = 15
IC_T_HIGH = 1.65

# Selection eligibility requires a configuration's selection-segment period
# count to clear BOTH an absolute floor and a floor relative to the largest
# period count in the pool — see the sibling delay-agent-dynamics repo's
# README for the real-data case (a thin-sample cell won a selection purely
# on variance, then collapsed on its own holdout) that motivated this.
MIN_SELECTION_N = 50
MIN_SELECTION_N_FRACTION = 0.15

# Optional: pin the live configuration instead of re-selecting daily, e.g.
# {"short_corr_window": 63, "horizon": 5, "window": 756}. None = select.
PINNED_CONFIG = None

# Structural-migration lead diagnostic: does a migration onset (an ETF's
# position jumping, z-scored against its own history) tend to sit before a
# large realized move? Both trigger definitions fixed here, not tuned.
LEAD_PARAMS = {
    "z_window": 252,
    "migration_z": 1.5,
    "event_window": 10,
    "event_vol_k": 2.0,
    "dedup_days": 21,
    "lead_window": 20,
}

SERIES_STEP = 3
