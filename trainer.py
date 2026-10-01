"""
trainer.py  —  P2 HYPERBOLIC-FLOWS trainer

Builds a hyperbolic (Poincare-disk) embedding of each universe's ETF
correlation structure — a data-driven analogue of the Global Macro / Rates /
Sectors / Industries / ETFs hierarchy, discovered from recent correlations
rather than assumed — and asks whether an ETF's position in that map (its
distance from the map's root, how fast that position is moving, how tightly
clustered it is locally) helps rank ETFs by forward return relative to the
universe average, over and above a baseline with no hyperbolic features at
all. See hyperbolic_model.py for the theory and the validation that led to
each design choice (why the map is combinatorially constructed rather than
fit by gradient descent, why relocation runs at a damped step size, why tau
is chosen adaptively per tree).

Design choices (same shape as the sibling engines, for the same reasons):
  * Cross-sectional target: features and forward return are z-scored across
    ETFs each day, so market drift/beta cancel.
  * Baseline ablation: every configuration is evaluated for the hyperbolic
    model AND a no-hyperbolic baseline on the same dates; only the
    difference is credit.
  * Non-overlapping test points spaced `horizon` days apart; Sharpe
    annualized by sqrt(252/horizon); cost charged on the fraction of the
    top-N replaced.
  * Selection/holdout split, with a sample-size-aware selection gate (an
    absolute AND relative period-count floor) and confidence capped at Low
    whenever that gate didn't pass — both added after real-data runs on the
    sibling delay-agent-dynamics engine showed what goes wrong without them.
  * The "detect structural migration" claim is tested explicitly (a
    migration-leads-large-moves diagnostic), not assumed.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import hyperbolic_model as hm
from data_manager import load_master_data, validate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d: Dict) -> Dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in d.items()}


def confidence_label(hold: Dict, gate_passed: bool = True) -> str:
    """
    Graded on the HOLDOUT of the selected configuration, and capped at Low
    whenever `gate_passed` is False — see config.MIN_SELECTION_N_FRACTION's
    comment for the real-data case (on the sibling delay-agent-dynamics
    engine) that motivated this cap: good holdout numbers from a
    configuration that never cleared its own selection-segment eligibility
    bar are not good evidence of anything systematic.
      High   : holdout rank-IC t >= IC_T_HIGH, net top-N spread > 0, and beats
               the baseline on both IC and net spread.
      Medium : IC > 0, net spread > 0, and beats the baseline on both.
      Low    : anything else, fewer than MIN_HOLDOUT_PERIODS test periods, or
               gate_passed is False.
    """
    if not gate_passed:
        return "Low"
    if hold.get("n", 0) < config.MIN_HOLDOUT_PERIODS:
        return "Low"
    ic, sp = hold["ic_mean"], hold["net_spread_mean"]
    d_ic, d_sp = hold["d_ic_mean"], hold["d_spread_mean"]
    beats_baseline = d_ic > 0 and d_sp > 0
    if hold["ic_t"] >= config.IC_T_HIGH and sp > 0 and beats_baseline:
        return "High"
    if ic > 0 and sp > 0 and beats_baseline:
        return "Medium"
    return "Low"


def select_config(grid: List[Dict], short_corr_window: Optional[int] = None):
    """
    Choose a configuration from the SELECTION segment only: eligible if
    rank-IC and net top-N spread are both positive there, AND the config's
    selection-segment period count clears both an absolute floor
    (MIN_SELECTION_N) and a floor relative to the largest period count in
    this pool (MIN_SELECTION_N_FRACTION) — see config.py's comment for why
    the relative floor matters. Among eligible configs, rank-sum of the two
    t-stats. If nothing is eligible, fall back to the size-filtered pool and
    report gate_passed=False.
    """
    pool = [g for g in grid if (short_corr_window is None or g["short_corr_window"] == short_corr_window)
            and g["selection"].get("n", 0) >= config.MIN_SELECTION_N]
    if not pool:
        return None, False
    max_n = max(g["selection"]["n"] for g in pool)
    pool = [g for g in pool if g["selection"]["n"] >= config.MIN_SELECTION_N_FRACTION * max_n]
    if not pool:
        return None, False
    elig = [g for g in pool if g["selection"]["ic_mean"] > 0 and g["selection"]["net_spread_mean"] > 0]
    gate = bool(elig)
    cand = elig if gate else pool
    ic_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["ic_t"]))}
    sp_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["net_spread_t"]))}
    best = max(cand, key=lambda g: (ic_order[id(g)] + sp_order[id(g)], g["selection"]["ic_t"]))
    return best, gate


def make_picks(scores: np.ndarray, tickers: List[str], dispersion: float, horizon: int,
               confidence: str, top_n: int) -> List[Dict]:
    order = np.argsort(-scores)[:top_n]
    return [{
        "ticker": tickers[j],
        "expected_return": round(float(scores[j] * dispersion * 100), 3),
        "score_z": round(float(scores[j]), 4),
        "horizon_days": int(horizon),
        "confidence": confidence,
    } for j in order]


def pattern_diagnostics(hyp: Dict, tickers: List[str], dates: List[str], returns: np.ndarray) -> Dict:
    """Today's snapshot + taxonomy-alignment check + a decimated time series, for the dashboard."""
    n = len(dates)
    pos, radius, migration, density = hyp["positions"], hyp["radius"], hyp["migration"], hyp["local_density"]
    last = n - 1
    out = {}
    valid_today = np.isfinite(pos[last]).all(axis=1)
    if valid_today.any():
        j = np.where(valid_today)[0]
        snapshot = [{"ticker": tickers[k], "x": _f(pos[last, k, 0]), "y": _f(pos[last, k, 1]),
                    "radius": _f(radius[last, k]), "migration": _f(migration[last, k]),
                    "local_density": _f(density[last, k]),
                    "asset_class": config.ASSET_CLASS_MAP.get(tickers[k])}
                   for k in j]
        out["snapshot"] = snapshot
        out["taxonomy_alignment"] = hm.taxonomy_alignment(pos[last][j], [tickers[k] for k in j], config.ASSET_CLASS_MAP)

    valid_hist = np.isfinite(radius).any(axis=1)
    idx = np.arange(0, n, config.SERIES_STEP)
    idx = idx[valid_hist[idx]]
    out["series"] = {
        "dates": [dates[i] for i in idx],
        "median_radius": [_f(np.nanmedian(radius[i])) if np.isfinite(radius[i]).any() else None for i in idx],
        "median_migration": [_f(np.nanmedian(migration[i])) if np.isfinite(migration[i]).any() else None for i in idx],
    }
    return out


def analyze_universe(name: str, tickers: List[str], prices_df: pd.DataFrame) -> Optional[Dict]:
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return None
    px_df = prices_df[available]
    px_df = px_df[~px_df.isna().any(axis=1)]
    if len(px_df) < 400:
        logger.warning(f"Not enough data for {name}")
        return None

    values = px_df.values
    returns = np.diff(np.log(values), axis=0)
    dates = [d.strftime("%Y-%m-%d") for d in px_df.index[1:]]
    n = len(dates)

    cfg = dict(ridge_lambda=config.RIDGE_LAMBDA, min_train_days=config.MIN_TRAIN_DAYS, cost_bps=config.TRADING_COST_BPS)
    fwd_by_h = {h: hm.compute_forward_returns(returns, h) for h in config.HORIZONS}
    yz_by_h = {h: hm.cs_zscore(fwd_by_h[h]) for h in config.HORIZONS}

    L = 21
    s1, s2 = hm.trailing_sum(returns, L), hm.trailing_sum(returns ** 2, L)
    vol21_raw = np.sqrt(np.maximum((s2 - s1 ** 2 / L) / (L - 1), 1e-10))

    grid, wf_store, live_ctx, patterns, hyp_by_scw = [], {}, {}, {}, {}
    for scw in config.SHORT_CORR_WINDOWS:
        logger.info(f"  [{name}] short_corr_window={scw}: building hyperbolic map series...")
        hyp = hm.build_hyperbolic_series(
            returns, config.TREE_CORR_WINDOW, config.TREE_REFIT_INTERVAL, scw, config.TARGET_MAX_RADIUS,
            config.RELOCATE_STEPS, config.RELOCATE_LR, config.MIGRATION_LOOKBACK, config.LOCAL_K,
            config.LINKAGE_METHOD,
        )
        hyp_by_scw[scw] = hyp
        Fz = hm.cs_zscore(hm.build_feature_array(returns, hyp))
        patterns[scw] = pattern_diagnostics(hyp, available, dates, returns)
        patterns[scw]["migration_lead"] = hm.migration_lead_diagnostic(hyp["migration"], returns, vol21_raw, dates, config.LEAD_PARAMS)

        for h in config.HORIZONS:
            cp = hm.build_cross_products(Fz, yz_by_h[h])
            for window in config.WINDOWS:
                wf = hm.walk_forward(Fz, cp, fwd_by_h[h], h, window, config.TOP_N, cfg)
                key = (scw, h, window)
                live_ctx[key] = (Fz, cp)
                if wf is None:
                    continue
                m = len(wf["rows"])
                split = int(m * config.SELECTION_FRACTION)
                entry = {
                    "short_corr_window": scw, "horizon": h, "window": window, "n_periods": m,
                    "selection": _clean(hm.summarize(wf, 0, split, h)),
                    "holdout": _clean(hm.summarize(wf, split, m, h)),
                    "full": _clean(hm.summarize(wf, 0, m, h)),
                }
                grid.append(entry)
                wf_store[key] = (wf, split)
        logger.info(f"  [{name}] short_corr_window={scw}: done ({len([g for g in grid if g['short_corr_window'] == scw])} configs)")

    if not grid:
        return None

    def live_for(g):
        Fz, cp = live_ctx[(g["short_corr_window"], g["horizon"], g["window"])]
        return hm.live_scores(Fz, cp, fwd_by_h[g["horizon"]], g["horizon"], g["window"], cfg)

    selected, gate = select_config(grid)
    pinned = False
    pin = config.PINNED_CONFIG
    if pin:
        match = [g for g in grid if g["short_corr_window"] == pin["short_corr_window"]
                and g["horizon"] == pin["horizon"] and g["window"] == pin["window"]]
        if match:
            selected, pinned = match[0], True
            gate = True
    if selected is None:
        return None

    ls = live_for(selected)
    conf = confidence_label(selected["holdout"], gate)
    picks = base_picks = []
    if ls is not None:
        picks = make_picks(ls["score_m"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)
        base_picks = make_picks(ls["score_b"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)

    scw_picks = {}
    for scw in patterns:
        g, sgate = select_config(grid, scw)
        if g is None:
            continue
        lst = live_for(g)
        sconf = confidence_label(g["holdout"], sgate)
        scw_picks[str(scw)] = {
            "config": {"short_corr_window": scw, "horizon": g["horizon"], "window": g["window"]},
            "gate_passed": sgate,
            "holdout": g["holdout"],
            "confidence": sconf,
            "picks": make_picks(lst["score_m"], available, lst["dispersion"], g["horizon"],
                                sconf, config.TOP_N) if lst is not None else [],
        }

    wf, split = wf_store[(selected["short_corr_window"], selected["horizon"], selected["window"])]
    curves = {
        "dates": [dates[i] for i in wf["rows"]],
        "hyp_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_m"])],
        "baseline_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_b"])],
        "split_index": int(split),
    }

    top = {
        "short_corr_window": selected["short_corr_window"], "horizon": selected["horizon"], "window": selected["window"],
        "gate_passed": gate, "pinned": pinned, "confidence": conf,
        "selection": selected["selection"], "holdout": selected["holdout"], "full": selected["full"],
        "n_periods": selected["n_periods"],
    }
    feature_weights = None
    if ls is not None:
        feature_weights = {nm: round(float(w), 5) for nm, w in zip(hm.FEATURE_NAMES, ls["beta_m"])}

    return {
        "tickers": available, "n_days": n, "last_date": dates[-1],
        "selected": top, "picks": picks, "baseline_picks": base_picks,
        "grid": grid, "scw_picks": scw_picks, "curves": curves,
        "patterns": {str(k): v for k, v in patterns.items()}, "feature_weights": feature_weights,
    }


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date, "algorithm": "HYPERBOLIC-FLOWS",
        "top_picks": {}, "selected": {}, "grid": {}, "scw_picks": {}, "baseline_picks": {},
        "curves": {}, "patterns": {}, "feature_weights": {}, "universe_meta": {},
        "settings": {"trading_cost_bps": config.TRADING_COST_BPS, "top_n": config.TOP_N,
                    "selection_fraction": config.SELECTION_FRACTION,
                    "tree_corr_window": config.TREE_CORR_WINDOW, "tree_refit_interval": config.TREE_REFIT_INTERVAL,
                    "target_max_radius": config.TARGET_MAX_RADIUS},
    }

    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df)
        if out is None:
            continue
        results["top_picks"][name] = out["picks"]
        results["selected"][name] = out["selected"]
        results["grid"][name] = out["grid"]
        results["scw_picks"][name] = out["scw_picks"]
        results["baseline_picks"][name] = out["baseline_picks"]
        results["curves"][name] = out["curves"]
        results["patterns"][name] = out["patterns"]
        results["feature_weights"][name] = out["feature_weights"]
        results["universe_meta"][name] = {"tickers": out["tickers"], "n_days": out["n_days"], "last_date": out["last_date"]}

        s = out["selected"]
        h = s["holdout"]
        logger.info(f"  ✅ Selected: short_corr_window={s['short_corr_window']} / {s['horizon']}d / window {s['window']} "
                    f"(selection gate {'passed' if s['gate_passed'] else 'NOT passed'}) | holdout n={h.get('n')} "
                    f"IC={h.get('ic_mean', 0):+.4f} (t={h.get('ic_t', 0):+.2f}) "
                    f"vs baseline IC={h.get('ic_base_mean', 0):+.4f} | net top-{config.TOP_N} excess "
                    f"{h.get('net_spread_mean', 0) * 1e4:+.1f}bps/period | confidence {s['confidence']}")
        for p in out["picks"]:
            logger.info(f"     {p['ticker']}: {p['expected_return']:+.3f}% excess over {p['horizon_days']}d ({p['confidence']})")

    output_path = f"hyperbolic_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")

    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
