"""
hyperbolic_model.py  —  Hyperbolic embedding of ETF correlation hierarchy

Idea
----
Euclidean space represents a tree poorly: embed a branching hierarchy with N
leaves in flat 2D space and leaves crowd together as N grows, because
Euclidean volume grows only polynomially with radius. Hyperbolic space's
volume grows EXPONENTIALLY with radius, matching how a tree's node count
grows with depth — so trees embed into the hyperbolic plane with arbitrarily
low distortion (Sarkar 2011, "Low Distortion Delaunay Embedding of Trees in
Hyperbolic Plane"), while the same tree embedded in Euclidean space cannot
avoid distortion as it grows.

This engine treats each universe's ETFs as leaves of a TREE inferred from
their own return correlation structure (hierarchical clustering on
correlation distance — the empirical, data-driven analogue of "Global Macro
→ Rates/USD/Commodities → Sectors → Industries → ETFs"), embeds that tree
combinatorially into the Poincaré disk, and tracks each ETF's position in
that map from day to day. "Structural migration" — an ETF's hyperbolic
position shifting — is the signal: an ETF moving away from its usual place
in the correlation hierarchy indicates its co-movement structure is
realigning, potentially before that shows up in a simpler correlation number.

Two time scales, on purpose (the lesson learned from the sibling engines'
whipsaw problems)
--------------------------------------------------------------------------
1. The MAP is refit only every `tree_refit_interval` days, from a LONG
   correlation window (`tree_corr_window`) via hierarchical clustering, then
   embedded ONCE via the closed-form Sarkar construction (no iterative
   optimization — see sarkar_embed_tree). This gives a stable coordinate
   system that persists for a while, the same way the barycenter engine's
   regime set or the morphogenesis field's grid persists between refits.
2. Each ETF's OWN position within that frozen map is re-estimated every
   trading day (a few Riemannian gradient steps, warm-started from
   yesterday) to match its CURRENT short-window correlations to every other
   (fixed) ETF in the map. This is what actually moves day to day, and what
   "migration" measures — never the map itself moving.

Why NOT fit the whole map by gradient descent every day: a generic stress
minimization (multidimensional scaling in hyperbolic space) was tested
against a known 15-node balanced binary tree and got stuck in a poor local
optimum (>100% mean relative distance error, worse than a 2D EUCLIDEAN MDS
of the same tree). The closed-form Sarkar construction on the same tree gets
relative error under 6% (with a large-enough scale factor tau) with no
iterative optimization at all, because it's a specific analytic answer for
tree metrics rather than a generic (and non-convex) fit — see the repo notes
for the validation. Using it for the infrequent MAP refit, and reserving
gradient descent only for the much easier "locate one point against fixed
anchors" daily update, is why the pipeline uses both techniques rather than
either alone.

Honest scope
------------
* The "hierarchy" is discovered from recent correlations, not the true
  causal macro→sector→industry structure — it is this engine's best guess at
  that structure from price data alone, and is checked for plausibility
  against a hand-labeled asset-class taxonomy (see ASSET_CLASS_MAP in
  config.py) as a diagnostic, not used as ground truth anywhere in the model.
* The forecast is a small linear model on the migration/radius/density
  readout, always compared against a baseline with no hyperbolic features at
  all, so the geometry is credited only for what it adds.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from typing import Dict, List, Optional, Tuple

BASE_FEATURES = ["mom5", "mom21", "vol21", "corr_to_market"]
HYP_FEATURES = ["radius", "radius_chg", "migration", "local_density", "density_chg"]
FEATURE_NAMES = BASE_FEATURES + HYP_FEATURES
N_BASE = len(BASE_FEATURES)
N_FEAT = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------
def rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    lo = np.maximum(np.arange(a.shape[0]) - w + 1, 0)
    return c[1:] - c[lo]


def trailing_sum(a: np.ndarray, w: int) -> np.ndarray:
    out = rolling_sum(a, w)
    out[: w - 1] = np.nan
    return out


def compute_forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    end = n - horizon
    if end > 0:
        fwd[:end] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


# ---------------------------------------------------------------------------
# Poincare disk geometry (real 2D coordinates throughout)
# ---------------------------------------------------------------------------
def poincare_dist_matrix(Z: np.ndarray) -> np.ndarray:
    """Pairwise Poincare distances. Z: (n,2). Returns (n,n), 0 on the diagonal."""
    diff2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    norms = np.sum(Z ** 2, axis=1)
    B = np.maximum(1 - norms, 1e-12)
    denom = np.maximum(B[:, None] * B[None, :], 1e-12)
    gamma = np.maximum(1 + 2 * diff2 / denom, 1.0 + 1e-12)
    D = np.arccosh(gamma)
    np.fill_diagonal(D, 0.0)
    return D


def poincare_dist_to_fixed(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Distance from each of m moving points Z (m,2) to each of k fixed anchors A (k,2). Returns (m,k)."""
    diff2 = np.sum((Z[:, None, :] - A[None, :, :]) ** 2, axis=2)
    Bz = np.maximum(1 - np.sum(Z ** 2, axis=1), 1e-12)
    Ba = np.maximum(1 - np.sum(A ** 2, axis=1), 1e-12)
    denom = np.maximum(Bz[:, None] * Ba[None, :], 1e-12)
    gamma = np.maximum(1 + 2 * diff2 / denom, 1.0 + 1e-12)
    return np.arccosh(gamma)


def mobius_add(a: complex, b: complex) -> complex:
    """Poincare-disk Mobius addition: the isometry translating b by a (maps 0 -> a when b=0)."""
    return (a + b) / (1 + np.conj(a) * b)


def tree_depth_weight(children_of: Dict[int, List[Tuple[int, float]]], root: int) -> float:
    """Maximum root-to-leaf cumulative RAW edge weight (before any tau scaling)."""
    best = 0.0
    stack = [(root, 0.0)]
    while stack:
        node, depth = stack.pop()
        kids = children_of.get(node, [])
        if not kids:
            best = max(best, depth)
            continue
        for c, w in kids:
            stack.append((c, depth + w))
    return best


def adaptive_tau(children_of: Dict[int, List[Tuple[int, float]]], root: int, target_max_radius: float,
                 tau_min: float = 0.5, tau_max: float = 12.0) -> float:
    """
    Choose tau so the DEEPEST leaf lands at `target_max_radius` hyperbolic
    units from the origin, rather than using one fixed tau for every tree. A
    fixed tau that looks reasonable on one validation tree can be badly wrong
    on another: it pushed anchors to Euclidean radius >0.95 (right at the
    numerical edge, where tiny Euclidean errors become huge hyperbolic
    distances) on a real correlation tree only a few levels deeper than the
    one it was tuned on. Clipped to [tau_min, tau_max] so a degenerate
    (near-zero depth, or extremely deep/unbalanced) tree can't produce an
    unreasonable scale either.
    """
    depth = tree_depth_weight(children_of, root)
    if depth <= 1e-6:
        return tau_max
    return float(np.clip(target_max_radius / depth, tau_min, tau_max))


def sarkar_embed_tree(children_of: Dict[int, List[Tuple[int, float]]], root: int, n_nodes: int,
                      tau: float) -> np.ndarray:
    """
    Exact (non-iterative) combinatorial embedding of a weighted tree into the
    Poincare disk (Sarkar 2011): place the root at the origin, then place
    each node's children at hyperbolic distance tau*edge_weight from it,
    angularly spread over the arc not already used by the direction back to
    its own parent, recursively. `children_of[p]` is a list of (child, raw
    edge weight) pairs. Returns real (n_nodes, 2) positions.

    tau scales every edge weight uniformly; a larger tau gives lower
    distortion (validated: on a 15-node balanced binary tree, mean relative
    distance error fell from ~14% at tau=1 to ~4% at tau=6) at the cost of
    pushing points closer to the disk boundary, which is why RELOCATE_TAU in
    config.py is chosen as a middle ground and the boundary itself is never
    approached — see place_at's radius clipping below.
    """
    pos = {root: 0.0 + 0.0j}
    incoming_angle = {root: None}
    stack = [root]
    while stack:
        p = stack.pop()
        kids = children_of.get(p, [])
        if not kids:
            continue
        deg = len(kids) + (0 if incoming_angle[p] is None else 1)
        base = 0.0 if incoming_angle[p] is None else incoming_angle[p]
        offset = 0 if incoming_angle[p] is None else 1
        for k, (c, w) in enumerate(kids):
            theta = base + 2 * np.pi * (k + offset) / deg
            r_euclid = np.tanh(min(tau * w, 30.0) / 2.0)
            r_euclid = min(r_euclid, 0.999)
            x0 = r_euclid * np.exp(1j * theta)
            pos[c] = mobius_add(pos[p], x0)
            incoming_angle[c] = theta + np.pi
            stack.append(c)
    Z = np.zeros((n_nodes, 2))
    for node, z in pos.items():
        Z[node] = [z.real, z.imag]
    norms = np.linalg.norm(Z, axis=1)
    over = norms >= 0.999
    if over.any():
        Z[over] = Z[over] / norms[over, None] * 0.999
    return Z


def build_correlation_tree(returns_window: np.ndarray, linkage_method: str = "average") -> Tuple[Dict, int, int]:
    """
    Hierarchical clustering on correlation distance (sqrt(2*(1-corr))) of the
    trailing `returns_window` (rows=days, cols=tickers). Returns
    (children_of, root, n_total_nodes) for sarkar_embed_tree, where leaves
    0..T-1 ARE the tickers (in column order) and internal nodes are
    T..2T-2 (root = 2T-2, scipy's convention). Edge weight from a node to
    its parent = parent merge height minus the node's own height (leaf
    height 0), floored at a small epsilon so no edge has zero length.
    """
    T = returns_window.shape[1]
    corr = np.corrcoef(returns_window, rowvar=False)
    corr = np.clip(corr, -1.0, 1.0)
    D = np.sqrt(np.maximum(2 * (1 - corr), 0.0))
    iu = np.triu_indices(T, k=1)
    condensed = D[iu]
    Z = linkage(condensed, method=linkage_method)

    height = {i: 0.0 for i in range(T)}
    children_of: Dict[int, List[Tuple[int, float]]] = {}
    for i, (a, b, dist, _cnt) in enumerate(Z):
        node = T + i
        a, b = int(a), int(b)
        height[node] = float(dist)
        w_a = max(height[node] - height[a], 1e-3)
        w_b = max(height[node] - height[b], 1e-3)
        children_of[node] = [(a, w_a), (b, w_b)]
    root = T + len(Z) - 1
    return children_of, root, 2 * T - 1


def relocate_points(Z_init: np.ndarray, anchors: np.ndarray, target_d: np.ndarray, w: np.ndarray,
                    n_steps: int, lr: float) -> np.ndarray:
    """
    Move each of m points (independently, vectorized) to best match its
    target hyperbolic distances to k FIXED anchors — an "out-of-sample
    insertion" against a frozen map, not a full re-embedding. Riemannian
    gradient descent (conformal-factor-scaled Euclidean gradient, per Nickel
    & Kiela 2017), warm-started from Z_init.
    """
    Z = Z_init.copy()
    valid = np.isfinite(target_d) & (w > 0)
    td = np.where(valid, target_d, 0.0)
    ww = np.where(valid, w, 0.0)
    for _ in range(n_steps):
        diff = Z[:, None, :] - anchors[None, :, :]           # (m,k,2)
        diff2 = np.sum(diff ** 2, axis=2)                      # (m,k)
        Bz = np.maximum(1 - np.sum(Z ** 2, axis=1), 1e-12)      # (m,)
        Ba = np.maximum(1 - np.sum(anchors ** 2, axis=1), 1e-12)  # (k,)
        denom = np.maximum(Bz[:, None] * Ba[None, :], 1e-12)
        gamma = np.maximum(1 + 2 * diff2 / denom, 1.0 + 1e-12)
        dH = np.arccosh(gamma)
        err = dH - td
        sq = np.sqrt(np.maximum(gamma ** 2 - 1, 1e-24))
        dgamma_dz = (4 * diff / denom[:, :, None]
                    + 4 * diff2[:, :, None] * Z[:, None, :] / ((Bz[:, None] ** 2 * Ba[None, :])[:, :, None]))
        ddH_dz = dgamma_dz / sq[:, :, None]
        coef = (2 * ww * err)[:, :, None]
        grad = np.sum(coef * ddH_dz, axis=1)                    # (m,2)
        rgrad = grad * (Bz[:, None] ** 2) / 4.0
        Z = Z - lr * rgrad
        norms = np.linalg.norm(Z, axis=1)
        over = norms >= 0.995
        if over.any():
            Z[over] = Z[over] / norms[over, None] * 0.995
    return Z


# ---------------------------------------------------------------------------
# Full feature pipeline for one universe
# ---------------------------------------------------------------------------
def build_hyperbolic_series(returns: np.ndarray, tree_corr_window: int, tree_refit_interval: int,
                            short_corr_window: int, target_max_radius: float, relocate_steps: int, relocate_lr: float,
                            migration_lookback: int, local_k: int,
                            linkage_method: str = "average") -> Dict[str, np.ndarray]:
    """
    Full history pass: rebuilds the map every `tree_refit_interval` days from
    a `tree_corr_window`-day correlation tree (Sarkar-embedded with a tau
    chosen per-tree so its deepest leaf lands at `target_max_radius`
    hyperbolic units — see adaptive_tau), and between refits relocates every
    ticker daily against that frozen map using its trailing
    `short_corr_window`-day correlations. Returns per-day, per-ticker
    positions and derived features. O(n_days) map-building cost is tiny
    (Sarkar is closed-form); daily relocation is a few vectorized gradient
    steps against <=~45 anchors — fast even across full history.
    """
    n_days, T = returns.shape
    positions = np.full((n_days, T, 2), np.nan)
    anchors = None
    days_since_refit = tree_refit_interval  # forces a build on the first valid day
    warm = None

    for t in range(n_days):
        if t + 1 < max(tree_corr_window, short_corr_window):
            continue
        if days_since_refit >= tree_refit_interval or anchors is None:
            window = returns[t + 1 - tree_corr_window: t + 1]
            try:
                children_of, root, n_nodes = build_correlation_tree(window, linkage_method)
                tau_t = adaptive_tau(children_of, root, target_max_radius)
                full_pos = sarkar_embed_tree(children_of, root, n_nodes, tau_t)
                anchors = full_pos[:T]
                warm = anchors.copy()
            except Exception:
                if anchors is None:
                    continue
            days_since_refit = 0

        short_window = returns[t + 1 - short_corr_window: t + 1]
        corr = np.corrcoef(short_window, rowvar=False)
        corr = np.clip(corr, -1.0, 1.0)
        D_today = np.sqrt(np.maximum(2 * (1 - corr), 0.0))
        w = np.ones((T, T))
        np.fill_diagonal(w, 0.0)

        z0 = warm if warm is not None else anchors
        new_pos = relocate_points(z0, anchors, D_today, w, relocate_steps, relocate_lr)
        positions[t] = new_pos
        warm = new_pos.copy()
        days_since_refit += 1

    valid = np.isfinite(positions).all(axis=2)
    radius = np.full((n_days, T), np.nan)
    density = np.full((n_days, T), np.nan)
    for t in range(n_days):
        if not valid[t].any():
            continue
        Z = positions[t]
        r = np.linalg.norm(Z, axis=1)
        radius[t] = 2 * np.arctanh(np.clip(r, 0, 0.999))
        Dtoday = poincare_dist_matrix(Z)
        k = min(local_k, T - 1)
        if k > 0:
            nn = np.sort(Dtoday, axis=1)[:, 1: 1 + k]
            density[t] = nn.mean(axis=1)

    migration = np.full((n_days, T), np.nan)
    radius_chg = np.full((n_days, T), np.nan)
    density_chg = np.full((n_days, T), np.nan)
    for t in range(migration_lookback, n_days):
        if valid[t].any() and valid[t - migration_lookback].any():
            both = valid[t] & valid[t - migration_lookback]
            d = np.linalg.norm(positions[t] - positions[t - migration_lookback], axis=1)
            diff2 = d ** 2
            Bz = np.maximum(1 - np.sum(positions[t] ** 2, axis=1), 1e-12)
            Bp = np.maximum(1 - np.sum(positions[t - migration_lookback] ** 2, axis=1), 1e-12)
            gamma = np.maximum(1 + 2 * diff2 / np.maximum(Bz * Bp, 1e-12), 1.0 + 1e-12)
            mig = np.arccosh(gamma)
            migration[t] = np.where(both, mig, np.nan)
            radius_chg[t] = np.where(both, radius[t] - radius[t - migration_lookback], np.nan)
            density_chg[t] = np.where(both, density[t] - density[t - migration_lookback], np.nan)

    return {"positions": positions, "radius": radius, "migration": migration,
            "local_density": density, "density_chg": density_chg, "radius_chg": radius_chg}


def build_feature_array(returns: np.ndarray, hyp: Dict[str, np.ndarray]) -> np.ndarray:
    """(n, T, N_FEAT): baseline (mom5, mom21, vol21, corr_to_market) + hyperbolic readout."""
    n, T = returns.shape
    mom5 = trailing_sum(returns, 5)
    L = 21
    mom21 = trailing_sum(returns, L)
    s1, s2 = trailing_sum(returns, L), trailing_sum(returns ** 2, L)
    vol21 = 0.5 * np.log(np.maximum((s2 - s1 ** 2 / L) / (L - 1), 1e-10))

    mkt = np.nanmean(returns, axis=1, keepdims=True)
    sr = trailing_sum(returns * mkt, L)
    sd_r = trailing_sum(returns ** 2, L)
    sd_m = trailing_sum(mkt ** 2, L)
    cov = sr / L - (trailing_sum(returns, L) / L) * (trailing_sum(mkt, L) / L)
    var_r = np.maximum(sd_r / L - (trailing_sum(returns, L) / L) ** 2, 1e-12)
    var_m = np.maximum(sd_m / L - (trailing_sum(mkt, L) / L) ** 2, 1e-12)
    corr_to_market = cov / np.sqrt(var_r * var_m)

    feats = np.full((n, T, N_FEAT), np.nan)
    feats[:, :, 0] = mom5
    feats[:, :, 1] = mom21
    feats[:, :, 2] = vol21
    feats[:, :, 3] = corr_to_market
    feats[:, :, 4] = hyp["radius"]
    feats[:, :, 5] = hyp["radius_chg"]
    feats[:, :, 6] = hyp["migration"]
    feats[:, :, 7] = hyp["local_density"]
    feats[:, :, 8] = hyp["density_chg"]
    return feats


# ---------------------------------------------------------------------------
# Cross-sectional regression machinery (same shape as the sibling engines)
# ---------------------------------------------------------------------------
def cs_zscore(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
        out = np.where(sd > 1e-9, (a - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    out[~np.isfinite(a)] = np.nan
    return out


def build_cross_products(Fz: np.ndarray, Yz: np.ndarray) -> Dict:
    n, T, p = Fz.shape
    feat_ok = np.isfinite(Fz).all(axis=(1, 2))
    y_ok = np.isfinite(Yz).all(axis=1)
    ok = feat_ok & y_ok
    F0 = np.where(ok[:, None, None], np.nan_to_num(Fz), 0.0)
    Y0 = np.where(ok[:, None], np.nan_to_num(Yz), 0.0)
    XtX = np.einsum("dtp,dtq->dpq", F0, F0)
    Xty = np.einsum("dtp,dt->dp", F0, Y0)
    return {
        "cXtX": np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)], axis=0),
        "cXty": np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)], axis=0),
        "cN": np.concatenate([[0], np.cumsum(ok.astype(int))]),
        "feat_ok": feat_ok,
    }


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, lam_rel: float) -> np.ndarray:
    p = XtX.shape[0]
    lam = lam_rel * np.trace(XtX) / p + 1e-12
    return np.linalg.solve(XtX + lam * np.eye(p), Xty)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def walk_forward(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                 top_n: int, cfg: Dict) -> Optional[Dict]:
    n, T, p = Fz.shape
    base_idx = list(range(N_BASE))
    lam = cfg["ridge_lambda"]
    min_train = cfg["min_train_days"]
    cost = cfg["cost_bps"] / 1e4

    rows, ic_m, ic_b, sg_m, sg_b, sn_m, sn_b = [], [], [], [], [], [], []
    prev_m, prev_b = set(), set()
    i = 0
    while i <= n - 1 - horizon:
        a, b = max(0, i - window), i - horizon
        if b < a or not cp["feat_ok"][i]:
            i += 1 if b < a else horizon
            continue
        if cp["cN"][b + 1] - cp["cN"][a] < min_train:
            i += horizon
            continue
        XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
        Xty = cp["cXty"][b + 1] - cp["cXty"][a]
        try:
            beta_m = ridge_solve(XtX, Xty, lam)
            beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], lam)
        except np.linalg.LinAlgError:
            i += horizon
            continue
        f = Fz[i]
        s_m = f @ beta_m
        s_b = f[:, base_idx] @ beta_b
        r = fwd[i]
        if not np.isfinite(r).all():
            i += horizon
            continue
        top_m = set(np.argsort(-s_m)[:top_n].tolist())
        top_b = set(np.argsort(-s_b)[:top_n].tolist())
        g_m = r[list(top_m)].mean() - r.mean()
        g_b = r[list(top_b)].mean() - r.mean()
        c_m = (top_n - len(top_m & prev_m)) / top_n * cost
        c_b = (top_n - len(top_b & prev_b)) / top_n * cost
        rows.append(i)
        ic_m.append(spearman(s_m, r))
        ic_b.append(spearman(s_b, r))
        sg_m.append(g_m)
        sg_b.append(g_b)
        sn_m.append(g_m - c_m)
        sn_b.append(g_b - c_b)
        prev_m, prev_b = top_m, top_b
        i += horizon

    if len(rows) < 10:
        return None
    return {k: np.array(v) for k, v in dict(rows=rows, ic_m=ic_m, ic_b=ic_b, sg_m=sg_m, sg_b=sg_b,
                                              sn_m=sn_m, sn_b=sn_b).items()}


def summarize(wf: Dict, lo: int, hi: int, horizon: int) -> Dict:
    sl = slice(lo, hi)
    ic_m, ic_b = wf["ic_m"][sl], wf["ic_b"][sl]
    sn_m, sn_b, sg_m = wf["sn_m"][sl], wf["sn_b"][sl], wf["sg_m"][sl]
    n = len(ic_m)
    if n < 3:
        return {"n": n}
    ann = np.sqrt(252.0 / horizon)
    sd_net = sn_m.std() + 1e-12
    sd_gross = sg_m.std() + 1e-12
    return {
        "n": int(n),
        "ic_mean": float(ic_m.mean()), "ic_t": tstat(ic_m),
        "ic_base_mean": float(ic_b.mean()), "ic_base_t": tstat(ic_b),
        "net_spread_mean": float(sn_m.mean()), "net_spread_t": tstat(sn_m),
        "gross_spread_mean": float(sg_m.mean()),
        "base_net_spread_mean": float(sn_b.mean()),
        "hit_rate": float((sn_m > 0).mean()),
        "sharpe_net": float(sn_m.mean() / sd_net * ann),
        "sharpe_gross": float(sg_m.mean() / sd_gross * ann),
        "d_ic_mean": float((ic_m - ic_b).mean()), "d_ic_t": tstat(ic_m - ic_b),
        "d_spread_mean": float((sn_m - sn_b).mean()), "d_spread_t": tstat(sn_m - sn_b),
    }


def live_scores(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                cfg: Dict) -> Optional[Dict]:
    n = Fz.shape[0]
    i = n - 1
    a, b = max(0, i - window), i - horizon
    if b < a or not cp["feat_ok"][i] or cp["cN"][b + 1] - cp["cN"][a] < cfg["min_train_days"]:
        return None
    XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
    Xty = cp["cXty"][b + 1] - cp["cXty"][a]
    base_idx = list(range(N_BASE))
    beta_m = ridge_solve(XtX, Xty, cfg["ridge_lambda"])
    beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], cfg["ridge_lambda"])
    f = Fz[i]
    disp = np.nanmean(np.nanstd(fwd[a:b + 1], axis=1))
    return {
        "score_m": f @ beta_m,
        "score_b": f[:, base_idx] @ beta_b,
        "beta_m": beta_m,
        "dispersion": float(disp) if np.isfinite(disp) else 0.0,
    }


# ---------------------------------------------------------------------------
# Structural-migration lead diagnostic
# ---------------------------------------------------------------------------
def _onsets(flag: np.ndarray, dedup: int) -> np.ndarray:
    out, last = [], -10 ** 9
    for t in np.where(flag)[0]:
        if t - last > dedup:
            out.append(t)
            last = t   # advance only on a RECORDED onset — see the sibling
                       # engine's writeup for why advancing on every flagged
                       # day (even unrecorded ones) breaks deduplication.
    return np.array(out, dtype=int)


def migration_lead_diagnostic(migration: np.ndarray, returns: np.ndarray, vol21: np.ndarray,
                              dates: List[str], lp: Dict, n_shifts: int = 200, seed: int = 0) -> Dict:
    """
    Does a large migration event (an ETF's position jumping — the crossing
    moment above its own trailing z-score threshold, not the whole elevated
    stretch) tend to sit before a large realized move? Per ETF, then pooled.

    The chance baseline is EMPIRICAL (circular-shift permutation), not an
    analytic uniform-random-placement formula. Validated reason: dedup'd
    onsets/events are a renewal process with a minimum spacing
    (`dedup_days`), not a uniform-random scatter — two independent
    quasi-periodic sequences with similar typical spacing overlap MORE than
    a naive uniform-placement formula predicts (a "beat frequency" effect),
    which was caught by testing this diagnostic on fully independent
    synthetic onset and event series: the analytic formula gave hit_rate
    0.56 vs its own chance_hit_rate 0.44 on data with NO real relationship
    by construction. A circular shift preserves each ETF's actual onset
    spacing pattern exactly (just rotates it to a random phase against the
    event timeline), so it reproduces that periodicity in the null instead
    of assuming it away.
    """
    n, T = migration.shape
    fwd_cum = np.full((n, T), np.nan)
    W = lp["event_window"]
    if n > W:
        cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
        fwd_cum[: n - W] = cum[W + 1: n + 1] - cum[1: n - W + 1]
    thresh = lp["event_vol_k"] * vol21 * np.sqrt(W)
    event_flag_all = np.abs(fwd_cum) > thresh

    mig_z = np.full((n, T), np.nan)
    zwin = lp["z_window"]
    for j in range(T):
        s = pd.Series(migration[:, j])
        mu = s.rolling(zwin, min_periods=63).mean()
        sd = s.rolling(zwin, min_periods=63).std()
        mig_z[:, j] = ((s - mu) / sd.replace(0, np.nan)).values
    above = np.nan_to_num(mig_z, nan=-9.0) > lp["migration_z"]
    prev_below = np.roll(~above, 1, axis=0)
    prev_below[0] = False
    onset_flag_all = above & prev_below

    LW, dedup = lp["lead_window"], lp["dedup_days"]
    rng = np.random.default_rng(seed)
    leads = []
    n_onsets = n_events = n_hits = n_false = 0
    chance_hits_total = chance_false_total = 0.0

    def hits_and_false(on_shifted, ev, span):
        on_mod = start + np.mod(on_shifted - start, span)
        h = 0
        for d in ev:
            if ((on_mod >= d - LW) & (on_mod < d)).any():
                h += 1
        f = sum(1 for m in on_mod if not ((ev > m) & (ev <= m + LW)).any())
        return h, f

    for j in range(T):
        ok = np.isfinite(mig_z[:, j]) & np.isfinite(fwd_cum[:, j])
        if not ok.any():
            continue
        start = int(np.argmax(ok))
        end = n
        span = end - start
        ev = _onsets(event_flag_all[:, j] & ok, dedup)
        on = _onsets(onset_flag_all[:, j] & ok, dedup)
        on = on[on >= start]
        ev = ev[ev >= start]
        n_onsets += len(on)
        n_events += len(ev)

        for d in ev:
            prior = on[(on >= d - LW) & (on < d)]
            if len(prior):
                n_hits += 1
                leads.append(int(d - prior.min()))
        n_false += sum(1 for m in on if not ((ev > m) & (ev <= m + LW)).any())

        if len(on) and len(ev) and span > 0:
            shifts = rng.integers(0, span, size=n_shifts)
            hit_counts = np.empty(n_shifts)
            false_counts = np.empty(n_shifts)
            for si, sh in enumerate(shifts):
                h, f = hits_and_false(on + sh, ev, span)
                hit_counts[si] = h
                false_counts[si] = f
            chance_hits_total += hit_counts.mean()
            chance_false_total += false_counts.mean()

    return {
        "available": True,
        "n_migration_onsets": int(n_onsets),
        "n_large_move_events": int(n_events),
        "hit_rate": float(n_hits / n_events) if n_events else None,
        "chance_hit_rate": float(chance_hits_total / n_events) if n_events else None,
        "median_lead_days": float(np.median(leads)) if leads else None,
        "false_alarm_rate": float(n_false / n_onsets) if n_onsets else None,
        "chance_false_alarm_rate": float(chance_false_total / n_onsets) if n_onsets else None,
    }


def taxonomy_alignment(positions_today: np.ndarray, tickers: List[str], asset_class_map: Dict[str, str]) -> Optional[Dict]:
    """
    Diagnostic only (never fed into the model): does the data-driven map
    roughly recover the hand-labeled asset-class taxonomy? Mean within-class
    hyperbolic distance vs mean between-class distance, and a silhouette-like
    ratio — higher means the discovered hierarchy agrees with the taxonomy
    more than it disagrees.
    """
    labels = [asset_class_map.get(t) for t in tickers]
    if len(set(l for l in labels if l)) < 2:
        return None
    D = poincare_dist_matrix(positions_today)
    within, between = [], []
    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] is None or labels[j] is None:
                continue
            (within if labels[i] == labels[j] else between).append(D[i, j])
    if not within or not between:
        return None
    w_mean, b_mean = float(np.mean(within)), float(np.mean(between))
    return {
        "mean_within_class_distance": round(w_mean, 4),
        "mean_between_class_distance": round(b_mean, 4),
        "separation_ratio": round(b_mean / w_mean, 3) if w_mean > 0 else None,
    }
