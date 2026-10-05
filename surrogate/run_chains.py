"""
run_chains.py — rj-MCMC chain runner for surrogate seismic datasets.

Quick-look mode  : run one short chain per dataset, plot results.
Multi-chain mode : run N independent chains, pool ensembles, compute
                   per-grid-point mean and uncertainty (std across chains).

Usage
-----
Datasets are read from data/, figures are written to outputs/.

python run_chains.py                              # quick-look on both datasets
python run_chains.py --dataset v1                 # only v1
python run_chains.py --dataset v2 --nchains 4    # 4 parallel chains on v2
python run_chains.py --dataset v2 --nchains 4 --nsteps 150000

Key optimisations vs naive implementation
-----------------------------------------
1. Sparse contribution matrix C[ray, cell] = total path length through cell.
   - Slowness updates: t_new = t_cur + delta_s * C[:,k]  -> O(~23 ops) not O(422k)
   - Accepted-state C is never overwritten by rejected proposals,
     eliminating spurious KDTree rebuilds after every geometry step.
2. OVERSAMPLE 2 pts/km (the notebook uses 5): fewer sample points -> faster KDTree queries.
3. multiprocessing.Pool for parallel independent chains.
Combined speedup: ~3-4x per chain, linear with core count for multi-chain runs.
"""

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.stats import norm as spnorm

warnings.filterwarnings("ignore")

# ── Dark plot theme ────────────────────────────────────────────────────────────
DARK, PANEL, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED = "#e6edf3", "#8b949e"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
})

# ── Hyperparameters ────────────────────────────────────────────────────────────
S_MIN, S_MAX   = 0.18, 0.62
DELTA_S        = S_MAX - S_MIN
N_MIN, N_MAX   = 2, 200
SIG_S, SIG_S2, SIG_C = 0.006, 0.012, 2.0
SIG_S_DR       = SIG_S * 0.25
SIG_C_DR       = SIG_C * 0.25
OVERSAMPLE     = 2          # pts/km (was 5; 2 pts/km is accurate and 4x faster overall)
_LOG_SIG2_SQRT2PI = np.log(SIG_S2 * np.sqrt(2.0 * np.pi))

RNG = np.random.default_rng(0)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Dataset:
    name: str
    d_obs: np.ndarray
    d_true: np.ndarray
    sources: np.ndarray
    receivers: np.ndarray
    slowness_true: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    s0: float
    noise_std: float

    # Filled by prepare_rays()
    all_pts_flat: np.ndarray  = field(default=None, repr=False)
    ray_pt_counts: np.ndarray = field(default=None, repr=False)
    ray_indices: np.ndarray   = field(default=None, repr=False)
    seg_lens_arr: np.ndarray  = field(default=None, repr=False)
    seg_weights: np.ndarray   = field(default=None, repr=False)  # per-point seg length
    n_rays: int               = field(default=0)
    GHOST_NUC: np.ndarray     = field(default=None, repr=False)
    N_GHOST: int              = field(default=0)

    # Accepted-state contribution matrix (never overwritten by rejected proposals)
    _C: object     = field(default=None, repr=False)   # sparse CSC (n_rays, n_nuc_total)
    _C_key: bytes  = field(default=None, repr=False)


def load_dataset(path: str, name: str) -> "Dataset":
    raw = np.load(path)
    gp  = raw["grid_params"]
    return Dataset(
        name=name,
        d_obs=raw["d_obs"].astype(np.float64),
        d_true=raw["d_true"].astype(np.float64),
        sources=raw["sources"].astype(np.float64),
        receivers=raw["receivers"].astype(np.float64),
        slowness_true=raw["slowness_true"].astype(np.float64),
        xc=raw["xc"].astype(np.float64),
        yc=raw["yc"].astype(np.float64),
        Nx=int(gp[0]), Ny=int(gp[1]),
        Lx=float(gp[2]), Ly=float(gp[3]),
        s0=float(gp[6]),
        noise_std=float(raw["noise_std"]),
    )


def prepare_rays(ds: Dataset) -> None:
    """Pre-compute ray sample points, ghost nuclei ring, and per-point seg weights."""
    pts_list, segl_list = [], []
    for sx, sy in ds.sources:
        for rx, ry in ds.receivers:
            ray_len = np.hypot(rx - sx, ry - sy)
            n_steps = max(int(ray_len * OVERSAMPLE), 8)
            ts_mid  = 0.5 * (np.linspace(0, 1, n_steps + 1)[:-1]
                             + np.linspace(0, 1, n_steps + 1)[1:])
            xm = sx + ts_mid * (rx - sx)
            ym = sy + ts_mid * (ry - sy)
            inside = (xm >= 0) & (xm <= ds.Lx) & (ym >= 0) & (ym <= ds.Ly)
            xm, ym = xm[inside], ym[inside]
            if len(xm) == 0:
                xm = np.array([0.5 * (sx + rx)])
                ym = np.array([0.5 * (sy + ry)])
            pts_list.append(np.column_stack([xm, ym]))
            segl_list.append(ray_len / n_steps)

    ds.all_pts_flat  = np.vstack(pts_list)
    ds.ray_pt_counts = np.array([len(p) for p in pts_list])
    ds.ray_indices   = np.repeat(np.arange(len(pts_list)), ds.ray_pt_counts)
    ds.seg_lens_arr  = np.array(segl_list)
    ds.seg_weights   = np.repeat(ds.seg_lens_arr, ds.ray_pt_counts)  # per sample point
    ds.n_rays        = len(pts_list)

    margin, n_per = 1.5, 10
    gpts = []
    xs = np.linspace(-margin, ds.Lx + margin, n_per + 2)
    ys = np.linspace(-margin, ds.Ly + margin, n_per + 2)
    for x in xs:
        gpts.extend([[x, -margin], [x, ds.Ly + margin]])
    for y in ys[1:-1]:
        gpts.extend([[-margin, y], [ds.Lx + margin, y]])
    ds.GHOST_NUC = np.array(gpts)
    ds.N_GHOST   = len(ds.GHOST_NUC)

    print(f"[{ds.name}] Rays: {ds.n_rays}  "
          f"Sample pts: {len(ds.all_pts_flat):,}  "
          f"Ghost nuclei: {ds.N_GHOST}")


# ══════════════════════════════════════════════════════════════════════════════
#  Contribution matrix & forward model
# ══════════════════════════════════════════════════════════════════════════════

def _build_C(ds: Dataset, nuclei: np.ndarray):
    """
    KDTree query + sparse contribution matrix for given mobile nuclei.
    Returns (C_csc, key) WITHOUT touching ds._C — safe to call for proposals.

    C[ray_r, cell_i] = total path length through Voronoi cell i for ray r.
    Travel times: t = C @ full_slowness_vector.
    """
    full_nuc = np.vstack([nuclei, ds.GHOST_NUC])
    n_mobile = len(nuclei)
    _, asgn  = cKDTree(full_nuc).query(ds.all_pts_flat)

    # Sample points exactly on the domain boundary (produced by same-edge
    # source–receiver pairs whose entire ray lies at x=0, x=Lx, y=0, or y=Ly)
    # are always closer to an interior mobile nucleus than to the ghost ring
    # 1.5 km outside the domain.  Force them onto the nearest ghost so that
    # boundary rays always use background slowness, preventing mobile nuclei
    # from "leaking" incorrect slowness values onto the boundary.
    pts = ds.all_pts_flat
    bnd = ((pts[:, 0] == 0.0) | (pts[:, 0] == ds.Lx) |
           (pts[:, 1] == 0.0) | (pts[:, 1] == ds.Ly))
    if bnd.any():
        _, g_idx = cKDTree(ds.GHOST_NUC).query(pts[bnd])
        asgn[bnd] = n_mobile + g_idx

    C = csr_matrix(
        (ds.seg_weights, (ds.ray_indices, asgn)),
        shape=(ds.n_rays, len(full_nuc)),
    ).tocsc()
    return C, full_nuc.tobytes()


def _ensure_C(ds: Dataset, nuclei: np.ndarray):
    """Return ds._C, rebuilding only if nuclei changed (accepted-state cache)."""
    key = np.vstack([nuclei, ds.GHOST_NUC]).tobytes()
    if key != ds._C_key:
        ds._C, ds._C_key = _build_C(ds, nuclei)
    return ds._C


def _full_s(ds: Dataset, s_vals: np.ndarray) -> np.ndarray:
    return np.concatenate([s_vals, np.full(ds.N_GHOST, ds.s0)])


def forward_model(ds: Dataset, nuclei: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    """Public forward model (uses C matrix; safe for plotting/residual calls)."""
    C = _ensure_C(ds, nuclei)
    return np.asarray(C @ _full_s(ds, s_vals)).ravel()


def log_likelihood(ds: Dataset, t_pred: np.ndarray) -> float:
    return -0.5 * np.sum(((t_pred - ds.d_obs) / ds.noise_std) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
#  rj-MCMC move types  (all carry t_cur through the signature)
# ══════════════════════════════════════════════════════════════════════════════

def _move_slowness(ds, nuclei, s_vals, t_cur, logL, sigma):
    """
    Perturb slowness of one cell. Uses incremental C[:,k] update ->
    only touches the ~23 rays that pass through cell k.
    No KDTree call; never invalidates the accepted C.
    """
    k     = int(RNG.integers(len(s_vals)))
    delta = float(RNG.normal(0.0, sigma))
    s_new = s_vals[k] + delta
    if not (S_MIN <= s_new <= S_MAX):
        return nuclei, s_vals, t_cur, logL, False

    C = _ensure_C(ds, nuclei)           # always a cache hit in slowness steps
    # Extract column k from CSC data arrays (O(nnz_k), no allocation of full array)
    lo, hi      = C.indptr[k], C.indptr[k + 1]
    col_rows    = C.indices[lo:hi]
    col_vals    = C.data[lo:hi]

    t_prop          = t_cur.copy()
    t_prop[col_rows] += delta * col_vals
    logL_prop        = log_likelihood(ds, t_prop)

    if np.log(RNG.random() + 1e-300) < logL_prop - logL:
        s_prop    = s_vals.copy(); s_prop[k] = s_new
        return nuclei, s_prop, t_prop, logL_prop, True
    return nuclei, s_vals, t_cur, logL, False


def _dr_slowness(ds, nuclei, s_vals, t_cur, logL):
    n, s, t, L, a = _move_slowness(ds, nuclei, s_vals, t_cur, logL, SIG_S)
    if a: return n, s, t, L, True
    return _move_slowness(ds, nuclei, s_vals, t_cur, logL, SIG_S_DR)


def _move_nucleus(ds, nuclei, s_vals, t_cur, logL, sigma):
    """
    Perturb position of one nucleus. Builds a TEMPORARY C for the proposal
    without overwriting ds._C, so the accepted-state C stays valid on rejection.
    """
    k       = int(RNG.integers(len(nuclei)))
    new_pos = nuclei[k] + RNG.normal(0.0, sigma, 2)
    if not (0.0 < new_pos[0] < ds.Lx and 0.0 < new_pos[1] < ds.Ly):
        return nuclei, s_vals, t_cur, logL, False

    nuc_p          = nuclei.copy(); nuc_p[k] = new_pos
    C_prop, key_p  = _build_C(ds, nuc_p)           # temporary, ds._C untouched
    t_prop         = np.asarray(C_prop @ _full_s(ds, s_vals)).ravel()
    logL_p         = log_likelihood(ds, t_prop)

    if np.log(RNG.random() + 1e-300) < logL_p - logL:
        ds._C, ds._C_key = C_prop, key_p            # promote proposal to accepted
        return nuc_p, s_vals, t_prop, logL_p, True
    return nuclei, s_vals, t_cur, logL, False       # ds._C still valid for nuclei


def _dr_nucleus(ds, nuclei, s_vals, t_cur, logL):
    n, s, t, L, a = _move_nucleus(ds, nuclei, s_vals, t_cur, logL, SIG_C)
    if a: return n, s, t, L, True
    # Stage-1 was rejected -> ds._C is still valid for original nuclei
    return _move_nucleus(ds, nuclei, s_vals, t_cur, logL, SIG_C_DR)


def _move_birth(ds, nuclei, s_vals, t_cur, logL):
    if len(nuclei) >= N_MAX:
        return nuclei, s_vals, t_cur, logL, False

    cx, cy  = float(RNG.uniform(0, ds.Lx)), float(RNG.uniform(0, ds.Ly))
    # Local slowness via brute-force distance (O(n_nuc), avoids KDTree)
    fn      = np.vstack([nuclei, ds.GHOST_NUC])
    fs      = _full_s(ds, s_vals)
    d2      = (fn[:, 0] - cx)**2 + (fn[:, 1] - cy)**2
    s_local = float(fs[np.argmin(d2)])

    u_v     = float(RNG.normal(0.0, SIG_S2))
    s_birth = s_local + u_v
    if not (S_MIN <= s_birth <= S_MAX):
        return nuclei, s_vals, t_cur, logL, False

    nuc_p          = np.vstack([nuclei, [cx, cy]])
    s_p            = np.append(s_vals, s_birth)
    C_prop, key_p  = _build_C(ds, nuc_p)
    t_prop         = np.asarray(C_prop @ _full_s(ds, s_p)).ravel()
    logL_p         = log_likelihood(ds, t_prop)

    log_alpha = (_LOG_SIG2_SQRT2PI - np.log(DELTA_S)
                 + u_v**2 / (2.0 * SIG_S2**2) + logL_p - logL)
    if np.log(RNG.random() + 1e-300) < log_alpha:
        ds._C, ds._C_key = C_prop, key_p
        return nuc_p, s_p, t_prop, logL_p, True
    return nuclei, s_vals, t_cur, logL, False


def _move_death(ds, nuclei, s_vals, t_cur, logL):
    if len(nuclei) <= N_MIN:
        return nuclei, s_vals, t_cur, logL, False

    k          = int(RNG.integers(len(nuclei)))
    s_rem      = float(s_vals[k])
    pos_rem    = nuclei[k].copy()
    nuc_p      = np.delete(nuclei, k, axis=0)
    s_p        = np.delete(s_vals, k)

    C_prop, key_p = _build_C(ds, nuc_p)
    fs_p          = _full_s(ds, s_p)
    t_prop        = np.asarray(C_prop @ fs_p).ravel()
    logL_p        = log_likelihood(ds, t_prop)

    # Slowness at removed nucleus in the new tessellation (brute-force distance)
    fn_p    = np.vstack([nuc_p, ds.GHOST_NUC])
    d2      = (fn_p[:, 0] - pos_rem[0])**2 + (fn_p[:, 1] - pos_rem[1])**2
    s_after = float(fs_p[np.argmin(d2)])

    log_alpha = (np.log(DELTA_S) - _LOG_SIG2_SQRT2PI
                 - (s_rem - s_after)**2 / (2.0 * SIG_S2**2) + logL_p - logL)
    if np.log(RNG.random() + 1e-300) < log_alpha:
        ds._C, ds._C_key = C_prop, key_p
        return nuc_p, s_p, t_prop, logL_p, True
    return nuclei, s_vals, t_cur, logL, False


# ══════════════════════════════════════════════════════════════════════════════
#  Main rj-MCMC sampler
# ══════════════════════════════════════════════════════════════════════════════

def rj_mcmc(
    ds: Dataset,
    n_steps: int,
    n_burn: int,
    thin: int,
    n_init: int = 8,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[List, dict]:
    global RNG
    RNG = np.random.default_rng(seed)
    ds._C = ds._C_key = None        # reset accepted-state cache

    nuclei = RNG.uniform(low=[0, 0], high=[ds.Lx, ds.Ly], size=(n_init, 2))
    s_vals = RNG.uniform(ds.s0 - 0.01, ds.s0 + 0.01, size=n_init)

    # Build initial C and travel times
    ds._C, ds._C_key = _build_C(ds, nuclei)
    t_cur = np.asarray(ds._C @ _full_s(ds, s_vals)).ravel()
    logL  = log_likelihood(ds, t_cur)

    ensemble    = []
    n_cells_log = np.zeros(n_steps, dtype=np.int32)
    logL_log    = np.zeros(n_steps)
    acc = dict(s=0, move=0, birth=0, death=0)
    tot = dict(s=0, move=0, birth=0, death=0)

    t_wall      = time.time()
    print_every = max(n_steps // 10, 1)

    for step in range(n_steps):
        if step % 2 == 0:
            tot["s"] += 1
            nuclei, s_vals, t_cur, logL, a = _dr_slowness(ds, nuclei, s_vals, t_cur, logL)
            if a: acc["s"] += 1
        else:
            choice = int(RNG.integers(3))
            if choice == 0:
                tot["birth"] += 1
                nuclei, s_vals, t_cur, logL, a = _move_birth(ds, nuclei, s_vals, t_cur, logL)
                if a: acc["birth"] += 1
            elif choice == 1:
                tot["death"] += 1
                nuclei, s_vals, t_cur, logL, a = _move_death(ds, nuclei, s_vals, t_cur, logL)
                if a: acc["death"] += 1
            else:
                tot["move"] += 1
                nuclei, s_vals, t_cur, logL, a = _dr_nucleus(ds, nuclei, s_vals, t_cur, logL)
                if a: acc["move"] += 1

        n_cells_log[step] = len(nuclei)
        logL_log[step]    = logL

        if step >= n_burn and (step - n_burn) % thin == 0:
            ensemble.append((nuclei.copy(), s_vals.copy()))

        if verbose and (step + 1) % print_every == 0:
            elapsed = time.time() - t_wall
            eta     = elapsed / (step + 1) * (n_steps - step - 1)
            print(f"  [{ds.name}] {(step+1)/n_steps*100:5.1f}%  "
                  f"step {step+1:>7,}  logL={logL:10.1f}  "
                  f"n={len(nuclei):3d}  ETA {eta:.0f}s")

    elapsed = time.time() - t_wall
    if verbose:
        print(f"  Finished in {elapsed:.1f}s  ({elapsed/n_steps*1000:.1f} ms/step)  "
              f"ensemble={len(ensemble)}")
        for k in ("s", "move", "birth", "death"):
            if tot[k]:
                print(f"    {k}: {acc[k]/tot[k]*100:.1f}% ({acc[k]}/{tot[k]})")

    return ensemble, dict(
        n_cells=n_cells_log, logL=logL_log,
        n_burn=n_burn, thin=thin, acc=acc, tot=tot, elapsed=elapsed,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Ensemble -> grid projection
# ══════════════════════════════════════════════════════════════════════════════

def project_ensemble(ds: Dataset, ensemble: List, nx: int = 80, ny: int = 80) -> dict:
    xg = np.linspace(0.0, ds.Lx, nx)
    yg = np.linspace(0.0, ds.Ly, ny)
    XG, YG = np.meshgrid(xg, yg)
    query  = np.column_stack([XG.ravel(), YG.ravel()])

    # Pre-compute which grid points sit exactly on the domain boundary so we
    # can force them to ghost-nucleus slowness (background) in every sample,
    # matching the boundary override applied in _build_C.
    bnd_q = ((query[:, 0] == 0.0) | (query[:, 0] == ds.Lx) |
             (query[:, 1] == 0.0) | (query[:, 1] == ds.Ly))
    if bnd_q.any():
        _, ghost_q_idx = cKDTree(ds.GHOST_NUC).query(query[bnd_q])

    K     = len(ensemble)
    stack = np.zeros((K, len(query)))
    for i, (nuc, s) in enumerate(ensemble):
        fn = np.vstack([nuc, ds.GHOST_NUC])
        fs = _full_s(ds, s)
        _, idx = cKDTree(fn).query(query)
        if bnd_q.any():
            idx[bnd_q] = len(nuc) + ghost_q_idx
        stack[i] = fs[idx]

    s_mean = stack.mean(0).reshape(ny, nx)
    s_std  = stack.std(0).reshape(ny, nx)

    ix_all = np.clip(np.floor(ds.all_pts_flat[:, 0] / (ds.Lx / nx)).astype(int), 0, nx-1)
    iy_all = np.clip(np.floor(ds.all_pts_flat[:, 1] / (ds.Ly / ny)).astype(int), 0, ny-1)
    hit    = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(hit, (iy_all, ix_all), 1)

    return dict(xg=xg, yg=yg, XG=XG, YG=YG,
                s_mean=s_mean, s_std=s_std, hit_mask=hit > 0)


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _styled(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    return ax


def plot_convergence(ds: Dataset, diag: dict, chain_id: int = 0) -> plt.Figure:
    n_steps = len(diag["logL"]); n_burn = diag["n_burn"]
    steps   = np.arange(n_steps)
    burn_kw = dict(color="#ff7b72", lw=1.3, ls="--", label="Burn-in end")

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor=DARK)
    for ax in axes.flat: _styled(ax)

    axes[0,0].plot(steps, diag["logL"], color="#58a6ff", lw=0.4, alpha=0.8)
    axes[0,0].axvline(n_burn, **burn_kw)
    axes[0,0].set(xlabel="MCMC step", ylabel="log p(d|m)")
    axes[0,0].set_title("A — Log-likelihood trace", fontweight="bold")
    axes[0,0].legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    axes[0,1].plot(steps, diag["n_cells"], color="#f0883e", lw=0.4, alpha=0.8)
    axes[0,1].axvline(n_burn, **burn_kw)
    axes[0,1].set(xlabel="MCMC step", ylabel="n (Voronoi cells)")
    axes[0,1].set_title("B — Dimension trace", fontweight="bold")
    axes[0,1].legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    post_n = diag["n_cells"][n_burn:]
    bins   = np.arange(max(1, post_n.min()-1), post_n.max()+2) - 0.5
    axes[1,0].hist(post_n, bins=bins, color="#79c0ff", edgecolor=DARK, lw=0.4, density=True)
    mode_n = int(np.bincount(post_n).argmax())
    axes[1,0].axvline(mode_n, color="#ff7b72", lw=1.5, ls="--",
                      label=f"Mode={mode_n}  Median={int(np.median(post_n))}")
    axes[1,0].set(xlabel="n", ylabel="p(n | d_obs)")
    axes[1,0].set_title("C — Posterior dimension", fontweight="bold")
    axes[1,0].legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    post_logL = diag["logL"][n_burn:]
    running   = np.cumsum(post_logL) / (np.arange(len(post_logL)) + 1)
    axes[1,1].plot(running, color="#56d364", lw=1.0)
    axes[1,1].set(xlabel="Post-burn step", ylabel="Running mean logL")
    axes[1,1].set_title("D — Convergence: running mean logL", fontweight="bold")

    fig.suptitle(f"[{ds.name}] Chain {chain_id} — Convergence Diagnostics",
                 color=TEXT, fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    return fig


def plot_results(
    ds: Dataset, proj: dict, ensemble: List, diag: dict,
    title_prefix: str = "", save_path: str = None,
) -> plt.Figure:
    from scipy.interpolate import RegularGridInterpolator
    XG, YG   = proj["XG"], proj["YG"]
    s_mean, s_std, hit_mask = proj["s_mean"], proj["s_std"], proj["hit_mask"]

    interp_fn   = RegularGridInterpolator((ds.yc, ds.xc), ds.slowness_true,
                                          method="nearest", bounds_error=False,
                                          fill_value=ds.s0)
    s_true_fine = interp_fn(np.column_stack([YG.ravel(), XG.ravel()])).reshape(YG.shape)

    s_lo, s_hi  = ds.s0 * 0.78, ds.s0 * 1.25
    norm        = TwoSlopeNorm(vmin=s_lo, vcenter=ds.s0, vmax=s_hi)
    cb_kw       = dict(fraction=0.046, pad=0.04)

    _n      = min(60, len(ensemble))
    t_stack = np.array([forward_model(ds, nuc, s) for nuc, s in ensemble[-_n:]])
    residuals = ds.d_obs - t_stack.mean(0)

    fig = plt.figure(figsize=(18, 12), facecolor=DARK)
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.36,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # A — true
    ax = _styled(fig.add_subplot(gs[0, 0]))
    im = ax.pcolormesh(XG, YG, np.where(hit_mask, s_true_fine, np.nan),
                       cmap="RdBu_r", norm=norm, shading="auto")
    cb = fig.colorbar(im, ax=ax, **cb_kw)
    cb.set_label("s [s/km]", color=MUTED, fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    ax.set(xlabel="X [km]", ylabel="Y [km]", aspect="equal",
           xlim=(0, ds.Lx), ylim=(0, ds.Ly))
    ax.set_title("A — True slowness", fontweight="bold", fontsize=9, pad=6)

    # B — mean
    ax = _styled(fig.add_subplot(gs[0, 1]))
    im = ax.pcolormesh(XG, YG, np.where(hit_mask, s_mean, np.nan),
                       cmap="RdBu_r", norm=norm, shading="auto")
    cb = fig.colorbar(im, ax=ax, **cb_kw)
    cb.set_label("s [s/km]", color=MUTED, fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    last_nuc = ensemble[-1][0]
    ax.scatter(last_nuc[:, 0], last_nuc[:, 1], c="white", s=10, zorder=6, alpha=0.7,
               label=f"Nuclei n={len(last_nuc)}")
    ax.set(xlabel="X [km]", ylabel="Y [km]", aspect="equal",
           xlim=(0, ds.Lx), ylim=(0, ds.Ly))
    ax.set_title(f"B — Ensemble mean  (K={len(ensemble)} samples)",
                 fontweight="bold", fontsize=9, pad=6)
    ax.legend(fontsize=6, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    # C — uncertainty
    ax = _styled(fig.add_subplot(gs[0, 2]))
    im = ax.pcolormesh(XG, YG, np.where(hit_mask, s_std, np.nan),
                       cmap="hot_r", shading="auto")
    cb = fig.colorbar(im, ax=ax, **cb_kw)
    cb.set_label("σ_s [s/km]", color=MUTED, fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    ax.scatter(ds.sources[:, 0], ds.sources[:, 1], c="#79c0ff", s=20, marker="^", zorder=5)
    ax.scatter(ds.receivers[:, 0], ds.receivers[:, 1], c="#ff7b72", s=20, marker="v", zorder=5)
    ax.set(xlabel="X [km]", ylabel="Y [km]", aspect="equal",
           xlim=(0, ds.Lx), ylim=(0, ds.Ly))
    ax.set_title("C — Posterior uncertainty σ_s", fontweight="bold", fontsize=9, pad=6)

    # D — error
    err = np.abs(s_mean - s_true_fine)
    ax = _styled(fig.add_subplot(gs[1, 0]))
    im = ax.pcolormesh(XG, YG, np.where(hit_mask, err, np.nan),
                       cmap="magma", shading="auto")
    cb = fig.colorbar(im, ax=ax, **cb_kw)
    cb.set_label("|Δs| [s/km]", color=MUTED, fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    ax.set(xlabel="X [km]", ylabel="Y [km]", aspect="equal",
           xlim=(0, ds.Lx), ylim=(0, ds.Ly))
    ax.set_title(f"D — |error|  mean={err[hit_mask].mean():.4f}  max={err[hit_mask].max():.4f}",
                 fontweight="bold", fontsize=9, pad=6)

    # E — p(n)
    n_ens = [len(e[0]) for e in ensemble]
    ax = _styled(fig.add_subplot(gs[1, 1]))
    bins_n = np.arange(max(1, min(n_ens)-1), max(n_ens)+2) - 0.5
    ax.hist(n_ens, bins=bins_n, color="#58a6ff", edgecolor=DARK, lw=0.4, density=True)
    ax.axvline(int(np.median(n_ens)), color="#f0883e", lw=1.8, ls="--",
               label=f"Median n={int(np.median(n_ens))}")
    ax.set(xlabel="n (Voronoi cells)", ylabel="p(n | d_obs)")
    ax.set_title("E — p(n | d_obs)", fontweight="bold", fontsize=9, pad=6)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    # F — residuals
    ax = _styled(fig.add_subplot(gs[1, 2]))
    ax.hist(residuals, bins=42, color="#56d364", edgecolor=DARK, lw=0.4,
            density=True, alpha=0.85, label="Residuals")
    xs = np.linspace(residuals.min(), residuals.max(), 300)
    ax.plot(xs, spnorm.pdf(xs, 0, ds.noise_std), color="#ff7b72", lw=2.0,
            label=f"N(0, σ={ds.noise_std:.4f})")
    ax.set(xlabel="d_obs - d_pred  [s]", ylabel="Density")
    ax.set_title("F — Data fit residuals", fontweight="bold", fontsize=9, pad=6)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

    fig.suptitle(f"{title_prefix}[{ds.name}] rj-MCMC Results",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.97)
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK)
        print(f"  Saved -> {save_path}")
    return fig


def plot_mean_solution(
    ds: Dataset, proj: dict, tag: str = "", save_path: str = None
) -> plt.Figure:
    """Standalone mean slowness map with labelled s and v colorbars."""
    XG, YG      = proj["XG"], proj["YG"]
    s_mean, hit = proj["s_mean"], proj["hit_mask"]
    s_lo, s_hi  = ds.s0 * 0.78, ds.s0 * 1.25
    norm        = TwoSlopeNorm(vmin=s_lo, vcenter=ds.s0, vmax=s_hi)

    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=DARK)
    _styled(ax)
    im = ax.pcolormesh(XG, YG, np.where(hit, s_mean, np.nan),
                       cmap="RdBu_r", norm=norm, shading="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Slowness  [s/km]", color=TEXT, fontsize=10)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    cb2 = cb.ax.twinx()
    cb2.set_ylim(1.0 / s_hi, 1.0 / s_lo)
    cb2.set_ylabel("Velocity  [km/s]", color=TEXT, fontsize=10)
    plt.setp(cb2.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cb2.yaxis.set_tick_params(color=MUTED)

    ax.scatter(ds.sources[:, 0], ds.sources[:, 1],
               c="#79c0ff", s=30, marker="^", zorder=5, label="Sources")
    ax.scatter(ds.receivers[:, 0], ds.receivers[:, 1],
               c="#ff7b72", s=30, marker="v", zorder=5, label="Receivers")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.set(xlabel="X  [km]", ylabel="Y  [km]",
           xlim=(0, ds.Lx), ylim=(0, ds.Ly), aspect="equal")
    title = f"[{ds.name}] Ensemble-mean slowness"
    if tag: title += f"  —  {tag}"
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"  Saved -> {save_path}")
    return fig


def plot_uncertainty(
    ds: Dataset, proj: dict, tag: str = "", save_path: str = None
) -> plt.Figure:
    """Standalone posterior std map with labelled colorbar."""
    XG, YG      = proj["XG"], proj["YG"]
    s_std, hit  = proj["s_std"], proj["hit_mask"]

    fig, ax = plt.subplots(figsize=(7, 6.5), facecolor=DARK)
    _styled(ax)
    im = ax.pcolormesh(XG, YG, np.where(hit, s_std, np.nan),
                       cmap="hot_r", shading="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Posterior std  σ_s  [s/km]", color=TEXT, fontsize=10)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=MUTED)

    ax.scatter(ds.sources[:, 0], ds.sources[:, 1],
               c="#79c0ff", s=30, marker="^", zorder=5, label="Sources")
    ax.scatter(ds.receivers[:, 0], ds.receivers[:, 1],
               c="#ff7b72", s=30, marker="v", zorder=5, label="Receivers")
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
    ax.set(xlabel="X  [km]", ylabel="Y  [km]",
           xlim=(0, ds.Lx), ylim=(0, ds.Ly), aspect="equal")
    title = f"[{ds.name}] Posterior uncertainty  σ_s"
    if tag: title += f"  —  {tag}"
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"  Saved -> {save_path}")
    return fig


def plot_multi_chain_comparison(
    ds: Dataset, chain_projs: List[dict], pooled_proj: dict, save_path: str = None
) -> plt.Figure:
    ncols = len(chain_projs) + 1
    s_lo, s_hi = ds.s0 * 0.78, ds.s0 * 1.25
    norm = TwoSlopeNorm(vmin=s_lo, vcenter=ds.s0, vmax=s_hi)

    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 9), facecolor=DARK)
    if ncols == 1: axes = axes[:, np.newaxis]

    for col in range(ncols):
        proj  = pooled_proj if col == ncols - 1 else chain_projs[col]
        label = "Pooled" if col == ncols - 1 else f"Chain {col}"
        XG, YG, hit = proj["XG"], proj["YG"], proj["hit_mask"]

        ax = _styled(axes[0, col])
        im = ax.pcolormesh(XG, YG, np.where(hit, proj["s_mean"], np.nan),
                           cmap="RdBu_r", norm=norm, shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xlabel="X [km]", ylabel="Y [km]", aspect="equal",
               xlim=(0, ds.Lx), ylim=(0, ds.Ly))
        ax.set_title(f"{label} — mean s", fontweight="bold", fontsize=9, pad=6, color=TEXT)

        ax = _styled(axes[1, col])
        im = ax.pcolormesh(XG, YG, np.where(hit, proj["s_std"], np.nan),
                           cmap="hot_r", shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xlabel="X [km]", ylabel="Y [km]", aspect="equal",
               xlim=(0, ds.Lx), ylim=(0, ds.Ly))
        ax.set_title(f"{label} — uncertainty σ_s", fontweight="bold", fontsize=9, pad=6, color=TEXT)

    fig.suptitle(f"[{ds.name}] Multi-chain comparison  (row 1: mean, row 2: σ)",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=DARK)
        print(f"  Saved -> {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level runners
# ══════════════════════════════════════════════════════════════════════════════

def run_quick_look(ds: Dataset, n_steps: int = 30_000, n_burn: int = 10_000,
                   thin: int = 10, seed: int = 7) -> None:
    print(f"\n{'='*60}\n Quick-look chain  [{ds.name}]  ({n_steps:,} steps)\n{'='*60}")
    ensemble, diag = rj_mcmc(ds, n_steps, n_burn, thin, seed=seed)

    fig = plot_convergence(ds, diag, chain_id=0)
    p   = OUT_DIR / f"chain_convergence_{ds.name}_quick.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=DARK); plt.close(fig)
    print(f"  Saved -> {p}")

    print(f"  Projecting {len(ensemble)} models ...")
    proj = project_ensemble(ds, ensemble)
    tag  = f"quick-look  K={len(ensemble)}"

    for fn, fp in [
        (plot_results,       OUT_DIR / f"chain_results_{ds.name}_quick.png"),
    ]:
        fig = fn(ds, proj, ensemble, diag, title_prefix="Quick-look  ", save_path=fp)
        plt.close(fig)

    fig = plot_mean_solution(ds, proj, tag=tag,
                             save_path=OUT_DIR / f"solution_mean_{ds.name}_quick.png")
    plt.close(fig)
    fig = plot_uncertainty(ds, proj, tag=tag,
                           save_path=OUT_DIR / f"solution_uncertainty_{ds.name}_quick.png")
    plt.close(fig)


# ── Parallel chain worker (module-level so it's picklable) ────────────────────

def _chain_worker(args):
    """Worker for multiprocessing.Pool — loads dataset fresh, runs one chain."""
    ds_path, ds_name, n_steps, n_burn, thin, seed = args
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = load_dataset(ds_path, ds_name)
    prepare_rays(ds)
    ensemble, diag = rj_mcmc(ds, n_steps, n_burn, thin, seed=seed, verbose=False)
    print(f"  [worker seed={seed}] done  ensemble={len(ensemble)}", flush=True)
    return ensemble, diag


def run_multi_chain(
    ds: Dataset,
    n_chains: int = 4,
    n_steps: int = 80_000,
    n_burn: int = 30_000,
    thin: int = 10,
    base_seed: int = 100,
    grid_nx: int = 80,
    grid_ny: int = 80,
    ds_path: str = None,     # required for parallel mode
    parallel: bool = True,
) -> None:
    print(f"\n{'='*60}\n Multi-chain  [{ds.name}]  "
          f"{n_chains} chains x {n_steps:,} steps  "
          f"({'parallel' if parallel and ds_path else 'sequential'})\n{'='*60}")

    seeds = [base_seed + c * 17 for c in range(n_chains)]

    if parallel and ds_path:
        args = [(ds_path, ds.name, n_steps, n_burn, thin, s) for s in seeds]
        n_workers = min(n_chains, mp.cpu_count())
        print(f"  Launching {n_chains} workers on {n_workers} cores ...")
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_chain_worker, args)
        all_ensembles = [r[0] for r in results]
        all_diags     = [r[1] for r in results]
    else:
        all_ensembles, all_diags = [], []
        for c, seed in enumerate(seeds):
            print(f"\n-- Chain {c}  (seed={seed}) --")
            ens, diag = rj_mcmc(ds, n_steps, n_burn, thin, seed=seed)
            all_ensembles.append(ens); all_diags.append(diag)

    chain_projs = []
    for c, (ens, diag) in enumerate(zip(all_ensembles, all_diags)):
        fig = plot_convergence(ds, diag, chain_id=c)
        p   = OUT_DIR / f"chain_convergence_{ds.name}_c{c}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight", facecolor=DARK); plt.close(fig)
        print(f"  Saved -> {p}")

        proj_c = project_ensemble(ds, ens, nx=grid_nx, ny=grid_ny)
        chain_projs.append(proj_c)

        fig = plot_results(ds, proj_c, ens, diag,
                           title_prefix=f"Chain {c}  ",
                           save_path=OUT_DIR / f"chain_results_{ds.name}_c{c}.png")
        plt.close(fig)

    print(f"\n-- Pooling {sum(len(e) for e in all_ensembles):,} samples --")
    pooled      = [m for ens in all_ensembles for m in ens]
    pooled_proj = project_ensemble(ds, pooled, nx=grid_nx, ny=grid_ny)

    fig = plot_results(ds, pooled_proj, all_ensembles[-1], all_diags[-1],
                       title_prefix=f"Pooled ({n_chains} chains)  ",
                       save_path=OUT_DIR / f"chain_results_{ds.name}_pooled.png")
    plt.close(fig)

    fig = plot_multi_chain_comparison(ds, chain_projs, pooled_proj,
                                      save_path=OUT_DIR / f"chain_comparison_{ds.name}.png")
    plt.close(fig)

    tag = f"{n_chains} chains pooled  K={len(pooled):,}"
    fig = plot_mean_solution(ds, pooled_proj, tag=tag,
                             save_path=OUT_DIR / f"solution_mean_{ds.name}_pooled.png")
    plt.close(fig)
    fig = plot_uncertainty(ds, pooled_proj, tag=tag,
                           save_path=OUT_DIR / f"solution_uncertainty_{ds.name}_pooled.png")
    plt.close(fig)

    print(f"\n[{ds.name}] Multi-chain complete.")
    n_ens_all = [len(m[0]) for m in pooled]
    print(f"  Chains        : {n_chains}")
    print(f"  Total samples : {len(pooled):,}")
    print(f"  Posterior n   : median={np.median(n_ens_all):.0f}  "
          f"IQR=[{np.percentile(n_ens_all,25):.0f}, {np.percentile(n_ens_all,75):.0f}]")
    print(f"  Mean σ_s      : {pooled_proj['s_std'][pooled_proj['hit_mask']].mean():.5f} s/km")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

DATASET_FILES = {
    "v1": str(HERE / "data" / "surrogate_seismic_dataset.npz"),
    "v2": str(HERE / "data" / "surrogate_seismic_dataset_v2.npz"),
}


def main():
    parser = argparse.ArgumentParser(description="rj-MCMC chain runner")
    parser.add_argument("--dataset",  choices=["v1", "v2", "both"], default="both")
    parser.add_argument("--nchains",  type=int, default=1)
    parser.add_argument("--nsteps",   type=int, default=None)
    parser.add_argument("--nburn",    type=int, default=None)
    parser.add_argument("--thin",     type=int, default=10)
    parser.add_argument("--seed",     type=int, default=7)
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable multiprocessing (useful for debugging)")
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    datasets = ["v1", "v2"] if args.dataset == "both" else [args.dataset]
    for key in datasets:
        path = DATASET_FILES[key]
        if not Path(path).exists():
            print(f"Warning: {path} not found, skipping {key}"); continue
        ds = load_dataset(path, key)
        print(f"\nLoaded [{key}]: {ds.Lx:.0f}x{ds.Ly:.0f} km  "
              f"{len(ds.sources)} src x {len(ds.receivers)} rec  "
              f"noise σ={ds.noise_std:.5f} s")
        prepare_rays(ds)

        if args.nchains <= 1:
            n_steps = args.nsteps or 30_000
            n_burn  = args.nburn  or max(5_000, n_steps // 3)
            run_quick_look(ds, n_steps=n_steps, n_burn=n_burn,
                           thin=args.thin, seed=args.seed)
        else:
            n_steps = args.nsteps or 80_000
            n_burn  = args.nburn  or max(20_000, n_steps // 3)
            run_multi_chain(
                ds, n_chains=args.nchains, n_steps=n_steps, n_burn=n_burn,
                thin=args.thin, base_seed=args.seed,
                ds_path=path, parallel=not args.no_parallel,
            )

    plt.show()


if __name__ == "__main__":
    main()
