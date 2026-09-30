#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, leave=True, disable=False):
        return iterable if iterable is not None else range(total or 0)

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
_cache_key = None
_cache_asgn = None
DATA: Dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RJ-MCMC Voronoi tomography for Denmark interstation travel-time data")
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", default="denmark_rjmcmc_run")
    p.add_argument("--period", type=float, default=None)
    p.add_argument("--min-snr", type=float, default=0.0)
    p.add_argument("--min-days", type=int, default=0)
    p.add_argument("--max-rays", type=int, default=None)
    p.add_argument("--holdout-frac", type=float, default=0.0)
    p.add_argument("--holdout-seed", type=int, default=17)
    p.add_argument("--padding-km", type=float, default=30.0)
    p.add_argument("--oversample", type=float, default=3.0)
    p.add_argument("--grid-nx", type=int, default=140)
    p.add_argument("--grid-ny", type=int, default=140)
    p.add_argument("--n-steps", type=int, default=30000)
    p.add_argument("--burn", type=int, default=6000)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--n-init", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise-std", type=float, default=None)
    p.add_argument("--noise-frac", type=float, default=0.02)
    p.add_argument("--vmin", type=float, default=1.2)
    p.add_argument("--vmax", type=float, default=5.6)
    p.add_argument("--sig-s", type=float, default=0.006)
    p.add_argument("--sig-s2", type=float, default=0.012)
    p.add_argument("--sig-c", type=float, default=15.0)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=200)
    p.add_argument("--tail-ensemble", type=int, default=60)
    p.add_argument("--save-coverage", action="store_true")
    return p.parse_args()


def required_columns_present(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def latlon_to_local_km(lats: np.ndarray, lons: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    R = 6371.0
    lat = np.deg2rad(lats)
    lon = np.deg2rad(lons)
    lat0r = math.radians(lat0)
    lon0r = math.radians(lon0)
    x = R * (lon - lon0r) * math.cos(lat0r)
    y = R * (lat - lat0r)
    return x, y


def load_ray_csv(path: str, period: Optional[float], min_snr: float, min_days: int, max_rays: Optional[int], seed: int, padding_km: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns_present(df, ["lat1", "lon1", "lat2", "lon2", "traveltime_s"])

    if period is not None and "period_s" in df.columns:
        df = df[np.isfinite(df["period_s"]) & np.isclose(df["period_s"], period)]
    if min_snr > 0 and "snr" in df.columns:
        df = df[np.isfinite(df["snr"]) & (df["snr"] >= min_snr)]
    if min_days > 0 and "days_used" in df.columns:
        df = df[np.isfinite(df["days_used"]) & (df["days_used"] >= min_days)]

    for c in ["lat1", "lon1", "lat2", "lon2", "traveltime_s"]:
        df = df[np.isfinite(df[c])]
    df = df[df["traveltime_s"] > 0].copy()

    if df.empty:
        raise ValueError("No rays left after filtering")

    lat0 = float(pd.concat([df["lat1"], df["lat2"]]).mean())
    lon0 = float(pd.concat([df["lon1"], df["lon2"]]).mean())
    x1, y1 = latlon_to_local_km(df["lat1"].to_numpy(), df["lon1"].to_numpy(), lat0, lon0)
    x2, y2 = latlon_to_local_km(df["lat2"].to_numpy(), df["lon2"].to_numpy(), lat0, lon0)

    df["x1_km"] = x1
    df["y1_km"] = y1
    df["x2_km"] = x2
    df["y2_km"] = y2

    xmin = min(df["x1_km"].min(), df["x2_km"].min()) - padding_km
    ymin = min(df["y1_km"].min(), df["y2_km"].min()) - padding_km
    df["x1_km"] -= xmin
    df["x2_km"] -= xmin
    df["y1_km"] -= ymin
    df["y2_km"] -= ymin

    dist_proj = np.hypot(df["x2_km"] - df["x1_km"], df["y2_km"] - df["y1_km"])
    df["dist_proj_km"] = dist_proj
    if "dist_km" not in df.columns:
        df["dist_km"] = dist_proj

    df["s_obs"] = df["traveltime_s"] / df["dist_proj_km"].replace(0, np.nan)
    df = df[np.isfinite(df["s_obs"]) & (df["dist_proj_km"] > 0)].copy()

    if max_rays is not None and len(df) > max_rays:
        df = df.sample(max_rays, random_state=seed).sort_index().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def split_holdout(df: pd.DataFrame, frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frac <= 0:
        return df.copy(), df.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < frac
    if mask.sum() == 0:
        mask[rng.integers(len(df))] = True
    if mask.sum() == len(df):
        mask[rng.integers(len(df))] = False
    return df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)


def build_ray_sampling(df: pd.DataFrame, oversample: float) -> Dict[str, np.ndarray]:
    pts_list = []
    segl_list = []
    for row in df.itertuples(index=False):
        sx, sy = float(row.x1_km), float(row.y1_km)
        rx, ry = float(row.x2_km), float(row.y2_km)
        ray_len = math.hypot(rx - sx, ry - sy)
        n_steps = max(int(ray_len * oversample), 8)
        ts = np.linspace(0.0, 1.0, n_steps + 1)
        ts_mid = 0.5 * (ts[:-1] + ts[1:])
        xm = sx + ts_mid * (rx - sx)
        ym = sy + ts_mid * (ry - sy)
        pts_list.append(np.column_stack([xm, ym]))
        segl_list.append(ray_len / n_steps)

    all_pts_flat = np.vstack(pts_list)
    ray_pt_counts = np.array([len(p) for p in pts_list], dtype=np.int32)
    ray_indices = np.repeat(np.arange(len(pts_list), dtype=np.int32), ray_pt_counts)
    seg_lens_arr = np.repeat(np.asarray(segl_list, dtype=np.float64), ray_pt_counts)
    return {
        "pts_list": pts_list,
        "all_pts_flat": all_pts_flat,
        "ray_pt_counts": ray_pt_counts,
        "ray_indices": ray_indices,
        "seg_lens_arr": seg_lens_arr,
    }


def _cell_assignments(nuclei: np.ndarray) -> np.ndarray:
    global _cache_key, _cache_asgn
    key = nuclei.tobytes()
    if key != _cache_key:
        tree = cKDTree(nuclei)
        _, asgn = tree.query(DATA["all_pts_flat"])
        _cache_key = key
        _cache_asgn = asgn
    return _cache_asgn


def forward_model(nuclei: np.ndarray, s_vals: np.ndarray) -> np.ndarray:
    asgn = _cell_assignments(nuclei)
    s_at_pts = s_vals[asgn]
    return np.bincount(
        DATA["ray_indices"],
        weights=s_at_pts * DATA["seg_lens_arr"],
        minlength=DATA["n_rays"],
    )


def predict_on_df(df: pd.DataFrame, nuclei: np.ndarray, s_vals: np.ndarray, oversample: float) -> np.ndarray:
    geom = build_ray_sampling(df, oversample=oversample)
    tree = cKDTree(nuclei)
    _, asgn = tree.query(geom["all_pts_flat"])
    s_at_pts = s_vals[asgn]
    return np.bincount(geom["ray_indices"], weights=s_at_pts * geom["seg_lens_arr"], minlength=len(df))


def log_likelihood(t_pred: np.ndarray) -> float:
    return -0.5 * np.sum(((t_pred - DATA["d_obs"]) / DATA["noise_std"]) ** 2)


def slowness_at(x: float, y: float, nuclei: np.ndarray, s_vals: np.ndarray) -> float:
    d2 = (nuclei[:, 0] - x) ** 2 + (nuclei[:, 1] - y) ** 2
    return float(s_vals[np.argmin(d2)])


def move_slowness(nuclei, s_vals, logL, sigma):
    k = int(RNG.integers(len(s_vals)))
    ds = float(RNG.normal(0.0, sigma))
    s_new = s_vals[k] + ds
    if not (DATA["S_MIN"] <= s_new <= DATA["S_MAX"]):
        return nuclei, s_vals, logL, False
    s_prop = s_vals.copy()
    s_prop[k] = s_new
    t_prop = forward_model(nuclei, s_prop)
    logL_prop = log_likelihood(t_prop)
    if np.log(RNG.random() + 1e-300) < logL_prop - logL:
        return nuclei, s_prop, logL_prop, True
    return nuclei, s_vals, logL, False


def move_nucleus(nuclei, s_vals, logL, sigma):
    k = int(RNG.integers(len(nuclei)))
    new_pos = nuclei[k] + RNG.normal(0.0, sigma, 2)
    if not (0.0 < new_pos[0] < DATA["Lx"] and 0.0 < new_pos[1] < DATA["Ly"]):
        return nuclei, s_vals, logL, False
    nuc_prop = nuclei.copy()
    nuc_prop[k] = new_pos
    t_prop = forward_model(nuc_prop, s_vals)
    logL_prop = log_likelihood(t_prop)
    if np.log(RNG.random() + 1e-300) < logL_prop - logL:
        return nuc_prop, s_vals, logL_prop, True
    return nuclei, s_vals, logL, False


def move_birth(nuclei, s_vals, logL):
    n = len(nuclei)
    if n >= DATA["N_MAX"]:
        return nuclei, s_vals, logL, False
    cx = float(RNG.uniform(0.0, DATA["Lx"]))
    cy = float(RNG.uniform(0.0, DATA["Ly"]))
    s_local = slowness_at(cx, cy, nuclei, s_vals)
    u_v = float(RNG.normal(0.0, DATA["SIG_S2"]))
    s_birth = s_local + u_v
    if not (DATA["S_MIN"] <= s_birth <= DATA["S_MAX"]):
        return nuclei, s_vals, logL, False
    nuc_prop = np.vstack([nuclei, [cx, cy]])
    s_prop = np.append(s_vals, s_birth)
    t_prop = forward_model(nuc_prop, s_prop)
    logL_prop = log_likelihood(t_prop)
    log_alpha = DATA["LOG_SIG2_SQRT2PI"] - np.log(DATA["DELTA_S"]) + u_v**2 / (2.0 * DATA["SIG_S2"]**2) + logL_prop - logL
    if np.log(RNG.random() + 1e-300) < log_alpha:
        return nuc_prop, s_prop, logL_prop, True
    return nuclei, s_vals, logL, False


def move_death(nuclei, s_vals, logL):
    n = len(nuclei)
    if n <= DATA["N_MIN"]:
        return nuclei, s_vals, logL, False
    k = int(RNG.integers(n))
    s_removed = float(s_vals[k])
    pos_removed = nuclei[k].copy()
    nuc_prop = np.delete(nuclei, k, axis=0)
    s_prop = np.delete(s_vals, k)
    t_prop = forward_model(nuc_prop, s_prop)
    logL_prop = log_likelihood(t_prop)
    s_after = slowness_at(pos_removed[0], pos_removed[1], nuc_prop, s_prop)
    log_alpha = np.log(DATA["DELTA_S"]) - DATA["LOG_SIG2_SQRT2PI"] - (s_removed - s_after) ** 2 / (2.0 * DATA["SIG_S2"]**2) + logL_prop - logL
    if np.log(RNG.random() + 1e-300) < log_alpha:
        return nuc_prop, s_prop, logL_prop, True
    return nuclei, s_vals, logL, False


def dr_slowness(nuclei, s_vals, logL):
    nuc, s, L, acc = move_slowness(nuclei, s_vals, logL, sigma=DATA["SIG_S"])
    if acc:
        return nuc, s, L, True
    return move_slowness(nuclei, s_vals, logL, sigma=DATA["SIG_S_DR"])


def dr_nucleus(nuclei, s_vals, logL):
    nuc, s, L, acc = move_nucleus(nuclei, s_vals, logL, sigma=DATA["SIG_C"])
    if acc:
        return nuc, s, L, True
    return move_nucleus(nuclei, s_vals, logL, sigma=DATA["SIG_C_DR"])


def rj_mcmc(n_steps: int, n_burn: int, thin: int, n_init: int, seed: int):
    global RNG
    RNG = np.random.default_rng(seed)

    nuclei = RNG.uniform(low=[0, 0], high=[DATA["Lx"], DATA["Ly"]], size=(n_init, 2))
    s_vals = np.clip(RNG.uniform(DATA["s0"] - 0.01, DATA["s0"] + 0.01, size=n_init), DATA["S_MIN"], DATA["S_MAX"])
    logL = log_likelihood(forward_model(nuclei, s_vals))

    ensemble = []
    n_cells_log = np.zeros(n_steps, dtype=np.int32)
    logL_log = np.zeros(n_steps, dtype=np.float64)
    acc = dict(s=0, move=0, birth=0, death=0)
    tot = dict(s=0, move=0, birth=0, death=0)

    t0 = time.time()
    pbar = tqdm(range(n_steps), desc="RJMCMC", total=n_steps)
    for step in pbar:
        if step % 2 == 0:
            tot["s"] += 1
            nuclei, s_vals, logL, a = dr_slowness(nuclei, s_vals, logL)
            if a:
                acc["s"] += 1
        else:
            choice = int(RNG.integers(3))
            if choice == 0:
                tot["birth"] += 1
                nuclei, s_vals, logL, a = move_birth(nuclei, s_vals, logL)
                if a:
                    acc["birth"] += 1
            elif choice == 1:
                tot["death"] += 1
                nuclei, s_vals, logL, a = move_death(nuclei, s_vals, logL)
                if a:
                    acc["death"] += 1
            else:
                tot["move"] += 1
                nuclei, s_vals, logL, a = dr_nucleus(nuclei, s_vals, logL)
                if a:
                    acc["move"] += 1

        n_cells_log[step] = len(nuclei)
        logL_log[step] = logL
        if step >= n_burn and (step - n_burn) % thin == 0:
            ensemble.append((nuclei.copy(), s_vals.copy()))
        if (step + 1) % max(50, n_steps // 200) == 0 or step == n_steps - 1:
            pbar.set_postfix(logL=f"{logL:.1f}", n_cells=len(nuclei))

    elapsed = time.time() - t0
    diag = dict(n_cells=n_cells_log, logL=logL_log, n_burn=n_burn, thin=thin, acc=acc, tot=tot, elapsed_s=elapsed)
    return ensemble, diag


def project_ensemble(ensemble: Sequence[Tuple[np.ndarray, np.ndarray]], nx: int, ny: int):
    xg = np.linspace(0.0, DATA["Lx"], nx)
    yg = np.linspace(0.0, DATA["Ly"], ny)
    XG, YG = np.meshgrid(xg, yg)
    query = np.column_stack([XG.ravel(), YG.ravel()])
    stack = np.zeros((len(ensemble), len(query)), dtype=np.float64)
    for i, (nuc, s) in enumerate(tqdm(ensemble, desc="Projecting", leave=False)):
        tree = cKDTree(nuc)
        _, idx = tree.query(query)
        stack[i] = s[idx]
    s_mean = stack.mean(axis=0).reshape(ny, nx)
    s_std = stack.std(axis=0).reshape(ny, nx)
    return xg, yg, XG, YG, s_mean, s_std


def save_diagnostics(diag: Dict, outpath: Path) -> None:
    n_steps = len(diag["logL"])
    burn = diag["n_burn"]
    steps = np.arange(n_steps)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(steps, diag["logL"], lw=0.5)
    axes[0, 0].axvline(burn, color="tab:red", ls="--", lw=1)
    axes[0, 0].set_title("Log-likelihood trace")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("log p(d|m)")

    axes[0, 1].plot(steps, diag["n_cells"], lw=0.5, color="tab:orange")
    axes[0, 1].axvline(burn, color="tab:red", ls="--", lw=1)
    axes[0, 1].set_title("Voronoi cell count trace")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("n cells")

    post_n = diag["n_cells"][burn:]
    bins = np.arange(max(1, int(post_n.min()) - 1), int(post_n.max()) + 2) - 0.5
    axes[1, 0].hist(post_n, bins=bins, density=True, color="tab:orange", edgecolor="black", alpha=0.8)
    axes[1, 0].set_title("Posterior on number of cells")
    axes[1, 0].set_xlabel("n cells")
    axes[1, 0].set_ylabel("Density")

    labels, vals = [], []
    for k in ("s", "move", "birth", "death"):
        if diag["tot"][k] > 0:
            labels.append(k)
            vals.append(diag["acc"][k] / diag["tot"][k] * 100)
    axes[1, 1].bar(labels, vals)
    axes[1, 1].set_title("Acceptance rates")
    axes[1, 1].set_ylabel("Percent")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def save_ray_coverage_local(df_train: pd.DataFrame, df_hold: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for row in df_train.itertuples(index=False):
        ax.plot([row.x1_km, row.x2_km], [row.y1_km, row.y2_km], color="tab:blue", alpha=0.25, lw=0.8)
    if len(df_hold) > 0:
        for row in df_hold.itertuples(index=False):
            ax.plot([row.x1_km, row.x2_km], [row.y1_km, row.y2_km], color="tab:orange", alpha=0.7, lw=1.0)
    ax.set_title("Ray coverage (local km domain)")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def save_ray_coverage_geo(df_train: pd.DataFrame, df_hold: pd.DataFrame, outpath: Path) -> None:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        return
    all_df = pd.concat([df_train, df_hold], ignore_index=True) if len(df_hold) else df_train
    minlon = float(min(all_df[["lon1", "lon2"]].min().min(), pd.concat([all_df["lon1"], all_df["lon2"]]).min()) - 1.0)
    maxlon = float(max(all_df[["lon1", "lon2"]].max().max(), pd.concat([all_df["lon1"], all_df["lon2"]]).max()) + 1.0)
    minlat = float(min(all_df[["lat1", "lat2"]].min().min(), pd.concat([all_df["lat1"], all_df["lat2"]]).min()) - 0.8)
    maxlat = float(max(all_df[["lat1", "lat2"]].max().max(), pd.concat([all_df["lat1"], all_df["lat2"]]).max()) + 0.8)

    fig = plt.figure(figsize=(9, 9))
    pc = ccrs.PlateCarree()
    proj = ccrs.LambertConformal(central_longitude=10.5, central_latitude=56.0)
    ax = plt.axes(projection=proj)
    ax.set_extent([minlon, maxlon, minlat, maxlat], crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#dfefff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6)
    for row in df_train.itertuples(index=False):
        ax.plot([row.lon1, row.lon2], [row.lat1, row.lat2], transform=pc, color="tab:blue", alpha=0.25, lw=0.8)
    if len(df_hold):
        for row in df_hold.itertuples(index=False):
            ax.plot([row.lon1, row.lon2], [row.lat1, row.lat2], transform=pc, color="tab:orange", alpha=0.8, lw=1.0)
    ax.set_title("Ray coverage (geographic)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close(fig)


def save_prediction_comparison(train_obs: np.ndarray, train_pred: np.ndarray, hold_obs: np.ndarray, hold_pred: np.ndarray, outpath: Path) -> Dict[str, float]:
    train_res = train_obs - train_pred
    train_rms = float(np.sqrt(np.mean(train_res**2)))
    hold_rms = float(np.sqrt(np.mean((hold_obs - hold_pred) ** 2))) if len(hold_obs) else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    obs_all = np.concatenate([train_obs, hold_obs]) if len(hold_obs) else train_obs
    pred_all = np.concatenate([train_pred, hold_pred]) if len(hold_obs) else train_pred
    lo = float(min(obs_all.min(), pred_all.min()))
    hi = float(max(obs_all.max(), pred_all.max()))

    axes[0].scatter(train_obs, train_pred, s=25, alpha=0.8, label=f"train (RMS={train_rms:.2f} s)")
    if len(hold_obs):
        axes[0].scatter(hold_obs, hold_pred, s=30, alpha=0.9, marker="s", label=f"holdout (RMS={hold_rms:.2f} s)")
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("Observed travel time [s]")
    axes[0].set_ylabel("Predicted travel time [s]")
    axes[0].set_title("Observed vs predicted")
    axes[0].legend()

    axes[1].hist(train_res, bins=20, alpha=0.8, label="train")
    if len(hold_obs):
        axes[1].hist(hold_obs - hold_pred, bins=20, alpha=0.7, label="holdout")
    axes[1].axvline(0.0, color="k", ls="--", lw=1)
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("Observed - predicted [s]")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return {"train_rms_s": train_rms, "holdout_rms_s": hold_rms}


def save_posterior_summary(df: pd.DataFrame, XG, YG, s_mean, s_std, residuals: np.ndarray, outpath: Path):
    v_mean = 1.0 / np.clip(s_mean, 1e-12, None)
    v_std = s_std / np.clip(s_mean, 1e-12, None) ** 2
    rms = float(np.sqrt(np.mean(residuals**2)))

    s_lo = max(DATA["S_MIN"], DATA["s0"] * 0.80)
    s_hi = min(DATA["S_MAX"], DATA["s0"] * 1.25)
    norm_s = TwoSlopeNorm(vmin=s_lo, vcenter=DATA["s0"], vmax=s_hi)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    im0 = axes[0, 0].pcolormesh(XG, YG, s_mean, shading="auto", cmap="RdBu_r", norm=norm_s)
    axes[0, 0].set_title("Posterior mean slowness [s/km]")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)

    im1 = axes[0, 1].pcolormesh(XG, YG, v_mean, shading="auto", cmap="viridis")
    axes[0, 1].set_title("Posterior mean velocity [km/s]")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)

    im2 = axes[1, 0].pcolormesh(XG, YG, v_std, shading="auto", cmap="magma")
    axes[1, 0].set_title("Posterior velocity uncertainty [km/s]")
    axes[1, 0].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.85)

    axes[1, 1].hist(residuals, bins=30, color="tab:blue", alpha=0.8, edgecolor="black")
    axes[1, 1].axvline(0.0, color="black", ls="--", lw=1)
    axes[1, 1].set_title(f"Training residuals, RMS = {rms:.3f} s")
    axes[1, 1].set_xlabel("Observed - predicted travel time [s]")
    axes[1, 1].set_ylabel("Count")

    for ax in axes.flat[:3]:
        for row in df.itertuples(index=False):
            ax.plot([row.x1_km, row.x2_km], [row.y1_km, row.y2_km], color="white", alpha=0.05, lw=0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return v_mean, v_std, rms


def draw_voronoi_model(ax, nuclei, s_vals, title="", norm=None, cmap="RdBu_r"):
    ng = 80
    xg = np.linspace(0, DATA["Lx"], ng)
    yg = np.linspace(0, DATA["Ly"], ng)
    XG, YG = np.meshgrid(xg, yg)
    tree = cKDTree(nuclei)
    _, idx = tree.query(np.column_stack([XG.ravel(), YG.ravel()]))
    s_map = s_vals[idx].reshape(ng, ng)
    ax.pcolormesh(XG, YG, s_map, cmap=cmap, norm=norm, shading="auto")
    ax.scatter(nuclei[:, 0], nuclei[:, 1], c="white", s=10, alpha=0.8)
    ax.set(xlim=(0, DATA["Lx"]), ylim=(0, DATA["Ly"]), aspect="equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelbottom=False, labelleft=False)


def save_gallery(ensemble: Sequence[Tuple[np.ndarray, np.ndarray]], outpath: Path) -> None:
    if len(ensemble) == 0:
        return
    n_show = min(9, len(ensemble))
    indices = np.linspace(0, len(ensemble) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = np.array(axes).reshape(-1)
    s_lo = max(DATA["S_MIN"], DATA["s0"] * 0.80)
    s_hi = min(DATA["S_MAX"], DATA["s0"] * 1.25)
    norm = TwoSlopeNorm(vmin=s_lo, vcenter=DATA["s0"], vmax=s_hi)
    for ax in axes[n_show:]:
        ax.axis("off")
    for ax, idx in zip(axes[:n_show], indices):
        nuc, s = ensemble[idx]
        draw_voronoi_model(ax, nuc, s, title=f"Sample {idx}", norm=norm)
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def posterior_tail_predictions(ensemble: Sequence[Tuple[np.ndarray, np.ndarray]], tail_ensemble: int) -> np.ndarray:
    use = ensemble[-min(tail_ensemble, len(ensemble)):]
    preds = np.empty((len(use), DATA["n_rays"]), dtype=np.float64)
    for i, (nuc, s) in enumerate(use):
        preds[i] = forward_model(nuc, s)
    return preds.mean(axis=0)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_all = load_ray_csv(args.csv, args.period, args.min_snr, args.min_days, args.max_rays, args.seed, args.padding_km)
    df_train, df_hold = split_holdout(df_all, args.holdout_frac, args.holdout_seed)

    geometry = build_ray_sampling(df_train, oversample=args.oversample)
    d_obs = df_train["traveltime_s"].to_numpy(dtype=np.float64)
    s_obs = df_train["s_obs"].to_numpy(dtype=np.float64)
    s0 = float(np.median(s_obs))
    noise_std = args.noise_std if args.noise_std is not None else max(float(np.median(d_obs) * args.noise_frac), 0.05)

    Lx = float(max(df_all["x1_km"].max(), df_all["x2_km"].max()) + args.padding_km)
    Ly = float(max(df_all["y1_km"].max(), df_all["y2_km"].max()) + args.padding_km)

    DATA.clear()
    DATA.update(geometry)
    DATA.update(dict(
        d_obs=d_obs,
        noise_std=noise_std,
        s0=s0,
        n_rays=len(df_train),
        Lx=Lx,
        Ly=Ly,
        S_MIN=1.0 / args.vmax,
        S_MAX=1.0 / args.vmin,
        DELTA_S=(1.0 / args.vmin) - (1.0 / args.vmax),
        N_MIN=args.n_min,
        N_MAX=args.n_max,
        SIG_S=args.sig_s,
        SIG_S2=args.sig_s2,
        SIG_C=args.sig_c,
        SIG_S_DR=args.sig_s * 0.25,
        SIG_C_DR=args.sig_c * 0.25,
        LOG_SIG2_SQRT2PI=float(np.log(args.sig_s2 * np.sqrt(2.0 * np.pi))),
    ))

    global _cache_key, _cache_asgn
    _cache_key = None
    _cache_asgn = None

    print("Loaded ray dataset")
    print(f"  Rays total      : {len(df_all)}")
    print(f"  Rays train      : {len(df_train)}")
    print(f"  Rays holdout    : {len(df_hold)}")
    print(f"  Domain          : {Lx:.1f} x {Ly:.1f} km")
    print(f"  Median velocity : {1.0 / s0:.3f} km/s")
    print(f"  Noise std       : {noise_std:.3f} s")
    print(f"  n_steps         : {args.n_steps:,}")

    ensemble, diag = rj_mcmc(args.n_steps, args.burn, args.thin, args.n_init, args.seed)
    if len(ensemble) == 0:
        raise RuntimeError("No posterior samples collected. Reduce burn-in or increase n_steps.")

    train_pred = posterior_tail_predictions(ensemble, args.tail_ensemble)
    train_res = d_obs - train_pred

    if len(df_hold):
        hold_preds = np.empty((min(args.tail_ensemble, len(ensemble)), len(df_hold)), dtype=np.float64)
        for i, (nuc, s) in enumerate(ensemble[-min(args.tail_ensemble, len(ensemble)):]):
            hold_preds[i] = predict_on_df(df_hold, nuc, s, args.oversample)
        hold_pred = hold_preds.mean(axis=0)
    else:
        hold_pred = np.empty(0, dtype=np.float64)

    print("Projecting ensemble to regular grid...")
    xg, yg, XG, YG, s_mean, s_std = project_ensemble(ensemble, nx=args.grid_nx, ny=args.grid_ny)

    df_train.to_csv(outdir / "filtered_rays_train.csv", index=False)
    df_hold.to_csv(outdir / "filtered_rays_holdout.csv", index=False)

    metrics = save_prediction_comparison(
        train_obs=d_obs,
        train_pred=train_pred,
        hold_obs=df_hold["traveltime_s"].to_numpy(dtype=np.float64) if len(df_hold) else np.empty(0),
        hold_pred=hold_pred,
        outpath=outdir / "prediction_comparison.png",
    )

    if args.save_coverage:
        save_ray_coverage_local(df_train, df_hold, outdir / "ray_coverage_local.png")
        save_ray_coverage_geo(df_train, df_hold, outdir / "ray_coverage_geo.png")

    v_mean, v_std, _ = save_posterior_summary(df_train, XG, YG, s_mean, s_std, train_res, outdir / "posterior_summary.png")
    save_diagnostics(diag, outdir / "diagnostics.png")
    save_gallery(ensemble, outdir / "voronoi_gallery.png")

    summary = {
        "input_csv": str(args.csv),
        "n_rays_total": int(len(df_all)),
        "n_rays_train": int(len(df_train)),
        "n_rays_holdout": int(len(df_hold)),
        "period_filter": args.period,
        "min_snr": args.min_snr,
        "min_days": args.min_days,
        "domain_km": {"Lx": Lx, "Ly": Ly},
        "background_slowness_s_per_km": s0,
        "background_velocity_km_per_s": 1.0 / s0,
        "noise_std_s": noise_std,
        "slowness_bounds_s_per_km": [DATA["S_MIN"], DATA["S_MAX"]],
        "velocity_bounds_km_per_s": [args.vmin, args.vmax],
        "n_steps": args.n_steps,
        "burn": args.burn,
        "thin": args.thin,
        "n_init": args.n_init,
        "seed": args.seed,
        "holdout_frac": args.holdout_frac,
        "train_rms_s": metrics["train_rms_s"],
        "holdout_rms_s": metrics["holdout_rms_s"],
        "acceptance_rates_pct": {k: (diag["acc"][k] / diag["tot"][k] * 100 if diag["tot"][k] else float("nan")) for k in diag["tot"]},
        "elapsed_s": float(diag["elapsed_s"]),
        "ensemble_size": int(len(ensemble)),
    }

    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        outdir / "posterior_grids.npz",
        xg=xg, yg=yg, XG=XG, YG=YG,
        s_mean=s_mean, s_std=s_std,
        v_mean=v_mean, v_std=v_std,
        n_cells=diag["n_cells"], logL=diag["logL"],
    )

    pd.DataFrame({
        "ray_index": np.arange(len(df_train)),
        "observed_t_s": d_obs,
        "predicted_t_s": train_pred,
        "residual_s": train_res,
    }).to_csv(outdir / "ray_predictions_train.csv", index=False)

    if len(df_hold):
        pd.DataFrame({
            "ray_index": np.arange(len(df_hold)),
            "observed_t_s": df_hold["traveltime_s"].to_numpy(dtype=np.float64),
            "predicted_t_s": hold_pred,
            "residual_s": df_hold["traveltime_s"].to_numpy(dtype=np.float64) - hold_pred,
        }).to_csv(outdir / "ray_predictions_holdout.csv", index=False)

    print("Saved outputs")
    print(f"  {outdir / 'filtered_rays_train.csv'}")
    print(f"  {outdir / 'filtered_rays_holdout.csv'}")
    print(f"  {outdir / 'run_summary.json'}")
    print(f"  {outdir / 'posterior_grids.npz'}")
    print(f"  {outdir / 'diagnostics.png'}")
    print(f"  {outdir / 'posterior_summary.png'}")
    print(f"  {outdir / 'prediction_comparison.png'}")
    if args.save_coverage:
        print(f"  {outdir / 'ray_coverage_local.png'}")
        print(f"  {outdir / 'ray_coverage_geo.png'}")
    print(f"  {outdir / 'voronoi_gallery.png'}")
    print(f"  {outdir / 'ray_predictions_train.csv'}")
    if len(df_hold):
        print(f"  {outdir / 'ray_predictions_holdout.csv'}")


if __name__ == "__main__":
    main()
