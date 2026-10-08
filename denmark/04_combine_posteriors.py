#!/usr/bin/env python3
"""
Combine the posterior grids of many independent RJMCMC chains.

Reads every completed chain folder written by 03_run_chains.py
(<chains-root>/chain_XXXX/posterior_grids.npz + run_summary.json), pools them
into one posterior mean / uncertainty, and writes the figures used in the report:

  diagnostics_combined.png             Figs 7-8  (traces + posteriors)
  posterior_summary_combined.png       Figs 9-10 (2x2 summary)
  geo_velocity_mean_combined.png       Fig 5 panels   (input to 05_make_paper_figures.py)
  geo_velocity_uncertainty_combined.png Fig 6 panels  (input to 05_make_paper_figures.py)
  geo_ray_coverage_combined.png        Fig 3 rays     (input to 05_make_paper_figures.py)

Pooling: each chain contributes in proportion to its ensemble size. With
per-chain mean m_c and std s_c the pooled moments are
  mean = sum w_c m_c,   var = sum w_c (s_c^2 + m_c^2) - mean^2.

Note: the surface-material maps used as background in Fig. 3 came from a
WFS-fetching helper that is no longer available, so they are not reproduced here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, MaxNLocator


LOCAL_FIG_DPI = 240
GEO_FIG_DPI = 260
STATION_COLOR = "#e11d48"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine posterior grids from many RJMCMC chains.")
    p.add_argument("--chains-root", required=True, help="Campaign folder containing chain_XXXX sub-folders")
    p.add_argument("--outdir", required=True)
    p.add_argument("--chain-glob", default="chain_*")
    p.add_argument("--velocity-cmap", default="viridis")
    p.add_argument("--uncertainty-cmap", default="magma")
    p.add_argument("--slowness-cmap", default="cividis")
    p.add_argument("--local-cmap", default="viridis")
    p.add_argument("--no-holdout", action="store_true", help="Do not draw holdout rays")
    p.add_argument("--station-labels", action="store_true", help="Label stations on geographic maps")
    return p.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _field_limits(field, lo_pct=1.0, hi_pct=99.0):
    """Robust colour limits so a few extreme grid points do not wash out the map."""
    vals = np.asarray(field)[np.isfinite(field)]
    if vals.size == 0:
        return None, None
    return float(np.percentile(vals, lo_pct)), float(np.percentile(vals, hi_pct))


# -------------------------
# Loading and combining
# -------------------------

def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_chain_dirs(chains_root: Path, chain_glob: str) -> list[Path]:
    """Chain folders that finished successfully (status complete, or no status file but a posterior grid)."""
    dirs = []
    for d in sorted(chains_root.glob(chain_glob)):
        if not d.is_dir() or not (d / "posterior_grids.npz").exists():
            continue
        status = _read_json(d / "chain_status.json")
        if status is not None and not status.get("complete", False):
            print(f"[skip] incomplete chain: {d.name}")
            continue
        dirs.append(d)
    if not dirs:
        raise FileNotFoundError(f"No completed chains matching '{chain_glob}' in {chains_root}")
    return dirs


def load_chain(chain_dir: Path) -> dict:
    with np.load(chain_dir / "posterior_grids.npz") as z:
        chain = {k: z[k] for k in z.files}
    summary = _read_json(chain_dir / "run_summary.json") or {}
    chain["burn"] = int(summary.get("burn", 0))
    chain["ensemble_size"] = int(summary.get("ensemble_size", 1))
    return chain


def assert_same_grid(a: dict, b: dict) -> None:
    for key in ("XG", "YG"):
        if a[key].shape != b[key].shape or not np.allclose(a[key], b[key]):
            raise ValueError(f"Chains were run on different model grids ({key} differs); they cannot be combined.")


def combine_mean_and_std(means, stds, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    m = np.stack(means)
    s = np.stack(stds)
    mean = np.tensordot(w, m, axes=1)
    second = np.tensordot(w, s**2 + m**2, axes=1)
    return mean, np.sqrt(np.clip(second - mean**2, 0.0, None))


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def extract_unique_stations_from_rays(*dfs) -> pd.DataFrame:
    """One row per station with both geographic and local-km coordinates."""
    rows = []
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        for end in ("1", "2"):
            part = pd.DataFrame(
                {
                    "station_id": df[f"net{end}"].astype(str) + "." + df[f"sta{end}"].astype(str),
                    "lat": df[f"lat{end}"],
                    "lon": df[f"lon{end}"],
                    "x_km": df[f"x{end}_km"],
                    "y_km": df[f"y{end}_km"],
                }
            )
            rows.append(part)
    stations = pd.concat(rows, ignore_index=True)
    return stations.drop_duplicates("station_id").sort_values("station_id").reset_index(drop=True)


# -------------------------
# Local km <-> lon/lat
# -------------------------
# 02_voronoi_rjmcmc.py projects lon/lat to km with an equirectangular projection
# plus a shift, which is affine, so a least-squares fit on the stations recovers it exactly.

def fit_affine_xy_to_lonlat(stations_df: pd.DataFrame) -> np.ndarray:
    A = np.column_stack([stations_df["x_km"], stations_df["y_km"], np.ones(len(stations_df))])
    coef, *_ = np.linalg.lstsq(A, stations_df[["lon", "lat"]].to_numpy(), rcond=None)
    return coef  # (3, 2)


def local_km_to_lonlat(x, y, transform: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = transform[0, 0] * x + transform[1, 0] * y + transform[2, 0]
    lat = transform[0, 1] * x + transform[1, 1] * y + transform[2, 1]
    return lon, lat


def compute_geo_context(stations_df, transform, XG, YG):
    LON, LAT = local_km_to_lonlat(XG, YG, transform)
    extent = [float(LON.min()), float(LON.max()), float(LAT.min()), float(LAT.max())]
    return LON, LAT, stations_df, extent


# -------------------------
# Generic styling helpers
# -------------------------

def style_local_axes(ax, XG, YG, xlabel="x [km]", ylabel="y [km]"):
    xmin, xmax = float(np.nanmin(XG)), float(np.nanmax(XG))
    ymin, ymax = float(np.nanmin(YG)), float(np.nanmax(YG))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xspan = xmax - xmin
    yspan = ymax - ymin
    xmajor = 100 if xspan > 500 else 50
    ymajor = 100 if yspan > 500 else 50
    xminor = xmajor / 2
    yminor = ymajor / 2

    ax.xaxis.set_major_locator(MultipleLocator(xmajor))
    ax.yaxis.set_major_locator(MultipleLocator(ymajor))
    ax.xaxis.set_minor_locator(MultipleLocator(xminor))
    ax.yaxis.set_minor_locator(MultipleLocator(yminor))

    ax.grid(True, which="major", color="#9aa0a6", alpha=0.34, linestyle=(0, (2.5, 3.5)), linewidth=0.9)
    ax.grid(True, which="minor", color="#c7cbd1", alpha=0.20, linestyle=(0, (1.5, 3.0)), linewidth=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#666")


def add_local_station_markers(ax, stations_df: pd.DataFrame | None):
    if stations_df is None or len(stations_df) == 0:
        return
    ax.scatter(
        stations_df["x_km"],
        stations_df["y_km"],
        s=58,
        color=STATION_COLOR,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    for _, row in stations_df.iterrows():
        sid = row["station_id"].split(".")[-1]
        txt = ax.text(row["x_km"] + 7.0, row["y_km"] + 5.0, sid, fontsize=9.5, color="#1f2937", zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])


def add_standard_geo_background(ax):
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.OCEAN, facecolor="#dbeafe", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f3f4f6", zorder=0)
    # Coastlines and borders sit above the tomography field so they stay readable.
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#111827", zorder=8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#374151", alpha=0.7, zorder=8)


def add_gridlines(ax, extent):
    import cartopy.crs as ccrs

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6, color="#6b7280", alpha=0.35, linestyle="--", zorder=9)
    gl.top_labels = False
    gl.right_labels = False
    return gl


def plot_station_markers(ax, stations_df: pd.DataFrame | None, station_labels=False):
    import cartopy.crs as ccrs

    if stations_df is None or len(stations_df) == 0:
        return
    ax.scatter(
        stations_df["lon"], stations_df["lat"], s=52, color=STATION_COLOR, edgecolor="white",
        linewidth=1.0, transform=ccrs.PlateCarree(), zorder=10,
    )
    if station_labels:
        for _, row in stations_df.iterrows():
            txt = ax.text(row["lon"] + 0.08, row["lat"] + 0.06, row["station_id"].split(".")[-1], fontsize=8.5,
                          color="#1f2937", transform=ccrs.PlateCarree(), zorder=11)
            txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])


# -------------------------
# Diagnostics plots
# -------------------------

def plot_diagnostics_traces(logL_list, n_cells_list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), sharex=False)

    for arr in logL_list:
        if arr is None or len(arr) == 0:
            continue
        axes[0].plot(arr, linewidth=1.0, alpha=0.78)
    axes[0].set_title("Log-likelihood traces")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("log p(d|m)")
    axes[0].grid(True, color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))
    axes[0].xaxis.set_major_locator(MaxNLocator(6))

    for arr in n_cells_list:
        if arr is None or len(arr) == 0:
            continue
        axes[1].plot(arr, linewidth=1.0, alpha=0.78)
    axes[1].set_title("Voronoi cell-count traces")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("n cells")
    axes[1].grid(True, color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))
    axes[1].xaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_diagnostics_posteriors(logL_post, n_cells_post, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    axes[0].hist(logL_post, bins=28, density=True, color="#0ea5b7", alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[0].set_title("Posterior on log-likelihood")
    axes[0].set_xlabel("log p(d|m)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    axes[1].hist(n_cells_post, bins=24, density=True, color="#5b5ce2", alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[1].set_title("Posterior on number of Voronoi cells")
    axes[1].set_xlabel("n cells")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_diagnostics(logL_list, n_cells_list, logL_post, n_cells_post, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8))

    for arr in logL_list:
        if arr is None or len(arr) == 0:
            continue
        axes[0, 0].plot(arr, linewidth=0.95, alpha=0.75)
    axes[0, 0].set_title("Log-likelihood traces")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("log p(d|m)")
    axes[0, 0].grid(True, color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    for arr in n_cells_list:
        if arr is None or len(arr) == 0:
            continue
        axes[0, 1].plot(arr, linewidth=0.95, alpha=0.75)
    axes[0, 1].set_title("Voronoi cell-count traces")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("n cells")
    axes[0, 1].grid(True, color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    axes[1, 0].hist(n_cells_post, bins=24, density=True, color="#5b5ce2", alpha=0.82, edgecolor="white", linewidth=0.35)
    axes[1, 0].set_title("Posterior on number of Voronoi cells")
    axes[1, 0].set_xlabel("n cells")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    axes[1, 1].hist(logL_post, bins=28, density=True, color="#0ea5b7", alpha=0.82, edgecolor="white", linewidth=0.35)
    axes[1, 1].set_title("Posterior on log-likelihood")
    axes[1, 1].set_xlabel("log p(d|m)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_summary(v_mean, v_std, s_mean, n_cells_post, XG, YG, out_path: Path,
                          velocity_cmap="viridis", uncertainty_cmap="magma", slowness_cmap="cividis"):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 11.0))
    panels = [
        (axes[0, 0], s_mean, slowness_cmap, "Combined posterior mean slowness [s/km]"),
        (axes[0, 1], v_mean, velocity_cmap, "Combined posterior mean velocity [km/s]"),
        (axes[1, 0], v_std, uncertainty_cmap, "Combined velocity uncertainty [km/s]"),
    ]
    for ax, field, cmap, title in panels:
        vmin, vmax = _field_limits(field)
        pcm = ax.pcolormesh(XG, YG, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        style_local_axes(ax, XG, YG)
        ax.set_title(title, fontsize=15)
        fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)

    ax = axes[1, 1]
    ax.hist(n_cells_post, bins=24, color="#5b5ce2", alpha=0.85, edgecolor="white", linewidth=0.35)
    ax.set_title("Combined posterior on number of cells", fontsize=15)
    ax.set_xlabel("n cells")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------
# Local-coordinate plots
# -------------------------

def plot_local_field_clean(field, XG, YG, stations_df, out_path: Path, title: str, cbar_label: str, cmap="viridis"):
    vmin, vmax = _field_limits(field)
    fig, ax = plt.subplots(figsize=(10.2, 7.9))
    pcm = ax.pcolormesh(XG, YG, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    style_local_axes(ax, XG, YG)
    add_local_station_markers(ax, stations_df)
    ax.set_title(title)
    cb = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_local_ray_coverage(train_df, holdout_df, stations_df, XG, YG, out_path: Path):
    fig, ax = plt.subplots(figsize=(10.2, 7.9))
    ax.set_facecolor("#f8fafc")
    style_local_axes(ax, XG, YG)

    if train_df is not None and len(train_df) > 0:
        for _, row in train_df.iterrows():
            ax.plot([row["x1_km"], row["x2_km"]], [row["y1_km"], row["y2_km"]], color="#2563eb", alpha=0.10, linewidth=0.75, zorder=2)
    if holdout_df is not None and len(holdout_df) > 0:
        for _, row in holdout_df.iterrows():
            ax.plot([row["x1_km"], row["x2_km"]], [row["y1_km"], row["y2_km"]], color="#f59e0b", alpha=0.16, linewidth=0.95, zorder=3)

    add_local_station_markers(ax, stations_df)
    legend_handles = [
        Line2D([0], [0], color="#2563eb", lw=1.8, alpha=0.60, label="train rays"),
        Line2D([0], [0], color="#f59e0b", lw=1.8, alpha=0.65, label="holdout rays"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STATION_COLOR, markeredgecolor="white", markersize=9, label="stations"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=True, framealpha=0.95)
    ax.set_title("Ray coverage (local coordinates)")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------
# Geo plots
# -------------------------

def plot_geo_single_field(field, stations_df, transform, XG, YG, out_path: Path, title: str, cbar_label: str, station_labels=False, cmap="viridis"):
    import cartopy.crs as ccrs

    LON, LAT, stations, extent = compute_geo_context(stations_df, transform, XG, YG)
    vmin, vmax = _field_limits(field)

    fig = plt.figure(figsize=(11.2, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    add_standard_geo_background(ax)
    add_gridlines(ax, extent)

    pcm = ax.pcolormesh(LON, LAT, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), zorder=5)
    plot_station_markers(ax, stations, station_labels=station_labels)
    ax.set_title(title)

    cb = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, fraction=0.045)
    cb.set_label(cbar_label)
    cb.ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(out_path, dpi=GEO_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_geo_ray_coverage(train_df, holdout_df, stations_df, transform, XG, YG, out_path: Path, station_labels=False):
    import cartopy.crs as ccrs

    _, _, stations, extent = compute_geo_context(stations_df, transform, XG, YG)
    fig = plt.figure(figsize=(11.2, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    add_standard_geo_background(ax)
    add_gridlines(ax, extent)

    if train_df is not None and len(train_df) > 0:
        for _, row in train_df.iterrows():
            ax.plot([row["lon1"], row["lon2"]], [row["lat1"], row["lat2"]], color="#2563eb", alpha=0.09, linewidth=0.70, transform=ccrs.PlateCarree(), zorder=3)
    if holdout_df is not None and len(holdout_df) > 0:
        for _, row in holdout_df.iterrows():
            ax.plot([row["lon1"], row["lon2"]], [row["lat1"], row["lat2"]], color="#f59e0b", alpha=0.16, linewidth=0.85, transform=ccrs.PlateCarree(), zorder=4)

    plot_station_markers(ax, stations, station_labels=station_labels)
    handles = [
        Line2D([0], [0], color="#2563eb", lw=1.8, alpha=0.55, label="train rays"),
        Line2D([0], [0], color="#f59e0b", lw=1.8, alpha=0.60, label="holdout rays"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STATION_COLOR, markeredgecolor="white", markersize=10, label="stations"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, framealpha=0.95)
    ax.set_title("Ray coverage")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=GEO_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------
# Main
# -------------------------

def main():
    apply_style()
    args = parse_args()

    chains_root = Path(args.chains_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chain_dirs = find_chain_dirs(chains_root, args.chain_glob)
    print(f"Found {len(chain_dirs)} completed chain folders.")

    chains = []
    meta_rows = []
    for chain_dir in chain_dirs:
        chain = load_chain(chain_dir)
        if chains:
            assert_same_grid(chains[0], chain)
        chains.append(chain)

        row = {"chain_dir": str(chain_dir.resolve())}
        for name in ("chain_status", "chain_meta", "run_summary"):
            block = _read_json(chain_dir / f"{name}.json") or {}
            for k, v in block.items():
                row[f"{name}.{k}"] = json.dumps(v) if isinstance(v, (dict, list)) else v
        meta_rows.append(row)

    ref = chains[0]
    XG, YG, xg, yg = ref["XG"], ref["YG"], ref["xg"], ref["yg"]
    weights = [c["ensemble_size"] for c in chains]

    v_mean, v_std = combine_mean_and_std([c["v_mean"] for c in chains], [c["v_std"] for c in chains], weights)
    s_mean, s_std = combine_mean_and_std([c["s_mean"] for c in chains], [c["s_std"] for c in chains], weights)

    # Full traces for the trace plots, post-burn-in samples for the posterior histograms.
    logL_list = [c["logL"] for c in chains]
    n_cells_list = [c["n_cells"] for c in chains]
    logL_post = np.concatenate([c["logL"][c["burn"]:] for c in chains])
    n_cells_post = np.concatenate([c["n_cells"][c["burn"]:] for c in chains])

    np.savez(
        outdir / "posterior_grids_combined.npz",
        xg=xg, yg=yg, XG=XG, YG=YG,
        s_mean=s_mean, s_std=s_std,
        v_mean=v_mean, v_std=v_std,
        n_cells=n_cells_post, logL=logL_post,
    )
    print(f"Saved: {outdir / 'posterior_grids_combined.npz'}")

    summary = {
        "n_chains_used": len(chain_dirs),
        "chains_root": str(chains_root.resolve()),
        "chain_glob": args.chain_glob,
        "grid_shape": list(XG.shape),
        "x_range_km": [float(np.nanmin(XG)), float(np.nanmax(XG))],
        "y_range_km": [float(np.nanmin(YG)), float(np.nanmax(YG))],
        "v_mean_range_km_s": [float(np.nanmin(v_mean)), float(np.nanmax(v_mean))],
        "v_std_range_km_s": [float(np.nanmin(v_std)), float(np.nanmax(v_std))],
        "s_mean_range_s_km": [float(np.nanmin(s_mean)), float(np.nanmax(s_mean))],
        "s_std_range_s_km": [float(np.nanmin(s_std)), float(np.nanmax(s_std))],
        "n_cells_posterior_median": float(np.median(n_cells_post)),
    }
    (outdir / "combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {outdir / 'combined_summary.json'}")
    pd.DataFrame(meta_rows).to_csv(outdir / "combined_chain_metadata.csv", index=False)
    print(f"Saved: {outdir / 'combined_chain_metadata.csv'}")

    # Diagnostics and summary (local coordinates)
    plot_combined_diagnostics(logL_list, n_cells_list, logL_post, n_cells_post, outdir / "diagnostics_combined.png")
    plot_diagnostics_traces(logL_list, n_cells_list, outdir / "diagnostics_traces_combined.png")
    plot_diagnostics_posteriors(logL_post, n_cells_post, outdir / "diagnostics_posteriors_combined.png")
    plot_combined_summary(
        v_mean, v_std, s_mean, n_cells_post, XG, YG, outdir / "posterior_summary_combined.png",
        velocity_cmap=args.velocity_cmap, uncertainty_cmap=args.uncertainty_cmap, slowness_cmap=args.slowness_cmap,
    )

    first_chain = chain_dirs[0]
    train_df = load_optional_csv(first_chain / "filtered_rays_train.csv")
    holdout_df = None if args.no_holdout else load_optional_csv(first_chain / "filtered_rays_holdout.csv")
    if train_df is None:
        print("[warn] filtered_rays_train.csv not found or empty; skipping station / ray / geographic plots.")
        return
    stations_df = extract_unique_stations_from_rays(train_df, holdout_df)

    plot_local_field_clean(v_mean, XG, YG, stations_df, outdir / "velocity_mean_local_combined.png", "Posterior mean velocity (local coordinates)", "Velocity [km/s]", cmap=args.local_cmap)
    plot_local_field_clean(v_std, XG, YG, stations_df, outdir / "velocity_uncertainty_local_combined.png", "Posterior velocity uncertainty (local coordinates)", "Velocity uncertainty [km/s]", cmap=args.uncertainty_cmap)
    plot_local_ray_coverage(train_df, holdout_df, stations_df, XG, YG, outdir / "ray_coverage_local_combined.png")

    try:
        import cartopy  # noqa: F401
    except ImportError:
        print("[warn] cartopy is not installed; skipping geographic plots.")
        return

    transform = fit_affine_xy_to_lonlat(stations_df)
    plot_geo_single_field(v_mean, stations_df, transform, XG, YG, outdir / "geo_velocity_mean_combined.png", "Posterior mean velocity", "Velocity [km/s]", station_labels=args.station_labels, cmap=args.velocity_cmap)
    plot_geo_single_field(v_std, stations_df, transform, XG, YG, outdir / "geo_velocity_uncertainty_combined.png", "Posterior velocity uncertainty", "Velocity uncertainty [km/s]", station_labels=args.station_labels, cmap=args.uncertainty_cmap)
    plot_geo_ray_coverage(train_df, holdout_df, stations_df, transform, XG, YG, outdir / "geo_ray_coverage_combined.png", station_labels=args.station_labels)


if __name__ == "__main__":
    main()
