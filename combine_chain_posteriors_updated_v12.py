from __future__ import annotations

import json
import math
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator

# Load v9 as a module so we can reuse all the data-loading / WFS / cartopy helpers
HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "combine_chain_posteriors_updated_v9.py"
spec = importlib.util.spec_from_file_location("combine_v9_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)


LOCAL_FIG_DPI = 240
GEO_FIG_DPI = 260

MATERIAL_ORDER = [
    "Sand",
    "Mud / clay / silt",
    "Gravel / coarse sediment",
    "Mixed unconsolidated / glacial",
    "Carbonate / chalk",
    "Crystalline / hard rock",
    "Sedimentary rock",
    "No classification",
]
PLOT_MATERIAL_ORDER = [m for m in MATERIAL_ORDER if m != "No classification"]
MATERIAL_COLORS = dict(base.UNIFIED_MATERIAL_CLASSES)


def apply_extra_style() -> None:
    base.apply_publication_style()
    plt.rcParams.update(
        {
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
        color="#e11d48",
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    for _, row in stations_df.iterrows():
        sid = row["station_id"].split(".")[-1]
        txt = ax.text(row["x_km"] + 7.0, row["y_km"] + 5.0, sid, fontsize=9.5, color="#1f2937", zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])


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



def plot_diagnostics_posteriors(logL_list, n_cells_list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    ll_arrays = [a for a in logL_list if a is not None and len(a) > 0]
    nc_arrays = [a for a in n_cells_list if a is not None and len(a) > 0]

    if ll_arrays:
        ll_all = np.concatenate(ll_arrays)
        axes[0].hist(ll_all, bins=28, density=True, color="#0ea5b7", alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[0].set_title("Posterior on log-likelihood")
    axes[0].set_xlabel("log p(d|m)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    if nc_arrays:
        nc_all = np.concatenate(nc_arrays)
        axes[1].hist(nc_all, bins=24, density=True, color="#5b5ce2", alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[1].set_title("Posterior on number of Voronoi cells")
    axes[1].set_xlabel("n cells")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def plot_combined_diagnostics_nice(logL_list, n_cells_list, out_path: Path):
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

    ll_arrays = [a for a in logL_list if a is not None and len(a) > 0]
    nc_arrays = [a for a in n_cells_list if a is not None and len(a) > 0]
    if nc_arrays:
        nc_all = np.concatenate(nc_arrays)
        axes[1, 0].hist(nc_all, bins=24, density=True, color="#5b5ce2", alpha=0.82, edgecolor="white", linewidth=0.35)
    axes[1, 0].set_title("Posterior on number of Voronoi cells")
    axes[1, 0].set_xlabel("n cells")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    if ll_arrays:
        ll_all = np.concatenate(ll_arrays)
        axes[1, 1].hist(ll_all, bins=28, density=True, color="#0ea5b7", alpha=0.82, edgecolor="white", linewidth=0.35)
    axes[1, 1].set_title("Posterior on log-likelihood")
    axes[1, 1].set_xlabel("log p(d|m)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(True, axis="y", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------
# Local-coordinate plots
# -------------------------

def plot_local_field_clean(field, XG, YG, stations_df, out_path: Path, title: str, cbar_label: str, cmap="viridis"):
    vmin, vmax = base._field_limits(field)
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
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e11d48", markeredgecolor="white", markersize=9, label="stations"),
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
    try:
        import cartopy.crs as ccrs
    except Exception:
        print("[warn] cartopy is not installed; skipping geographic plot.")
        return

    LON, LAT, stations, extent = base.compute_geo_context(stations_df, transform, XG, YG)
    vmin, vmax = base._field_limits(field)

    fig = plt.figure(figsize=(11.2, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    base.add_standard_geo_background(ax)
    base.add_gridlines(ax, extent)

    pcm = ax.pcolormesh(LON, LAT, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), zorder=5)
    base.plot_station_markers(ax, stations, station_labels=station_labels)
    ax.set_title(title)

    cb = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, fraction=0.045)
    cb.set_label(cbar_label)
    cb.ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(out_path, dpi=GEO_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_geo_ray_coverage(train_df, holdout_df, stations_df, transform, XG, YG, out_path: Path, station_labels=False):
    try:
        import cartopy.crs as ccrs
    except Exception:
        print("[warn] cartopy is not installed; skipping geographic ray-coverage plot.")
        return

    _, _, stations, extent = base.compute_geo_context(stations_df, transform, XG, YG)
    fig = plt.figure(figsize=(11.2, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    base.add_standard_geo_background(ax)
    base.add_gridlines(ax, extent)

    if train_df is not None and len(train_df) > 0:
        for _, row in train_df.iterrows():
            lon1, lat1 = base.local_km_to_lonlat(row["x1_km"], row["y1_km"], transform)
            lon2, lat2 = base.local_km_to_lonlat(row["x2_km"], row["y2_km"], transform)
            ax.plot([lon1, lon2], [lat1, lat2], color="#2563eb", alpha=0.09, linewidth=0.70, transform=ccrs.PlateCarree(), zorder=3)
    if holdout_df is not None and len(holdout_df) > 0:
        for _, row in holdout_df.iterrows():
            lon1, lat1 = base.local_km_to_lonlat(row["x1_km"], row["y1_km"], transform)
            lon2, lat2 = base.local_km_to_lonlat(row["x2_km"], row["y2_km"], transform)
            ax.plot([lon1, lon2], [lat1, lat2], color="#f59e0b", alpha=0.16, linewidth=0.85, transform=ccrs.PlateCarree(), zorder=4)

    base.plot_station_markers(ax, stations, station_labels=station_labels)
    handles = [
        Line2D([0], [0], color="#2563eb", lw=1.8, alpha=0.55, label="train rays"),
        Line2D([0], [0], color="#f59e0b", lw=1.8, alpha=0.60, label="holdout rays"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e11d48", markeredgecolor="white", markersize=10, label="stations"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, framealpha=0.95)
    ax.set_title("Ray coverage")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=GEO_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _material_legend_handles(include_no_classification=True):
    handles = []
    for label in MATERIAL_ORDER:
        if label == "No classification" and not include_no_classification:
            continue
        handles.append(Patch(facecolor=MATERIAL_COLORS[label], edgecolor="none", label=label))
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#e11d48", markeredgecolor="#1f2937", markersize=9.5, label="stations"))
    return handles


def save_surface_material_legend(out_path: Path):
    handles = _material_legend_handles(include_no_classification=True)
    fig, ax = plt.subplots(figsize=(12.5, 1.6))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=5, frameon=True, framealpha=0.96, columnspacing=1.1, handlelength=1.1, handletextpad=0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_geo_context_nice(materials, stations_df, transform, XG, YG, out_path: Path, station_labels=False):
    try:
        import cartopy.crs as ccrs
    except Exception:
        print("[warn] cartopy is not installed; skipping materials context plot.")
        return

    _, _, stations, extent = base.compute_geo_context(stations_df, transform, XG, YG)
    fig = plt.figure(figsize=(11.2, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    base.add_standard_geo_background(ax)
    base.add_harmonized_surface_materials(ax, materials, zorder_base=1, alpha=0.98)
    base.add_gridlines(ax, extent)
    base.plot_station_markers(ax, stations, station_labels=station_labels)
    ax.set_title("Harmonized surface materials")

    handles = _material_legend_handles(include_no_classification=True)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=True, framealpha=0.96, columnspacing=1.0, handlelength=1.0, handletextpad=0.5)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    fig.savefig(out_path, dpi=GEO_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_geo_overlay_nice(field, stations_df, transform, XG, YG, materials, out_path: Path, title: str, cbar_label: str, station_labels=False, overlay_alpha=0.24, cmap="viridis"):
    try:
        import cartopy.crs as ccrs
    except Exception:
        print("[warn] cartopy is not installed; skipping overlay plot.")
        return

    LON, LAT, stations, extent = base.compute_geo_context(stations_df, transform, XG, YG)
    vmin, vmax = base._field_limits(field)

    fig = plt.figure(figsize=(11.2, 8.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    base.add_standard_geo_background(ax)
    base.add_harmonized_surface_materials(ax, materials, zorder_base=1, alpha=0.98)
    base.add_gridlines(ax, extent)

    pcm = ax.pcolormesh(LON, LAT, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, alpha=overlay_alpha, transform=ccrs.PlateCarree(), zorder=7)
    base.plot_station_markers(ax, stations, station_labels=station_labels)
    ax.set_title(title)

    cb = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, fraction=0.045)
    cb.set_label(cbar_label)
    cb.ax.yaxis.set_major_locator(MaxNLocator(6))

    handles = _material_legend_handles(include_no_classification=True)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=True, framealpha=0.96, columnspacing=1.0, handlelength=1.0, handletextpad=0.5)

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    fig.savefig(out_path, dpi=GEO_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_geo_two_panel(v_mean, v_std, stations_df, transform, XG, YG, materials, out_path: Path, velocity_cmap="viridis", uncertainty_cmap="magma"):
    try:
        import cartopy.crs as ccrs
    except Exception:
        print("[warn] cartopy is not installed; skipping two-panel geo plot.")
        return

    LON, LAT, stations, extent = base.compute_geo_context(stations_df, transform, XG, YG)
    fig = plt.figure(figsize=(14.2, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    base.add_standard_geo_background(ax1)
    base.add_harmonized_surface_materials(ax1, materials, zorder_base=1, alpha=0.98)
    base.add_gridlines(ax1, extent)
    pcm1 = ax1.pcolormesh(LON, LAT, v_mean, shading="auto", cmap=velocity_cmap, vmin=np.nanmin(v_mean), vmax=np.nanmax(v_mean), transform=ccrs.PlateCarree(), alpha=0.22, zorder=7)
    base.plot_station_markers(ax1, stations, station_labels=False)
    ax1.set_title("Posterior mean velocity")

    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    base.add_standard_geo_background(ax2)
    base.add_gridlines(ax2, extent)
    pcm2 = ax2.pcolormesh(LON, LAT, v_std, shading="auto", cmap=uncertainty_cmap, vmin=np.nanmin(v_std), vmax=np.nanmax(v_std), transform=ccrs.PlateCarree(), zorder=7)
    base.plot_station_markers(ax2, stations, station_labels=False)
    ax2.set_title("Posterior velocity uncertainty")

    cb1 = fig.colorbar(pcm1, ax=ax1, orientation="horizontal", pad=0.08, fraction=0.050)
    cb1.set_label("Velocity [km/s]")
    cb1.ax.xaxis.set_major_locator(MaxNLocator(6))
    cb2 = fig.colorbar(pcm2, ax=ax2, orientation="horizontal", pad=0.08, fraction=0.050)
    cb2.set_label("Velocity uncertainty [km/s]")
    cb2.ax.xaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()
    fig.savefig(out_path, dpi=GEO_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -------------------------
# Grid-sampled material comparison plots
# -------------------------

def build_material_class_grid(LON, LAT, materials):
    try:
        from shapely.geometry import Point
        from shapely.prepared import prep
        from shapely.ops import unary_union
    except Exception:
        return np.full(LON.shape, "No classification", dtype=object)

    points = [Point(float(lo), float(la)) for lo, la in zip(LON.ravel(), LAT.ravel())]
    labels = np.full(len(points), "No classification", dtype=object)

    # rock classes first, then unconsolidated classes; this gives a stable priority when polygons overlap
    priority = [
        "Crystalline / hard rock",
        "Sedimentary rock",
        "Carbonate / chalk",
        "Gravel / coarse sediment",
        "Mud / clay / silt",
        "Sand",
        "Mixed unconsolidated / glacial",
    ]

    for cls in priority:
        geoms = [g for g in materials.get(cls, []) if getattr(g, "is_empty", True) is False]
        if not geoms:
            continue
        try:
            union = unary_union(geoms)
            prepared = prep(union)
        except Exception:
            continue
        undecided = np.where(labels == "No classification")[0]
        for idx in undecided:
            pt = points[idx]
            try:
                inside = prepared.contains(pt) or prepared.intersects(pt)
            except Exception:
                inside = False
            if inside:
                labels[idx] = cls

    return labels.reshape(LON.shape)



def material_stats_dataframe(class_grid, v_mean, s_mean, v_std):
    rows = []
    valid = np.isfinite(v_mean) & np.isfinite(s_mean) & np.isfinite(v_std)
    unc = np.where(v_std > 0, v_std, np.nan)

    for cls in MATERIAL_ORDER:
        mask = valid & (class_grid == cls)
        n = int(np.sum(mask))
        if n == 0:
            rows.append({"class": cls, "n_cells": 0})
            continue

        vv = v_mean[mask]
        ss = s_mean[mask]
        uu = v_std[mask]
        weights = 1.0 / np.maximum(uu, 0.05) ** 2
        rows.append(
            {
                "class": cls,
                "n_cells": n,
                "velocity_mean": float(np.nanmean(vv)),
                "velocity_median": float(np.nanmedian(vv)),
                "velocity_std": float(np.nanstd(vv)),
                "velocity_weighted_mean": float(np.average(vv, weights=weights)),
                "slowness_mean": float(np.nanmean(ss)),
                "slowness_median": float(np.nanmedian(ss)),
                "slowness_std": float(np.nanstd(ss)),
                "slowness_weighted_mean": float(np.average(ss, weights=weights)),
                "uncertainty_mean": float(np.nanmean(uu)),
                "uncertainty_median": float(np.nanmedian(uu)),
            }
        )
    return pd.DataFrame(rows)



def plot_classwise_boxplot(values_by_class, xlabel: str, out_path: Path, colors=None, title=None):
    classes = [cls for cls in PLOT_MATERIAL_ORDER if cls in values_by_class and len(values_by_class[cls]) > 0]
    if not classes:
        return
    data = [values_by_class[cls] for cls in classes]
    labels = [f"{cls}  (n={len(values_by_class[cls])})" for cls in classes]

    fig_h = max(5.4, 0.60 * len(classes) + 2.2)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=labels, showfliers=False, widths=0.65)
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor((colors or MATERIAL_COLORS)[cls])
        patch.set_alpha(0.88)
        patch.set_edgecolor("#4b5563")
        patch.set_linewidth(0.9)
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#374151")
            artist.set_linewidth(1.15)
    ax.set_xlabel(xlabel)
    ax.set_title(title or xlabel)
    ax.grid(True, axis="x", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))
    ax.tick_params(axis='y', labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_classwise_violin(values_by_class, xlabel: str, out_path: Path, title: str):
    classes = [cls for cls in PLOT_MATERIAL_ORDER if cls in values_by_class and len(values_by_class[cls]) > 1]
    if not classes:
        return
    data = [values_by_class[cls] for cls in classes]

    fig_h = max(5.6, 0.62 * len(classes) + 2.3)
    fig, ax = plt.subplots(figsize=(11.4, fig_h))
    vp = ax.violinplot(data, vert=False, showmeans=False, showmedians=True, showextrema=False)
    for body, cls in zip(vp["bodies"], classes):
        body.set_facecolor(MATERIAL_COLORS[cls])
        body.set_edgecolor("#374151")
        body.set_alpha(0.82)
    vp["cmedians"].set_color("#111827")
    vp["cmedians"].set_linewidth(1.25)

    ax.set_yticks(np.arange(1, len(classes) + 1))
    ax.set_yticklabels([f"{cls}  (n={len(values_by_class[cls])})" for cls in classes])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))
    ax.tick_params(axis='y', labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_material_counts(stats_df: pd.DataFrame, out_path: Path):
    df = stats_df.copy()
    if df.empty:
        return
    fig_h = max(4.8, 0.58 * len(df) + 1.6)
    fig, ax = plt.subplots(figsize=(10.4, fig_h))
    classes = df["class"].tolist()
    counts = df["n_cells"].fillna(0).to_numpy()
    colors = [MATERIAL_COLORS.get(c, "#9ca3af") for c in classes]
    ax.barh(classes, counts, color=colors, edgecolor="#4b5563", linewidth=0.5)
    ax.set_xlabel("Number of model-grid cells")
    ax.set_title("Surface-material coverage on the tomography grid")
    ax.grid(True, axis="x", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_weighted_means(stats_df: pd.DataFrame, out_path: Path):
    df = stats_df[stats_df["class"] != "No classification"].copy()
    if df.empty:
        return
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, max(5.4, 0.58 * len(df) + 2.0)), sharey=True)

    colors = [MATERIAL_COLORS[c] for c in df["class"]]
    axes[0].barh(y, df["velocity_weighted_mean"], color=colors, edgecolor="#4b5563", linewidth=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["class"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Weighted mean velocity [km/s]")
    axes[0].set_title("Weighted class means")
    axes[0].grid(True, axis="x", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    axes[1].barh(y, df["slowness_weighted_mean"], color=colors, edgecolor="#4b5563", linewidth=0.5)
    axes[1].set_xlabel("Weighted mean slowness [s/km]")
    axes[1].set_title("Weighted class means")
    axes[1].grid(True, axis="x", color="#c7cbd1", alpha=0.35, linestyle=(0, (2.5, 3.5)))

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_uncertainty_crossplots(v_mean, s_mean, v_std, class_grid, out_path: Path):
    valid = np.isfinite(v_mean) & np.isfinite(s_mean) & np.isfinite(v_std) & (class_grid != "No classification")
    if not np.any(valid):
        return

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))

    hb1 = axes[0].hexbin(v_mean[valid], v_std[valid], gridsize=28, cmap="viridis", mincnt=1)
    axes[0].set_xlabel("Velocity [km/s]")
    axes[0].set_ylabel("Velocity uncertainty [km/s]")
    axes[0].set_title("Velocity vs uncertainty")
    axes[0].grid(True, color="#c7cbd1", alpha=0.20, linestyle=(0, (2.5, 3.5)))
    cb1 = fig.colorbar(hb1, ax=axes[0], pad=0.02)
    cb1.set_label("Cell count")

    hb2 = axes[1].hexbin(s_mean[valid], v_std[valid], gridsize=28, cmap="cividis", mincnt=1)
    axes[1].set_xlabel("Slowness [s/km]")
    axes[1].set_ylabel("Velocity uncertainty [km/s]")
    axes[1].set_title("Slowness vs uncertainty")
    axes[1].grid(True, color="#c7cbd1", alpha=0.20, linestyle=(0, (2.5, 3.5)))
    cb2 = fig.colorbar(hb2, ax=axes[1], pad=0.02)
    cb2.set_label("Cell count")

    hb3 = axes[2].hexbin(v_mean[valid], s_mean[valid], gridsize=28, cmap="plasma", mincnt=1)
    axes[2].set_xlabel("Velocity [km/s]")
    axes[2].set_ylabel("Slowness [s/km]")
    axes[2].set_title("Velocity vs slowness")
    axes[2].grid(True, color="#c7cbd1", alpha=0.20, linestyle=(0, (2.5, 3.5)))
    cb3 = fig.colorbar(hb3, ax=axes[2], pad=0.02)
    cb3.set_label("Cell count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=LOCAL_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_values_by_class(class_grid, field):
    out = {}
    for cls in PLOT_MATERIAL_ORDER:
        mask = np.isfinite(field) & (class_grid == cls)
        if np.any(mask):
            out[cls] = field[mask].ravel()
    return out


# -------------------------
# Main
# -------------------------

def main():
    apply_extra_style()
    args = base.parse_args()

    chains_root = Path(args.chains_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chain_dirs = base.find_chain_dirs(chains_root, args.chain_glob)
    print(f"Found {len(chain_dirs)} chain folders.")

    chain_records = []
    loaded = []
    for chain_dir in chain_dirs:
        npz_path = chain_dir / "posterior_grids.npz"
        loaded_chain = base.load_chain_npz(npz_path)
        if loaded:
            base.assert_same_grid(loaded[0], loaded_chain)
        loaded.append(loaded_chain)

        rec = {"chain_dir": str(chain_dir.resolve())}
        for maybe_json, key in [
            (chain_dir / "chain_status.json", "chain_status"),
            (chain_dir / "chain_meta.json", "chain_meta"),
            (chain_dir / "run_summary.json", "run_summary"),
        ]:
            if maybe_json.exists():
                try:
                    rec[key] = json.loads(maybe_json.read_text(encoding="utf-8"))
                except Exception:
                    rec[key] = None
            else:
                rec[key] = None
        chain_records.append(rec)

    ref = loaded[0]
    XG = ref["XG"]
    YG = ref["YG"]
    xg = ref["xg"]
    yg = ref["yg"]

    v_mean_list = [c["v_mean"] for c in loaded]
    v_std_list = [c["v_std"] for c in loaded]
    s_mean_list = [c["s_mean"] for c in loaded]
    s_std_list = [c["s_std"] for c in loaded]
    logL_list = [c["logL"] for c in loaded]
    n_cells_list = [c["n_cells"] for c in loaded]

    v_mean_combined, v_std_combined = base.combine_mean_and_std(v_mean_list, v_std_list)
    s_mean_combined, s_std_combined = base.combine_mean_and_std(s_mean_list, s_std_list)

    combined_npz_path = outdir / "posterior_grids_combined.npz"
    np.savez(
        combined_npz_path,
        xg=xg,
        yg=yg,
        XG=XG,
        YG=YG,
        s_mean=s_mean_combined,
        s_std=s_std_combined,
        v_mean=v_mean_combined,
        v_std=v_std_combined,
        n_cells=np.concatenate([a for a in n_cells_list if a is not None and len(a) > 0]),
        logL=np.concatenate([a for a in logL_list if a is not None and len(a) > 0]),
    )
    print(f"Saved: {combined_npz_path}")

    summary = {
        "n_chains_used": len(chain_dirs),
        "chains_root": str(chains_root.resolve()),
        "chain_glob": args.chain_glob,
        "grid_shape": list(XG.shape),
        "x_range_km": [float(np.nanmin(XG)), float(np.nanmax(XG))],
        "y_range_km": [float(np.nanmin(YG)), float(np.nanmax(YG))],
        "v_mean_range_km_s": [float(np.nanmin(v_mean_combined)), float(np.nanmax(v_mean_combined))],
        "v_std_range_km_s": [float(np.nanmin(v_std_combined)), float(np.nanmax(v_std_combined))],
        "s_mean_range_s_km": [float(np.nanmin(s_mean_combined)), float(np.nanmax(s_mean_combined))],
        "s_std_range_s_km": [float(np.nanmin(s_std_combined)), float(np.nanmax(s_std_combined))],
    }
    (outdir / "combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {outdir / 'combined_summary.json'}")

    meta_rows = []
    for rec in chain_records:
        row = {"chain_dir": rec["chain_dir"]}
        for block_name in ["chain_status", "chain_meta", "run_summary"]:
            block = rec.get(block_name) or {}
            if isinstance(block, dict):
                for k, v in block.items():
                    row[f"{block_name}.{k}"] = json.dumps(v) if isinstance(v, (dict, list)) else v
        meta_rows.append(row)
    pd.DataFrame(meta_rows).to_csv(outdir / "combined_chain_metadata.csv", index=False)
    print(f"Saved: {outdir / 'combined_chain_metadata.csv'}")

    # Diagnostics
    plot_combined_diagnostics_nice(logL_list, n_cells_list, outdir / "diagnostics_combined.png")
    plot_diagnostics_traces(logL_list, n_cells_list, outdir / "diagnostics_traces_combined.png")
    plot_diagnostics_posteriors(logL_list, n_cells_list, outdir / "diagnostics_posteriors_combined.png")
    base.plot_combined_summary(
        v_mean_combined,
        v_std_combined,
        s_mean_combined,
        n_cells_list,
        XG,
        YG,
        outdir / "posterior_summary_combined.png",
        velocity_cmap=args.velocity_cmap,
        uncertainty_cmap=args.uncertainty_cmap,
    )

    first_chain = chain_dirs[0]
    train_df = base.load_optional_csv(first_chain / "filtered_rays_train.csv")
    holdout_df = None if args.no_holdout else base.load_optional_csv(first_chain / "filtered_rays_holdout.csv")
    stations_df_local = base.extract_unique_stations_from_rays(train_df, holdout_df) if train_df is not None and len(train_df) > 0 else None

    # Local plots: clean versions + separate ray coverage
    plot_local_field_clean(v_mean_combined, XG, YG, stations_df_local, outdir / "velocity_mean_local_combined.png", "Posterior mean velocity (local coordinates)", "Velocity [km/s]", cmap=args.local_cmap)
    plot_local_field_clean(v_std_combined, XG, YG, stations_df_local, outdir / "velocity_uncertainty_local_combined.png", "Posterior velocity uncertainty (local coordinates)", "Velocity uncertainty [km/s]", cmap=args.uncertainty_cmap)
    plot_local_ray_coverage(train_df, holdout_df, stations_df_local, XG, YG, outdir / "ray_coverage_local_combined.png")

    if train_df is None or len(train_df) == 0:
        print("[warn] filtered_rays_train.csv not found or empty; skipping geographic / materials comparison plots.")
        return

    stations_df = base.extract_unique_stations_from_rays(train_df, holdout_df)
    transform = base.fit_affine_xy_to_lonlat(stations_df)
    LON, LAT, _, extent = base.compute_geo_context(stations_df, transform, XG, YG)

    # Better geographic singles
    plot_geo_single_field(v_mean_combined, stations_df, transform, XG, YG, outdir / "geo_velocity_mean_combined.png", "Posterior mean velocity", "Velocity [km/s]", station_labels=args.station_labels, cmap=args.velocity_cmap)
    plot_geo_single_field(v_std_combined, stations_df, transform, XG, YG, outdir / "geo_velocity_uncertainty_combined.png", "Posterior velocity uncertainty", "Velocity uncertainty [km/s]", station_labels=False, cmap=args.uncertainty_cmap)
    plot_geo_ray_coverage(train_df, holdout_df, stations_df, transform, XG, YG, outdir / "geo_ray_coverage_combined.png", station_labels=args.station_labels)

    materials, material_meta = base.fetch_harmonized_surface_materials(extent, args)
    if materials is None:
        print("[warn] Harmonized surface-materials plots were skipped because neither land nor marine WFS data could be fetched.")
        return

    base.write_surface_materials_summary(outdir / "surface_materials_harmonized_summary.json", material_meta)
    save_surface_material_legend(outdir / "surface_materials_legend.png")
    plot_geo_context_nice(materials, stations_df, transform, XG, YG, outdir / "geo_surface_materials_harmonized_context_combined.png", station_labels=args.station_labels)
    plot_geo_overlay_nice(v_mean_combined, stations_df, transform, XG, YG, materials, outdir / "geo_velocity_surface_materials_harmonized_overlay_combined.png", "Posterior mean velocity over harmonized surface materials", "Velocity [km/s]", station_labels=args.station_labels, overlay_alpha=min(float(getattr(args, 'overlay_alpha', 0.28)), 0.28), cmap=args.velocity_cmap)
    plot_geo_two_panel(v_mean_combined, v_std_combined, stations_df, transform, XG, YG, materials, outdir / "geo_velocity_surface_materials_harmonized_comparison_combined.png", velocity_cmap=args.velocity_cmap, uncertainty_cmap=args.uncertainty_cmap)
    plot_geo_two_panel(v_mean_combined, v_std_combined, stations_df, transform, XG, YG, materials, outdir / "geo_velocity_mean_uncertainty_two_panel_combined.png", velocity_cmap=args.velocity_cmap, uncertainty_cmap=args.uncertainty_cmap)

    # Grid-sampled material comparison plots
    class_grid = build_material_class_grid(LON, LAT, materials)
    stats_df = material_stats_dataframe(class_grid, v_mean_combined, s_mean_combined, v_std_combined)
    stats_df.to_csv(outdir / "surface_materials_classwise_stats.csv", index=False)
    print(f"Saved: {outdir / 'surface_materials_classwise_stats.csv'}")

    values_velocity = build_values_by_class(class_grid, v_mean_combined)
    values_slowness = build_values_by_class(class_grid, s_mean_combined)
    values_uncertainty = build_values_by_class(class_grid, v_std_combined)

    plot_classwise_boxplot(values_velocity, "Velocity [km/s]", outdir / "classwise_velocity_boxplot.png", title="Velocity by surface-material class")
    plot_classwise_boxplot(values_slowness, "Slowness [s/km]", outdir / "classwise_slowness_boxplot.png", title="Slowness by surface-material class")
    plot_classwise_boxplot(values_uncertainty, "Velocity uncertainty [km/s]", outdir / "classwise_uncertainty_boxplot.png", title="Uncertainty by surface-material class")
    plot_classwise_violin(values_velocity, "Velocity [km/s]", outdir / "classwise_velocity_violin.png", title="Velocity distribution by surface-material class")
    plot_classwise_violin(values_slowness, "Slowness [s/km]", outdir / "classwise_slowness_violin.png", title="Slowness distribution by surface-material class")
    plot_classwise_violin(values_uncertainty, "Velocity uncertainty [km/s]", outdir / "classwise_uncertainty_violin.png", title="Uncertainty distribution by surface-material class")
    plot_material_counts(stats_df, outdir / "surface_material_counts.png")
    plot_weighted_means(stats_df, outdir / "surface_material_weighted_means.png")
    plot_uncertainty_crossplots(v_mean_combined, s_mean_combined, v_std_combined, class_grid, outdir / "velocity_slowness_uncertainty_crossplots.png")


if __name__ == "__main__":
    main()
