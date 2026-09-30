from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DOUBLE_COL_IN = 7.1
PANEL_DPI = 320
LABEL_FS = 8.5
LEGEND_FS = 8


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_fraction(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = img.size
    return img.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))


def trim_whitespace(img: Image.Image, threshold: int = 245, pad: int = 6) -> Image.Image:
    arr = np.asarray(img)
    mask = np.any(arr < threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return img.crop((x0, y0, x1, y1))


def panel_label(ax, txt: str) -> None:
    ax.text(
        0.012,
        0.988,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=LABEL_FS,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="0.7", alpha=0.97),
    )


def extract_map_frame_and_bar(img: Image.Image):
    """
    Keep the embedded axis tick labels fully visible, including the left y-axis values
    and the x-axis longitude labels. Crop away most of the original title only.
    """
    map_img = crop_fraction(img, 0.0, 0.035, 0.905, 0.985)
    cbar_img = crop_fraction(img, 0.895, 0.025, 0.998, 0.985)
    return trim_whitespace(map_img, pad=4), trim_whitespace(cbar_img, pad=4)


def extract_hist_panel(img: Image.Image) -> Image.Image:
    """
    Keep the full histogram title and the y-axis label.
    """
    crop = crop_fraction(img, 0.455, 0.47, 0.998, 0.998)
    return trim_whitespace(crop, pad=4)


def extract_context_map(img: Image.Image) -> Image.Image:
    """
    Keep axis values and enough room for an in-plot legend.
    """
    crop = crop_fraction(img, 0.0, 0.03, 0.998, 0.975)
    return trim_whitespace(crop, pad=4)


def _ray_mask_from_image(ray_img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    rays = np.asarray(ray_img.resize(target_size, Image.Resampling.BILINEAR))
    r = rays[..., 0].astype(np.int16)
    g = rays[..., 1].astype(np.int16)
    b = rays[..., 2].astype(np.int16)

    blue_mask = (b > 165) & (b - r > 20) & (b - g > 8) & ((r + g + b) / 3 > 120)
    orange_mask = (r > 185) & (g > 135) & (b < 170) & (r - b > 35) & ((r + g + b) / 3 > 125)
    mask = blue_mask | orange_mask
    mask &= ~((r > 220) & (g > 220) & (b > 220))

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=3))
    return np.asarray(mask_img) > 0


def overlay_rays_on_materials(ray_img: Image.Image, materials_img: Image.Image) -> Image.Image:
    base = np.asarray(materials_img).copy()
    mask = _ray_mask_from_image(ray_img, materials_img.size)

    ray_color = np.array([70, 145, 255], dtype=np.uint8)
    alpha = 0.68
    base[mask] = (alpha * ray_color + (1.0 - alpha) * base[mask]).astype(np.uint8)
    return Image.fromarray(base)


def save_context_overlay(ray_path: Path, mats_path: Path, out_path: Path):
    ray = extract_context_map(load_image(ray_path))
    mats = extract_context_map(load_image(mats_path))
    overlay = overlay_rays_on_materials(ray, mats)

    fig_h = DOUBLE_COL_IN * overlay.size[1] / overlay.size[0] * 0.82
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, fig_h))
    ax.imshow(overlay)
    ax.axis("off")
    panel_label(ax, "Ray coverage on surface materials")

    handles = [
        Line2D([0], [0], color=np.array([70, 145, 255]) / 255.0, lw=2.0, label="rays"),
        Line2D([0], [0], marker="o", markersize=5.5, color="none",
               markerfacecolor="crimson", markeredgecolor="white", label="stations"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.84),
        ncol=1,
        frameon=True,
        fontsize=LEGEND_FS,
        handlelength=2.8,
        borderaxespad=0.3,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.7",
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.992, bottom=0.02)
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_two_map_shared_bar(left_path: Path, right_path: Path, out_path: Path, left_label: str, right_label: str):
    left_map, shared_bar = extract_map_frame_and_bar(load_image(left_path))
    right_map, _ = extract_map_frame_and_bar(load_image(right_path))

    ratio = max(left_map.size[1] / left_map.size[0], right_map.size[1] / right_map.size[0])
    fig_h = DOUBLE_COL_IN * ratio / 2.0 + 0.20
    fig = plt.figure(figsize=(DOUBLE_COL_IN, fig_h))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.10])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    ax1.imshow(left_map)
    ax1.axis("off")
    panel_label(ax1, left_label)

    ax2.imshow(right_map)
    ax2.axis("off")
    panel_label(ax2, right_label)

    cax.imshow(shared_bar)
    cax.axis("off")

    fig.subplots_adjust(left=0.008, right=0.995, top=0.992, bottom=0.02, wspace=0.022)
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_appendix_pair(mean_path: Path, unc_path: Path, out_path: Path, ensemble_label: str):
    mean_map, mean_bar = extract_map_frame_and_bar(load_image(mean_path))
    unc_map, unc_bar = extract_map_frame_and_bar(load_image(unc_path))

    ratio = max(mean_map.size[1] / mean_map.size[0], unc_map.size[1] / unc_map.size[0])
    fig_h = DOUBLE_COL_IN * ratio / 2.0 + 0.20
    fig = plt.figure(figsize=(DOUBLE_COL_IN, fig_h))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.10, 1, 0.10])

    ax1 = fig.add_subplot(gs[0, 0])
    cb1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cb2 = fig.add_subplot(gs[0, 3])

    ax1.imshow(mean_map)
    ax1.axis("off")
    panel_label(ax1, f"(a) Mean, {ensemble_label}")

    cb1.imshow(mean_bar)
    cb1.axis("off")

    ax2.imshow(unc_map)
    ax2.axis("off")
    panel_label(ax2, f"(b) Uncertainty, {ensemble_label}")

    cb2.imshow(unc_bar)
    cb2.axis("off")

    fig.subplots_adjust(left=0.008, right=0.995, top=0.992, bottom=0.02, wspace=0.022)
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_ncells_compare(sum100: Path, sum200: Path, out_path: Path):
    h100 = extract_hist_panel(load_image(sum100))
    h200 = extract_hist_panel(load_image(sum200))

    ratio = max(h100.size[1] / h100.size[0], h200.size[1] / h200.size[0])
    fig_h = DOUBLE_COL_IN * ratio / 2.0 + 0.14
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, fig_h))

    for ax, img, lab in zip(axes, [h100, h200], ["(a) 100k steps", "(b) 200k steps"]):
        ax.imshow(img)
        ax.axis("off")
        panel_label(ax, lab)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.992, bottom=0.02, wspace=0.02)
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make compact paper figures from combined tomography PNGs.")
    p.add_argument("--dir-100k", type=Path, required=True)
    p.add_argument("--dir-200k", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("./paper_figures"))
    p.add_argument("--ray-coverage", type=Path, default=None)
    p.add_argument("--surface-materials", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ray = args.ray_coverage or args.dir_100k / "geo_ray_coverage_combined.png"
    mats = args.surface_materials or args.dir_100k / "geo_surface_materials_harmonized_context_combined.png"

    if not ray.exists() and (args.dir_200k / "geo_ray_coverage_combined.png").exists():
        ray = args.dir_200k / "geo_ray_coverage_combined.png"
    if not mats.exists() and (args.dir_200k / "geo_surface_materials_harmonized_context_combined.png").exists():
        mats = args.dir_200k / "geo_surface_materials_harmonized_context_combined.png"

    required = [
        args.dir_100k / "posterior_summary_combined.png",
        args.dir_200k / "posterior_summary_combined.png",
        args.dir_100k / "geo_velocity_mean_combined.png",
        args.dir_200k / "geo_velocity_mean_combined.png",
        args.dir_100k / "geo_velocity_uncertainty_combined.png",
        args.dir_200k / "geo_velocity_uncertainty_combined.png",
        ray,
        mats,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    save_context_overlay(ray, mats, args.outdir / "paper_context_overlay.png")
    save_ncells_compare(
        args.dir_100k / "posterior_summary_combined.png",
        args.dir_200k / "posterior_summary_combined.png",
        args.outdir / "paper_posterior_ncells_compare.png",
    )
    save_two_map_shared_bar(
        args.dir_100k / "geo_velocity_mean_combined.png",
        args.dir_200k / "geo_velocity_mean_combined.png",
        args.outdir / "paper_mean_compare_shared.png",
        "(a) Mean, 100k",
        "(b) Mean, 200k",
    )
    save_two_map_shared_bar(
        args.dir_100k / "geo_velocity_uncertainty_combined.png",
        args.dir_200k / "geo_velocity_uncertainty_combined.png",
        args.outdir / "paper_uncertainty_compare_shared.png",
        "(a) Uncertainty, 100k",
        "(b) Uncertainty, 200k",
    )
    save_appendix_pair(
        args.dir_100k / "geo_velocity_mean_combined.png",
        args.dir_100k / "geo_velocity_uncertainty_combined.png",
        args.outdir / "paper_appendix_maps_100k.png",
        "100k",
    )
    save_appendix_pair(
        args.dir_200k / "geo_velocity_mean_combined.png",
        args.dir_200k / "geo_velocity_uncertainty_combined.png",
        args.outdir / "paper_appendix_maps_200k.png",
        "200k",
    )


if __name__ == "__main__":
    main()
