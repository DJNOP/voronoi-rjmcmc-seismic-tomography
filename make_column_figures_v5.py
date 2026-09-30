from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DOUBLE_COL_IN = 7.1
PANEL_DPI = 320
LABEL_FS = 8.0
LEGEND_FS = 6.7


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_fraction(img: Image.Image, left: float, top: float, right: float, bottom: float) -> Image.Image:
    w, h = img.size
    return img.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))


def trim_whitespace(img: Image.Image, threshold: int = 245, pad: int = 4) -> Image.Image:
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


def add_border(img: Image.Image, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0,
               color: tuple[int, int, int] = (245, 245, 245)) -> Image.Image:
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=color)


def panel_label(ax, txt: str) -> None:
    ax.text(
        0.012,
        0.965,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=LABEL_FS,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.7", alpha=0.98),
        zorder=10,
    )


def extract_map_frame_and_bar(img: Image.Image):
    # Crop away the source title completely while keeping the embedded lon/lat axes.
    map_img = crop_fraction(img, 0.010, 0.118, 0.905, 0.988)
    cbar_img = crop_fraction(img, 0.900, 0.090, 0.998, 0.988)
    map_img = trim_whitespace(map_img, pad=2)
    cbar_img = trim_whitespace(cbar_img, pad=2)
    # Small top border keeps panel labels from sitting on the image edge.
    map_img = add_border(map_img, left=4, top=8, right=4, bottom=4)
    cbar_img = add_border(cbar_img, left=2, top=8, right=2, bottom=4)
    return map_img, cbar_img


def extract_hist_panel(img: Image.Image) -> Image.Image:
    # Bottom-right panel only; crop tighter to remove leftover titles/axes from neighbours.
    crop = crop_fraction(img, 0.505, 0.595, 0.998, 0.998)
    crop = trim_whitespace(crop, pad=2)
    return add_border(crop, left=8, top=8, right=4, bottom=4)


def extract_context_map(ray_img: Image.Image, mat_img: Image.Image) -> tuple[Image.Image, Image.Image]:
    # Remove the original map title and the bottom geology legend.
    ray_crop = crop_fraction(ray_img, 0.010, 0.118, 0.998, 0.875)
    mat_crop = crop_fraction(mat_img, 0.010, 0.118, 0.998, 0.875)
    ray_crop = add_border(trim_whitespace(ray_crop, pad=2), left=4, top=8, right=4, bottom=4)
    mat_crop = add_border(trim_whitespace(mat_crop, pad=2), left=4, top=8, right=4, bottom=4)
    return ray_crop, mat_crop


def _ray_mask_from_image(ray_img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    rays = np.asarray(ray_img.resize(target_size, Image.Resampling.BILINEAR))
    r = rays[..., 0].astype(np.int16)
    g = rays[..., 1].astype(np.int16)
    b = rays[..., 2].astype(np.int16)

    blue_mask = (b > 160) & (b - r > 18) & (b - g > 8) & ((r + g + b) / 3 > 118)
    orange_mask = (r > 180) & (g > 130) & (b < 170) & (r - b > 30) & ((r + g + b) / 3 > 120)
    mask = (blue_mask | orange_mask) & ~((r > 225) & (g > 225) & (b > 225))

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=3))
    return np.asarray(mask_img) > 0


def overlay_rays_on_materials(ray_img: Image.Image, materials_img: Image.Image) -> Image.Image:
    base = np.asarray(materials_img).copy()
    mask = _ray_mask_from_image(ray_img, materials_img.size)

    ray_color = np.array([70, 145, 255], dtype=np.uint8)
    alpha = 0.72
    base[mask] = (alpha * ray_color + (1.0 - alpha) * base[mask]).astype(np.uint8)
    return Image.fromarray(base)


def save_context_overlay(ray_path: Path, mats_path: Path, out_path: Path):
    ray_raw = load_image(ray_path)
    mats_raw = load_image(mats_path)
    ray, mats = extract_context_map(ray_raw, mats_raw)
    overlay = overlay_rays_on_materials(ray, mats)

    fig_h = DOUBLE_COL_IN * overlay.size[1] / overlay.size[0] * 0.80
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, fig_h))
    ax.imshow(overlay)
    ax.axis("off")
    panel_label(ax, "Ray coverage on surface materials")

    handles = [
        Line2D([0], [0], color=np.array([70, 145, 255]) / 255.0, lw=2.0, label="rays"),
        Line2D([0], [0], marker="o", markersize=5.0, color="none",
               markerfacecolor="crimson", markeredgecolor="white", label="stations"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.86),
        ncol=1,
        frameon=True,
        fontsize=LEGEND_FS,
        handlelength=2.6,
        labelspacing=0.32,
        borderpad=0.30,
        borderaxespad=0.18,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.7",
    )
    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.01)
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_two_map_shared_bar(left_path: Path, right_path: Path, out_path: Path, left_label: str, right_label: str):
    left_map, shared_bar = extract_map_frame_and_bar(load_image(left_path))
    right_map, _ = extract_map_frame_and_bar(load_image(right_path))

    ratio = max(left_map.size[1] / left_map.size[0], right_map.size[1] / right_map.size[0])
    fig_h = DOUBLE_COL_IN * ratio / 2.0 + 0.16
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

    fig.subplots_adjust(left=0.004, right=0.996, top=0.992, bottom=0.01, wspace=0.020)
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_appendix_pair(mean_path: Path, unc_path: Path, out_path: Path, ensemble_label: str):
    mean_map, mean_bar = extract_map_frame_and_bar(load_image(mean_path))
    unc_map, unc_bar = extract_map_frame_and_bar(load_image(unc_path))

    ratio = max(mean_map.size[1] / mean_map.size[0], unc_map.size[1] / unc_map.size[0])
    fig_h = DOUBLE_COL_IN * ratio / 2.0 + 0.16
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

    fig.subplots_adjust(left=0.004, right=0.996, top=0.992, bottom=0.01, wspace=0.020)
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_ncells_compare(sum100: Path, sum200: Path, out_path: Path):
    h100 = extract_hist_panel(load_image(sum100))
    h200 = extract_hist_panel(load_image(sum200))

    ratio = max(h100.size[1] / h100.size[0], h200.size[1] / h200.size[0])
    fig_h = DOUBLE_COL_IN * ratio / 2.0 + 0.12
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, fig_h))

    for ax, img, lab in zip(axes, [h100, h200], ["(a) 100k steps", "(b) 200k steps"]):
        ax.imshow(img)
        ax.axis("off")
        panel_label(ax, lab)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.992, bottom=0.01, wspace=0.012)
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
