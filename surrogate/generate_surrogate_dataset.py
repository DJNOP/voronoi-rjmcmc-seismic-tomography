"""
Seismic Tomography Surrogate Dataset v2
========================================
Larger domain, denser grid, more sources/receivers, and
geometrically symmetric ray coverage.

Key changes from v1
-------------------
- Domain  : 50 × 50 km  (was 30 × 30)
- Grid    : 50 × 50 = 2500 cells  (was 30 × 30)
- Sources : 48  (12 per edge, was 24 total)
- Receivers: 48  (12 per edge, was 24 total)
- Offset  : exactly half the inter-source spacing
            → perfect interleaving, equal coverage per edge
- 4 symmetry quadrants confirmed equal by design

Symmetry guarantee
------------------
n_src = n_rec = 4k  (k = 12 stations per edge).
Sources start at x=0, y=0 and step by perimeter/n_src.
Receivers are offset by exactly perimeter/(2*n_src) — half a step —
so they interleave perfectly between sources.
Because each edge is the same length AND hosts the same number of both
sources and receivers, every edge produces an identical sub-problem.
Ray coverage is therefore 4-fold symmetric by construction.

Outputs
-------
data/surrogate_seismic_dataset_v2.npz   (input to the notebook and run_chains.py)
figures/surrogate_dataset.png            (panel A is Fig. 1 of the report)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRID / DOMAIN
# ─────────────────────────────────────────────────────────────────────────────
Nx, Ny = 50, 50
Lx, Ly = 50.0, 50.0
dx, dy = Lx / Nx, Ly / Ny

xc = np.linspace(dx / 2, Lx - dx / 2, Nx)
yc = np.linspace(dy / 2, Ly - dy / 2, Ny)
XX, YY = np.meshgrid(xc, yc)           # (Ny, Nx)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRUE SLOWNESS MODEL  (3 anomalies scaled to the larger domain)
# ─────────────────────────────────────────────────────────────────────────────
s0 = 1.0 / 3.0          # background  →  v₀ = 3 km/s

slowness_true = np.full((Ny, Nx), s0)

# Anomaly 1 — RECTANGLE  (high slowness, +18 %)   upper-left quadrant
r_x1, r_x2 = 7.0,  20.0
r_y1, r_y2 = 28.0, 42.0
mask_rect   = (XX >= r_x1) & (XX <= r_x2) & (YY >= r_y1) & (YY <= r_y2)
slowness_true[mask_rect] += 0.06

# Anomaly 2 — CIRCLE  (low slowness, −15 %)        right-centre
cx, cy, cr  = 37.0, 35.0, 7.5
mask_circ   = (XX - cx)**2 + (YY - cy)**2 <= cr**2
slowness_true[mask_circ] -= 0.05

# Anomaly 3 — TRIANGLE  (high slowness, +20 %)     lower-centre
T = np.array([[17.0,  5.0],
              [36.0,  6.0],
              [27.0, 22.0]])

def _sign(p1x, p1y, p2x, p2y, p3x, p3y):
    return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

def in_triangle(px, py, T):
    ax, ay = T[0];  bx, by = T[1];  cx2, cy2 = T[2]
    d1 = _sign(px, py, ax, ay, bx, by)
    d2 = _sign(px, py, bx, by, cx2, cy2)
    d3 = _sign(px, py, cx2, cy2, ax, ay)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)

mask_tri = in_triangle(XX, YY, T)
slowness_true[mask_tri] += 0.067

delta_s = slowness_true - s0

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SYMMETRIC SOURCE / RECEIVER GEOMETRY
#
#  Symmetry is achieved by three design choices:
#   (a) n_src = n_rec = 4 × n_per_edge  (same count on every edge)
#   (b) Square domain  (Lx = Ly)  → identical edge lengths
#   (c) Receiver offset = exactly half the source spacing
#       = perimeter / (2 × n_src)
#  This guarantees 4-fold rotational symmetry in the station layout
#  and therefore in the ray coverage map.
# ─────────────────────────────────────────────────────────────────────────────
n_per_edge = 12          # stations per edge
n_src      = 4 * n_per_edge   # = 48
n_rec      = 4 * n_per_edge   # = 48
perimeter  = 2.0 * (Lx + Ly)
src_step   = perimeter / n_src

# Start sources at half-step so no station ever lands exactly on a corner.
# With step = perimeter/n_src and n_per_edge*step = Lx exactly, starting
# at step/2 means sources sit at the centres of their edge-sections and
# all 12 stay cleanly on their respective edge.
# Receivers are then offset by one full step = two half-steps from sources,
# which still gives perfect interleaving (receivers sit at +step/2 relative
# to sources, which is the midpoint between consecutive source positions).
src_offset = src_step / 2.0          # sources centred in each section
offset     = src_offset + src_step   # receivers: one step further


def boundary_points(n, Lx, Ly, offset=0.0):
    """n equally-spaced points around the rectangular perimeter."""
    perimeter = 2.0 * (Lx + Ly)
    pts = []
    for i in range(n):
        d = (i * perimeter / n + offset) % perimeter
        if   d < Lx:            pts.append((d,      0.0))
        elif d < Lx + Ly:       pts.append((Lx,     d - Lx))
        elif d < 2*Lx + Ly:     pts.append((2*Lx + Ly - d, Ly))
        else:                   pts.append((0.0,    perimeter - d))
    return np.array(pts)


sources   = boundary_points(n_src, Lx, Ly, offset=src_offset)
receivers = boundary_points(n_rec, Lx, Ly, offset=offset)

# Verify symmetry: each edge should have exactly n_per_edge of each
def count_per_edge(pts, Lx, Ly, tol=0.5):
    bottom = np.sum(pts[:, 1] < tol)
    right  = np.sum(pts[:, 0] > Lx - tol)
    top    = np.sum(pts[:, 1] > Ly - tol)
    left   = np.sum(pts[:, 0] < tol)
    return bottom, right, top, left

src_counts = count_per_edge(sources,   Lx, Ly)
rec_counts = count_per_edge(receivers, Lx, Ly)
print("Sources   per edge (B/R/T/L):", src_counts)
print("Receivers per edge (B/R/T/L):", rec_counts)
assert all(c == n_per_edge for c in src_counts), "Source symmetry broken!"
assert all(c == n_per_edge for c in rec_counts), "Receiver symmetry broken!"
print("✓ Symmetric geometry verified")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FORWARD MODEL  —  straight-ray tomography  d = G · m
# ─────────────────────────────────────────────────────────────────────────────
def ray_path_lengths(sx, sy, rx, ry, xc, yc, dx, dy, Nx, Ny, oversample=12):
    """Path-length matrix [km] for a straight ray through the regular grid."""
    L       = np.zeros((Ny, Nx))
    ray_len = np.hypot(rx - sx, ry - sy)
    if ray_len < 1e-9:
        return L
    n_steps = max(int(ray_len / min(dx, dy) * oversample), 200)
    ts  = np.linspace(0.0, 1.0, n_steps + 1)
    xs  = sx + ts * (rx - sx)
    ys  = sy + ts * (ry - sy)
    seg = ray_len / n_steps
    xm  = 0.5 * (xs[:-1] + xs[1:])
    ym  = 0.5 * (ys[:-1] + ys[1:])
    ix  = np.floor(xm / dx).astype(int)
    iy  = np.floor(ym / dy).astype(int)
    ok  = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)
    np.add.at(L, (iy[ok], ix[ok]), seg)
    return L


n_rays = n_src * n_rec
G      = np.zeros((n_rays, Ny * Nx), dtype=np.float32)
m_true = slowness_true.flatten()

print(f"\nBuilding G  [{n_rays} rays × {Ny*Nx} cells] …")
ray_idx = 0
for i, (sx, sy) in enumerate(sources):
    for j, (rx, ry) in enumerate(receivers):
        L = ray_path_lengths(sx, sy, rx, ry, xc, yc, dx, dy, Nx, Ny)
        G[ray_idx] = L.flatten()
        ray_idx += 1
    if (i + 1) % 12 == 0:
        print(f"  source {i+1}/{n_src} done")

print("Computing travel times …")
d_true = G @ m_true.astype(np.float32)

# Noise: 0.5 % of mean travel time
noise_std = 0.005 * float(np.mean(d_true))
rng       = np.random.default_rng(42)
d_obs     = d_true + rng.normal(0.0, noise_std, n_rays).astype(np.float32)

print(f"\n{'='*58}")
print(f"  Grid            : {Nx} × {Ny} = {Nx*Ny} cells")
print(f"  Domain          : {Lx} × {Ly} km")
print(f"  Cell size       : {dx:.2f} × {dy:.2f} km")
print(f"  Sources         : {n_src}  ({n_per_edge}/edge)")
print(f"  Receivers       : {n_rec}  ({n_per_edge}/edge)")
print(f"  Rays            : {n_rays}")
print(f"  G shape         : {G.shape}")
print(f"  Background vel  : {1/s0:.1f} km/s")
print(f"  Noise σ         : {noise_std:.6f} s  ({noise_std/np.mean(d_true)*100:.2f} % of mean t)")
print(f"{'='*58}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE
# ─────────────────────────────────────────────────────────────────────────────
np.savez_compressed(
    DATA_DIR / "surrogate_seismic_dataset_v2.npz",
    d_obs         = d_obs,
    d_true        = d_true,
    noise_std     = np.float32(noise_std),
    G             = G,
    slowness_true = slowness_true.astype(np.float32),
    delta_s       = delta_s.astype(np.float32),
    sources       = sources.astype(np.float32),
    receivers     = receivers.astype(np.float32),
    xc            = xc.astype(np.float32),
    yc            = yc.astype(np.float32),
    grid_params   = np.array([Nx, Ny, Lx, Ly, dx, dy, s0, noise_std]),
)
print(f"Saved → {DATA_DIR / 'surrogate_seismic_dataset_v2.npz'}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  FIGURE
# ─────────────────────────────────────────────────────────────────────────────
DARK, PANEL, BORDER = '#0d1117', '#161b22', '#30363d'
TEXT, MUTED         = '#e6edf3', '#8b949e'

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

fig = plt.figure(figsize=(20, 12), facecolor=DARK)
gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.36,
                       left=0.06, right=0.97, top=0.93, bottom=0.07)

norm = TwoSlopeNorm(vmin=slowness_true.min(), vcenter=s0, vmax=slowness_true.max())

# ── A: True slowness model ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[:, 0]); style_ax(ax)
im = ax.pcolormesh(XX, YY, slowness_true, cmap='RdBu_r', norm=norm, shading='auto')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Slowness [s/km]', color=MUTED, fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)

# Anomaly outlines
ax.add_patch(mpatches.Rectangle((r_x1,r_y1), r_x2-r_x1, r_y2-r_y1,
    lw=1.5, ec='#ff7b72', fc='none', ls='--', zorder=5))
ax.add_patch(plt.Circle((cx,cy), cr, lw=1.5, ec='#79c0ff', fc='none', ls='--', zorder=5))
ax.add_patch(MplPolygon(T, lw=1.5, ec='#f0883e', fc='none', ls='--', zorder=5, closed=True))
ax.annotate("① Rect",   xy=((r_x1+r_x2)/2,(r_y1+r_y2)/2), color='#ff7b72',
            fontsize=8, ha='center', va='center', fontweight='bold')
ax.annotate("② Circle", xy=(cx,cy),                        color='#79c0ff',
            fontsize=8, ha='center', va='center', fontweight='bold')
ax.annotate("③ Tri",    xy=(T[:,0].mean(), T[:,1].mean()+1.5), color='#f0883e',
            fontsize=8, ha='center', va='center', fontweight='bold')

ax.scatter(sources[:,0],   sources[:,1],   c='#ff7b72', s=30, marker='^', zorder=6,
           linewidths=0.4, edgecolors='#fff', label=f'Sources ({n_src}, {n_per_edge}/edge)')
ax.scatter(receivers[:,0], receivers[:,1], c='#79c0ff', s=30, marker='v', zorder=6,
           linewidths=0.4, edgecolors='#fff', label=f'Receivers ({n_rec}, {n_per_edge}/edge)')
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, loc='upper right')
ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]')
ax.set_title('A — True Slowness Model', fontsize=10, fontweight='bold', pad=8)
ax.set_aspect('equal'); ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)

# ── B: Symmetry verification — hit-count map ────────────────────────────────
ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
hit = np.zeros((Ny, Nx))
for k in range(n_rays):
    hit += (G[k].reshape(Ny, Nx) > 0).astype(float)

# Verify 4-fold symmetry numerically
q1 = hit[:Ny//2, :Nx//2]          # bottom-left
q2 = hit[:Ny//2, Nx//2:]          # bottom-right
q3 = hit[Ny//2:, :Nx//2]          # top-left
q4 = hit[Ny//2:, Nx//2:]          # top-right
sym_score = 1 - (np.std([q1.mean(), q2.mean(),
                          q3.mean(), q4.mean()]) /
                 np.mean([q1.mean(), q2.mean(), q3.mean(), q4.mean()]))
print(f"\nSymmetry score (1=perfect): {sym_score:.4f}")

im = ax.pcolormesh(XX, YY, hit/hit.max(), cmap='hot', shading='auto', vmin=0, vmax=1)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('Normalised hit count', color=MUTED, fontsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)

# Draw quadrant lines + per-quadrant mean
ax.axhline(Ly/2, color=BORDER, lw=1.0, ls=':')
ax.axvline(Lx/2, color=BORDER, lw=1.0, ls=':')
for q, (tx, ty) in zip([q1,q2,q3,q4],
                        [(12,12),(38,12),(12,38),(38,38)]):
    ax.text(tx, ty, f'{q.mean():.0f}', color=TEXT, fontsize=8,
            ha='center', va='center', fontweight='bold',
            bbox=dict(fc=PANEL, ec=BORDER, boxstyle='round,pad=0.2', alpha=0.8))

ax.scatter(sources[:,0],   sources[:,1],   c='#ff7b72', s=20, marker='^', zorder=5)
ax.scatter(receivers[:,0], receivers[:,1], c='#79c0ff', s=20, marker='v', zorder=5)
ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]')
ax.set_title(f'B — Ray Coverage  (symmetry score = {sym_score:.4f})',
             fontsize=9, fontweight='bold', pad=6)
ax.set_aspect('equal'); ax.set_xlim(0,Lx); ax.set_ylim(0,Ly)

# ── C: v1 vs v2 station layout comparison ───────────────────────────────────
ax = fig.add_subplot(gs[1, 1]); style_ax(ax)

def boundary_points_v1(n, Lx, Ly, offset=0.0):
    perimeter = 2*(Lx+Ly); pts=[]
    for i in range(n):
        d = (i*perimeter/n + offset) % perimeter
        if   d < Lx:        pts.append((d*Lx/Lx, 0.0))
        elif d < Lx+Ly:     pts.append((Lx,       d-Lx))
        elif d < 2*Lx+Ly:   pts.append((2*Lx+Ly-d, Ly))
        else:               pts.append((0.0, perimeter-d))
    return np.array(pts)

src_v1 = boundary_points_v1(24, 30., 30., offset=0.0)
rec_v1 = boundary_points_v1(24, 30., 30., offset=15.0)  # old: offset=Lx/2

# Old hit count on 30×30
hit_v1 = np.zeros((30,30)); dx1=30/30; dy1=30/30
xc1=np.linspace(dx1/2,30-dx1/2,30); yc1=np.linspace(dy1/2,30-dy1/2,30)
for si in src_v1:
    for ri in rec_v1:
        rl=np.hypot(ri[0]-si[0],ri[1]-si[1]); ns=max(int(rl/dx1*8),12)
        ts=np.linspace(0,1,ns); xm=si[0]+ts*(ri[0]-si[0]); ym=si[1]+ts*(ri[1]-si[1])
        ix=np.floor(xm/dx1).astype(int); iy=np.floor(ym/dy1).astype(int)
        ok=(ix>=0)&(ix<30)&(iy>=0)&(iy<30)
        np.add.at(hit_v1,(iy[ok],ix[ok]),1)
XX1,YY1=np.meshgrid(xc1,yc1)

ax.pcolormesh(XX1, YY1, hit_v1/hit_v1.max(), cmap='hot', shading='auto', vmin=0, vmax=1)
ax.scatter(src_v1[:,0], src_v1[:,1], c='#ff7b72', s=20, marker='^', zorder=5, label='v1 src (24)')
ax.scatter(rec_v1[:,0], rec_v1[:,1], c='#79c0ff', s=20, marker='v', zorder=5, label='v1 rec (24)')

q1_v1=hit_v1[:15,:15]; q2_v1=hit_v1[:15,15:]; q3_v1=hit_v1[15:,:15]; q4_v1=hit_v1[15:,15:]
sym_v1 = 1 - np.std([q1_v1.mean(),q2_v1.mean(),q3_v1.mean(),q4_v1.mean()])/\
             np.mean([q1_v1.mean(),q2_v1.mean(),q3_v1.mean(),q4_v1.mean()])
for q, (tx,ty) in zip([q1_v1,q2_v1,q3_v1,q4_v1],
                       [(7,7),(23,7),(7,23),(23,23)]):
    ax.text(tx,ty,f'{q.mean():.0f}',color=TEXT,fontsize=8,ha='center',va='center',
            fontweight='bold', bbox=dict(fc=PANEL,ec=BORDER,boxstyle='round,pad=0.2',alpha=0.8))
ax.axhline(15, color=BORDER, lw=1.0, ls=':')
ax.axvline(15, color=BORDER, lw=1.0, ls=':')
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]')
ax.set_title(f'C — v1 Coverage  (symmetry score = {sym_v1:.4f})',
             fontsize=9, fontweight='bold', pad=6)
ax.set_aspect('equal'); ax.set_xlim(0,30); ax.set_ylim(0,30)

# ── D: Observed travel-time matrix ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2]); style_ax(ax)
im = ax.imshow(d_obs.reshape(n_src, n_rec), cmap='plasma', aspect='auto', origin='lower')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('t_obs [s]', color=MUTED, fontsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
ax.set_xlabel('Receiver index'); ax.set_ylabel('Source index')
ax.set_title('D — Observed Travel Times d_obs', fontsize=9, fontweight='bold', pad=6)

# ── E: Noise residuals ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2]); style_ax(ax)
from scipy.stats import norm as spnorm
residuals = d_obs - d_true
ax.hist(residuals, bins=50, color='#58a6ff', edgecolor=PANEL, lw=0.4,
        density=True, alpha=0.85)
xs = np.linspace(residuals.min(), residuals.max(), 300)
ax.plot(xs, spnorm.pdf(xs, 0, noise_std), color='#ff7b72', lw=1.8,
        label=f'N(0, σ={noise_std:.4f})')
ax.set_xlabel('Residual [s]'); ax.set_ylabel('Density')
ax.set_title('E — Noise residuals', fontsize=9, fontweight='bold', pad=6)
ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

fig.suptitle('Seismic Tomography Surrogate Dataset v2  —  Symmetric 48×48 Geometry',
             color=TEXT, fontsize=13, fontweight='bold', y=0.97)

plt.savefig(FIG_DIR / 'surrogate_dataset.png',
            dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
print(f"Saved → {FIG_DIR / 'surrogate_dataset.png'}")
print("\nDone.")
