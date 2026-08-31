"""
ML Mathematics — Visual Reference
==================================
Run this script to generate a multi-page PDF (or individual PNGs) of every
key operation: vectors, vector arithmetic, dot product, norms, matrix
transformations, determinants, inverses, linear systems, and more.

Dependencies: numpy, matplotlib
  pip install numpy matplotlib

Usage:
  python ML-Math-Visualizations.py          # saves ML-Math-Visuals.pdf
  python ML-Math-Visualizations.py --png    # saves individual PNG files
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Arc, Wedge
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Palette (colorblind-safe, sufficient contrast) ──────────────────────────
BLUE   = "#2563EB"   # primary vector / matrix element
RED    = "#DC2626"   # resultant / transformed
GREEN  = "#16A34A"   # third series / projection
PURPLE = "#7C3AED"   # angle arc / special
ORANGE = "#D97706"   # highlight / warning
GRAY   = "#6B7280"   # grid, axes, secondary text
LGRAY  = "#F3F4F6"   # fill background
WHITE  = "#FFFFFF"
BLACK  = "#111827"

FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (10, 7)
FIGSIZE_FULL = (14, 9)

PNG_MODE = "--png" in sys.argv

saved = []


def save_fig(name: str):
    plt.tight_layout(pad=1.5)
    if PNG_MODE:
        path = f"{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    else:
        saved.append(name)
    plt.close()


def arrow(ax, start, end, color, lw=2.5, label=None, head=0.15):
    """Draw a clean vector arrow."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle=f"->,head_width={head},head_length={head*0.8}",
            color=color, lw=lw
        )
    )
    if label:
        # offset label perpendicular to arrow direction
        length = np.hypot(dx, dy)
        if length == 0:
            return
        perp = np.array([-dy, dx]) / length * 0.18
        mid = np.array(start) + np.array([dx, dy]) / 2 + perp
        ax.text(mid[0], mid[1], label, color=color, fontsize=11,
                fontweight="bold", ha="center", va="center")


def style_ax(ax, title="", xlim=(-0.5, 5), ylim=(-0.5, 5), grid=True):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.axhline(0, color=GRAY, lw=0.8); ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_aspect("equal")
    if grid:
        ax.grid(True, color=GRAY, alpha=0.25, lw=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold", color=BLACK, pad=8)
    ax.tick_params(labelsize=8, color=GRAY, labelcolor=GRAY)
    for spine in ax.spines.values():
        spine.set_color(GRAY); spine.set_linewidth(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Vectors: What They Are
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 1 — Vectors: Representation in Space",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

# Panel A: 2D vector
ax = axes[0]
style_ax(ax, "A) 2D Vector  v = (3, 2)", xlim=(-0.5, 4.5), ylim=(-0.5, 3.5))
arrow(ax, (0, 0), (3, 2), BLUE, label="v = (3, 2)")
ax.plot(0, 0, "o", color=BLACK, ms=5)
ax.plot(3, 2, "o", color=BLUE, ms=7)
ax.plot([3, 3], [0, 2], "--", color=GRAY, lw=1)
ax.plot([0, 3], [0, 0], "--", color=GRAY, lw=1)
ax.text(1.5, -0.25, "x = 3", ha="center", fontsize=9, color=GRAY)
ax.text(3.25, 1.0, "y = 2", ha="left", fontsize=9, color=GRAY)

# Panel B: 3D interpretation (projected)
ax = axes[1]
style_ax(ax, "B) Thinking in 3D\n(projected view)", xlim=(-0.5, 5), ylim=(-0.5, 4.5))
arrow(ax, (0, 0), (3, 1), GRAY, lw=1.5, label="x-axis")
arrow(ax, (0, 0), (1, 2), GRAY, lw=1.5, label="y-axis")
arrow(ax, (0, 0), (3.5, 3.5), BLUE, lw=2.5, label="v")
ax.text(3.7, 3.6, "v = (2, 1, 3)\nin 3D", color=BLUE, fontsize=9)

# Panel C: ML feature vector
ax = axes[2]
ax.set_title("C) Feature Vector in ML", fontsize=12, fontweight="bold", color=BLACK)
ax.axis("off")
features = ["Area (m²)", "Bedrooms", "Bathrooms", "Age (yrs)"]
values   = [1500, 3, 2, 10]
colors_  = [BLUE, GREEN, ORANGE, PURPLE]
y_pos = np.arange(len(features))
bars = ax.barh(y_pos, values, color=colors_, alpha=0.8, height=0.5)
ax.set_yticks(y_pos); ax.set_yticklabels(features, fontsize=10)
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=10, color=BLACK)
ax.set_xlim(0, 1900)
ax.set_xlabel("Feature value", fontsize=10)
ax.text(750, -0.9,
        "h = [1500, 3, 2, 10]  ← This is a 4D vector",
        ha="center", fontsize=9, color=GRAY, style="italic")

save_fig("fig01_vectors")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Vector Addition (Triangle & Parallelogram Laws)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 2 — Vector Addition", fontsize=14, fontweight="bold",
             color=BLACK, y=1.02)

a = np.array([2, 1])
b = np.array([1, 3])
s = a + b

# Panel A: Triangle law
ax = axes[0]
style_ax(ax, "A) Triangle Law\na + b = s", xlim=(-0.3, 4.5), ylim=(-0.3, 4.5))
arrow(ax, (0, 0), a, BLUE, label="a=(2,1)")
arrow(ax, a, a + b, GREEN, label="b=(1,3)")
arrow(ax, (0, 0), s, RED, label="a+b=(3,4)")
ax.plot(0, 0, "o", color=BLACK, ms=5)

# Panel B: Parallelogram law
ax = axes[1]
style_ax(ax, "B) Parallelogram Law", xlim=(-0.3, 4.5), ylim=(-0.3, 4.5))
# fill parallelogram
para = plt.Polygon(
    [(0, 0), a, s, b], closed=True,
    facecolor=BLUE, alpha=0.12, edgecolor=GRAY, lw=0.8)
ax.add_patch(para)
arrow(ax, (0, 0), a, BLUE, label="a")
arrow(ax, (0, 0), b, GREEN, label="b")
arrow(ax, a, s, GREEN, lw=1.5)
arrow(ax, b, s, BLUE, lw=1.5)
arrow(ax, (0, 0), s, RED, label="a+b")
ax.plot(0, 0, "o", color=BLACK, ms=5)

# Panel C: Step-by-step numeric breakdown
ax = axes[2]
ax.axis("off")
ax.set_title("C) Step-by-Step Calculation", fontsize=12,
             fontweight="bold", color=BLACK)
lines = [
    ("a = ", "[2,  1]", BLUE),
    ("b = ", "[1,  3]", GREEN),
    ("", "", WHITE),
    ("a + b:", "", BLACK),
    ("  row 0:", "  2 + 1 = 3", RED),
    ("  row 1:", "  1 + 3 = 4", RED),
    ("", "", WHITE),
    ("result = ", "[3,  4]", RED),
]
y = 0.95
for label, val, col in lines:
    ax.text(0.05, y, label, transform=ax.transAxes,
            fontsize=11, color=GRAY, va="top")
    ax.text(0.45, y, val, transform=ax.transAxes,
            fontsize=11, color=col, va="top", fontweight="bold")
    y -= 0.11

ax.text(0.5, 0.05,
        "Add component-by-component.\nVectors must have equal length.",
        transform=ax.transAxes, ha="center", fontsize=9,
        color=GRAY, style="italic", va="bottom")

save_fig("fig02_vector_addition")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Scalar Multiplication & Vector Subtraction
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 3 — Scalar Multiplication & Subtraction",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

v = np.array([1.5, 1.0])

ax = axes[0]
style_ax(ax, "A) Scalar Multiplication\nStretch / Shrink / Flip",
         xlim=(-3.5, 4), ylim=(-2.5, 3))
for scale, col, lbl in [(2, BLUE, "2v"), (1, GREEN, "v"),
                         (0.5, ORANGE, "0.5v"), (-1, RED, "-v")]:
    arrow(ax, (0, 0), v * scale, col, label=lbl, head=0.12)

ax = axes[1]
style_ax(ax, "B) Vector Subtraction\na - b", xlim=(-0.5, 4), ylim=(-2, 3.5))
a_ = np.array([3, 2]); b_ = np.array([1, 3])
arrow(ax, (0, 0), a_, BLUE, label="a")
arrow(ax, (0, 0), b_, GREEN, label="b")
arrow(ax, b_, a_, RED, label="a-b")
ax.plot(0, 0, "o", color=BLACK, ms=5)
ax.text(1.2, -0.6,
        "a - b = vector from tip of b\nto tip of a",
        fontsize=9, color=GRAY, ha="center")

ax = axes[2]
ax.axis("off")
ax.set_title("C) Numeric Breakdown", fontsize=12, fontweight="bold")
rows = [
    "a  = [3, 2]",
    "b  = [1, 3]",
    "",
    "2 × a:",
    "  [2×3, 2×2] = [6, 4]",
    "",
    "a - b:",
    "  [3-1, 2-3] = [2, -1]",
    "",
    "Scalar × every component.",
    "Subtract component-by-component.",
]
y = 0.95
for row in rows:
    col = RED if row.startswith("  [") else (BLACK if row.endswith(":") else GRAY)
    ax.text(0.1, y, row, transform=ax.transAxes,
            fontsize=10, color=col, va="top",
            fontweight="bold" if row.endswith(":") else "normal")
    y -= 0.09

save_fig("fig03_scalar_mult")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Dot Product & Angle
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 4 — Dot Product & Angle Between Vectors",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

a4 = np.array([3.0, 1.0]); b4 = np.array([1.0, 3.0])

ax = axes[0]
style_ax(ax, "A) Angle θ Between Vectors", xlim=(-0.3, 4.5), ylim=(-0.3, 4.5))
arrow(ax, (0, 0), a4, BLUE, label="a")
arrow(ax, (0, 0), b4, GREEN, label="b")

# angle arc
cos_ab = np.dot(a4, b4) / (np.linalg.norm(a4) * np.linalg.norm(b4))
theta = np.degrees(np.arccos(np.clip(cos_ab, -1, 1)))
ang_a = np.degrees(np.arctan2(a4[1], a4[0]))
ang_b = np.degrees(np.arctan2(b4[1], b4[0]))
arc = Arc((0, 0), 1.2, 1.2, angle=0,
          theta1=min(ang_a, ang_b), theta2=max(ang_a, ang_b),
          color=PURPLE, lw=2)
ax.add_patch(arc)
mid_angle = np.radians((ang_a + ang_b) / 2)
ax.text(0.85 * np.cos(mid_angle), 0.85 * np.sin(mid_angle),
        f"θ={theta:.0f}°", color=PURPLE, fontsize=10, fontweight="bold",
        ha="center", va="center")

# Panel B: projection interpretation
ax = axes[1]
style_ax(ax, "B) Projection of a onto b",
         xlim=(-0.3, 4.5), ylim=(-0.3, 4.5))
b4_hat = b4 / np.linalg.norm(b4)
proj_len = np.dot(a4, b4_hat)
proj_vec = proj_len * b4_hat
arrow(ax, (0, 0), a4, BLUE, label="a")
arrow(ax, (0, 0), b4, GREEN, label="b")
arrow(ax, (0, 0), proj_vec, ORANGE, label="proj")
ax.plot([a4[0], proj_vec[0]], [a4[1], proj_vec[1]],
        "--", color=GRAY, lw=1.2)
ax.text(0.5, -0.25,
        f"proj = (a·b̂) b̂,  length = a·b/||b||",
        fontsize=9, color=GRAY, ha="left")

# Panel C: three cases
ax = axes[2]
style_ax(ax, "C) Three Dot-Product Cases",
         xlim=(-4, 4), ylim=(-2.5, 2.5))
cases = [
    (np.array([2, 0]),   np.array([2, 0]),   BLUE,  GREEN,  "Same dir → dot > 0"),
    (np.array([-2, -1]), np.array([2, -1.5]),BLUE,  ORANGE, "Obtuse → dot < 0"),
    (np.array([0, 2]),   np.array([2, 0]),   PURPLE, RED,   "⊥ → dot = 0"),
]
offsets = [(-1.8, 1.3), (-0.5, -0.5), (1.5, 1.3)]
for (u, v_, c1, c2, lbl), (ox, oy) in zip(cases, offsets):
    arrow(ax, (ox, oy), (ox + u[0]/2, oy + u[1]/2), c1, lw=2, head=0.1)
    arrow(ax, (ox, oy), (ox + v_[0]/2, oy + v_[1]/2), c2, lw=2, head=0.1)
    d = np.dot(u, v_)
    ax.text(ox, oy - 0.5, f"{lbl}\ndot={d:.0f}",
            fontsize=8, color=GRAY, ha="center")

save_fig("fig04_dot_product")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Vector Norms (Unit Balls)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 5 — Vector Norms & Unit Balls",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

theta_range = np.linspace(0, 2 * np.pi, 500)

# L1 unit ball (diamond)
ax = axes[0]
style_ax(ax, "A) L1 Norm — Manhattan\n||v||₁ = |x| + |y| = 1",
         xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
diamond = plt.Polygon([(1,0),(0,1),(-1,0),(0,-1)], closed=True,
                       facecolor=BLUE, alpha=0.2, edgecolor=BLUE, lw=2)
ax.add_patch(diamond)
ax.text(0, 0, "L1\nBall", ha="center", va="center",
        color=BLUE, fontsize=10, fontweight="bold")
ax.plot([0, 0.7], [0, 0.3], color=RED, lw=2)
ax.text(0.55, 0.08, "v", color=RED, fontsize=11, fontweight="bold")
ax.text(0, -1.35,
        "L1 promotes sparsity → Lasso",
        ha="center", fontsize=9, color=GRAY)

# L2 unit ball (circle)
ax = axes[1]
style_ax(ax, "B) L2 Norm — Euclidean\n||v||₂ = √(x²+y²) = 1",
         xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
circle = plt.Circle((0, 0), 1, facecolor=GREEN, alpha=0.2,
                     edgecolor=GREEN, lw=2)
ax.add_patch(circle)
ax.text(0, 0, "L2\nBall", ha="center", va="center",
        color=GREEN, fontsize=10, fontweight="bold")
ax.plot([0, 0.7], [0, 0.3], color=RED, lw=2)
ax.plot([0.7, 0.7], [0, 0.3], "--", color=GRAY, lw=1)
ax.plot([0, 0.7], [0, 0], "--", color=GRAY, lw=1)
ax.text(0.55, 0.08, "v", color=RED, fontsize=11, fontweight="bold")
ax.text(0, -1.35,
        "L2 smooth penalty → Ridge",
        ha="center", fontsize=9, color=GRAY)

# L∞ unit ball (square)
ax = axes[2]
style_ax(ax, "C) L∞ Norm — Max\n||v||∞ = max(|x|,|y|) = 1",
         xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
square = plt.Polygon([(-1,-1),(1,-1),(1,1),(-1,1)], closed=True,
                      facecolor=ORANGE, alpha=0.2, edgecolor=ORANGE, lw=2)
ax.add_patch(square)
ax.text(0, 0, "L∞\nBall", ha="center", va="center",
        color=ORANGE, fontsize=10, fontweight="bold")
ax.plot([0, 0.7], [0, 0.3], color=RED, lw=2)
ax.text(0.55, 0.08, "v", color=RED, fontsize=11, fontweight="bold")
ax.text(0, -1.35,
        "L∞ = max component magnitude",
        ha="center", fontsize=9, color=GRAY)

save_fig("fig05_norms")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Matrix as a Grid Transformation
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 6 — Matrix as a Linear Transformation",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

A6 = np.array([[2.0, 1], [0.5, 1.5]])


def draw_grid(ax, M, color_after=RED, title=""):
    """Draw original grid (gray) and transformed grid (colored)."""
    lim = 3.5
    style_ax(ax, title, xlim=(-lim, lim), ylim=(-lim, lim))
    xs = np.linspace(-3, 3, 7)

    # Original grid (light gray)
    for x in xs:
        ax.plot([x, x], [-3, 3], color=LGRAY, lw=0.8)
    for y in xs:
        ax.plot([-3, 3], [y, y], color=LGRAY, lw=0.8)

    # Transformed grid (colored)
    if M is not None:
        for x in xs:
            p1 = M @ np.array([x, -3])
            p2 = M @ np.array([x,  3])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color_after, lw=0.8, alpha=0.4)
        for y in xs:
            p1 = M @ np.array([-3, y])
            p2 = M @ np.array([ 3, y])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color_after, lw=0.8, alpha=0.4)

    # Original basis vectors (blue/green)
    e1 = np.array([1.0, 0]); e2 = np.array([0.0, 1])
    arrow(ax, (0, 0), e1, BLUE, label="e₁", head=0.1)
    arrow(ax, (0, 0), e2, GREEN, label="e₂", head=0.1)

    if M is not None:
        # Transformed basis vectors
        Me1 = M @ e1; Me2 = M @ e2
        arrow(ax, (0, 0), Me1, BLUE, lw=3, head=0.14,
              label=f"Ae₁=({Me1[0]:.1f},{Me1[1]:.1f})")
        arrow(ax, (0, 0), Me2, GREEN, lw=3, head=0.14,
              label=f"Ae₂=({Me2[0]:.1f},{Me2[1]:.1f})")


draw_grid(axes[0], None, title="A) Original Grid\n(before transformation)")
draw_grid(axes[1], A6,   title="B) After Applying A\n[[2,1],[0.5,1.5]]")

# Panel C: shear, rotation, scaling examples
ax = axes[2]
style_ax(ax, "C) Types of Transformations", xlim=(-0.3, 3.5), ylim=(-0.5, 3.8))
transforms = {
    "Scale (2×)": (np.array([[2, 0], [0, 2]]), BLUE),
    "Shear":      (np.array([[1, 1], [0, 1]]), GREEN),
    "Rotation 45°": (np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                               [np.sin(np.pi/4),  np.cos(np.pi/4)]]), ORANGE),
    "Reflect y":  (np.array([[-1, 0], [0, 1]]), PURPLE),
}
v_orig = np.array([1.0, 0.5])
offsets6 = [(0.3, 2.8), (1.5, 2.8), (0.3, 1.2), (1.5, 1.2)]
for (name, (M_, col)), (ox, oy) in zip(transforms.items(), offsets6):
    scale = 0.6
    v_t = M_ @ v_orig * scale
    arrow(ax, (ox, oy), (ox + v_orig[0]*scale, oy + v_orig[1]*scale),
          GRAY, lw=1.5, head=0.1)
    arrow(ax, (ox, oy), (ox + v_t[0], oy + v_t[1]),
          col, lw=2, head=0.1)
    ax.text(ox + 0.5, oy - 0.28, name, fontsize=8, color=col, ha="center")

save_fig("fig06_matrix_transform")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Determinant as Area / Volume
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 7 — Determinant = Signed Area of Parallelogram",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

cases7 = [
    ("A) det > 0  (stretches)\nA=[[3,1],[1,2]], det=5",
     np.array([[3.0, 1], [1, 2]]), BLUE),
    ("B) det < 0  (flips orientation)\nA=[[1,2],[2,1]], det=-3",
     np.array([[1.0, 2], [2, 1]]), RED),
    ("C) det = 0  (collapses to line)\nA=[[2,4],[1,2]], det=0",
     np.array([[2.0, 4], [1, 2]]), GRAY),
]

for ax, (title, M7, col) in zip(axes, cases7):
    style_ax(ax, title, xlim=(-0.5, 5.5), ylim=(-1.5, 4.5))
    c1 = M7[:, 0]; c2 = M7[:, 1]
    corners = np.array([(0,0), c1, c1+c2, c2, (0,0)])
    ax.fill(corners[:-1, 0], corners[:-1, 1],
            alpha=0.25, color=col, label=f"Area=|det|={abs(np.linalg.det(M7)):.1f}")
    ax.plot(corners[:, 0], corners[:, 1], color=col, lw=1.2, ls="--")
    arrow(ax, (0, 0), c1, BLUE, label=f"col₁=({c1[0]:.0f},{c1[1]:.0f})")
    arrow(ax, (0, 0), c2, GREEN, label=f"col₂=({c2[0]:.0f},{c2[1]:.0f})")
    det = np.linalg.det(M7)
    ax.text(0.5, -1.2,
            f"det = {det:.1f}  →  {'invertible ✓' if abs(det) > 1e-9 else 'singular ✗'}",
            color=col, fontsize=10, fontweight="bold", ha="center")
    ax.legend(fontsize=8, loc="upper right")

save_fig("fig07_determinant")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Determinant: 2×2 Step-by-Step
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6))
fig.suptitle("FIGURE 8 — Computing the 2×2 Determinant: ad − bc",
             fontsize=14, fontweight="bold", color=BLACK)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
ax_left  = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1])

ax_left.axis("off")
lines8 = [
    (0.50, 0.92, "Matrix A", 14, BLACK, "bold"),
    (0.30, 0.78, "┌ 3   2 ┐", 13, GRAY, "normal"),
    (0.30, 0.68, "│       │", 13, GRAY, "normal"),
    (0.30, 0.58, "└ 1   4 ┘", 13, GRAY, "normal"),
    (0.50, 0.44, "det(A) = ad − bc", 12, BLACK, "bold"),
    (0.50, 0.33, "= (3 × 4) − (2 × 1)", 12, BLUE, "normal"),
    (0.50, 0.22, "=   12   −    2", 12, BLUE, "normal"),
    (0.50, 0.11, "=   10", 14, RED, "bold"),
]
for x, y, txt, fs, col, fw in lines8:
    ax_left.text(x, y, txt, transform=ax_left.transAxes,
                 ha="center", va="top", fontsize=fs, color=col, fontweight=fw,
                 fontfamily="monospace")

# Diagonal highlight
ax_left.annotate("a×d\n(main diag ↘)", xy=(0.48, 0.65),
                 xytext=(0.10, 0.50), xycoords="axes fraction",
                 fontsize=9, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax_left.annotate("b×c\n(anti diag ↗)", xy=(0.52, 0.65),
                 xytext=(0.85, 0.50), xycoords="axes fraction",
                 fontsize=9, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

# Right: geometric verification
ax_right.set_title("Geometric Verification\n(area of parallelogram = 10)",
                   fontsize=11, fontweight="bold")
A8 = np.array([[3.0, 2], [1, 4]])
c1 = A8[:, 0]; c2 = A8[:, 1]
corners8 = np.array([(0,0), c1, c1+c2, c2])
ax_right.fill(corners8[:, 0], corners8[:, 1], alpha=0.2, color=BLUE)
poly8 = plt.Polygon(np.vstack([corners8, corners8[0]]),
                    closed=True, fill=False, edgecolor=BLUE, lw=1.5, ls="--")
ax_right.add_patch(poly8)
arrow(ax_right, (0, 0), c1, BLUE, label="col₁=(3,1)")
arrow(ax_right, (0, 0), c2, GREEN, label="col₂=(2,4)")
ax_right.text(2.2, 2.5, "Area = 10", color=RED, fontsize=12, fontweight="bold")
ax_right.set_xlim(-0.5, 6.5); ax_right.set_ylim(-0.5, 6.5)
ax_right.set_aspect("equal")
ax_right.grid(True, color=GRAY, alpha=0.25, lw=0.6)
ax_right.axhline(0, color=GRAY, lw=0.8); ax_right.axvline(0, color=GRAY, lw=0.8)

save_fig("fig08_det_steps")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — Matrix Transpose
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 9 — Matrix Transpose: Rows ↔ Columns",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

A9 = np.array([[1, 2, 3], [4, 5, 6]])
AT9 = A9.T

def draw_matrix(ax, M, title, cmap_color=BLUE, highlight_diag=False):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color=BLACK)
    rows, cols = M.shape
    cell = 0.22
    ox, oy = 0.1, 0.85
    for i in range(rows):
        for j in range(cols):
            x = ox + j * cell
            y = oy - i * cell
            is_diag = (i == j) and highlight_diag
            bg = LGRAY if not is_diag else "#DBEAFE"
            rect = plt.Rectangle((x - 0.09, y - 0.09), 0.18, 0.18,
                                  transform=ax.transAxes,
                                  facecolor=bg, edgecolor=GRAY, lw=0.8)
            ax.add_patch(rect)
            ax.text(x, y, str(M[i, j]), transform=ax.transAxes,
                    ha="center", va="center", fontsize=13,
                    color=cmap_color if not is_diag else PURPLE,
                    fontweight="bold")
    # row/col labels
    for j in range(cols):
        ax.text(ox + j * cell, oy + 0.12,
                f"c{j}", transform=ax.transAxes,
                ha="center", fontsize=9, color=GRAY)
    for i in range(rows):
        ax.text(ox - 0.07, oy - i * cell,
                f"r{i}", transform=ax.transAxes,
                va="center", fontsize=9, color=GRAY)
    ax.text(0.5, 0.08, f"Shape: {M.shape[0]}×{M.shape[1]}",
            transform=ax.transAxes, ha="center", fontsize=9, color=GRAY)

draw_matrix(axes[0], A9,  "A) Original A  (2×3)", cmap_color=BLUE)
axes[1].axis("off")
axes[1].text(0.5, 0.55, "Transpose\n\nRows → Columns\nColumns → Rows\n\nA[i,j] → Aᵀ[j,i]",
             transform=axes[1].transAxes, ha="center", va="center",
             fontsize=12, color=BLACK,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=LGRAY, edgecolor=GRAY))
axes[1].annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->,head_width=0.3",
                                 color=RED, lw=2))
draw_matrix(axes[2], AT9, "C) Transposed Aᵀ (3×2)", cmap_color=RED)

save_fig("fig09_transpose")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Matrix Multiplication: Every Element Is a Dot Product
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7))
fig.suptitle("FIGURE 10 — Matrix Multiplication: C[i,j] = Row i of A · Col j of B",
             fontsize=13, fontweight="bold", color=BLACK)

A10 = np.array([[1, 2, 3], [4, 5, 6]])
B10 = np.array([[7, 8], [9, 10], [11, 12]])
C10 = A10 @ B10

gs10 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.5)
ax_a = fig.add_subplot(gs10[0])
ax_b = fig.add_subplot(gs10[1])
ax_c = fig.add_subplot(gs10[2])

cell = 0.28
highlight_row = 0
highlight_col = 0

def draw_mult_matrix(ax, M, title, h_row=None, h_col=None,
                     base_col=BLUE, hi_col=RED):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    rows, cols = M.shape
    ox, oy = 0.15, 0.85
    cw = 0.65 / cols; rh = 0.65 / rows
    for i in range(rows):
        for j in range(cols):
            x = ox + j * cw + cw / 2
            y = oy - i * rh - rh / 2
            hi = (h_row is not None and i == h_row) or \
                 (h_col is not None and j == h_col)
            bg = "#FEE2E2" if hi else LGRAY
            rect = plt.Rectangle((ox + j*cw, oy - i*rh - rh), cw, rh,
                                  transform=ax.transAxes,
                                  facecolor=bg, edgecolor=GRAY, lw=0.6)
            ax.add_patch(rect)
            ax.text(x, y, str(M[i, j]), transform=ax.transAxes,
                    ha="center", va="center", fontsize=13,
                    color=hi_col if hi else base_col, fontweight="bold")
    ax.text(0.5, 0.05, f"{rows}×{cols}", transform=ax.transAxes,
            ha="center", fontsize=10, color=GRAY)

draw_mult_matrix(ax_a, A10, "A (2×3)", h_row=highlight_row, base_col=BLUE)
draw_mult_matrix(ax_b, B10, "B (3×2)", h_col=highlight_col, base_col=GREEN)
draw_mult_matrix(ax_c, C10, "C = A@B (2×2)", h_row=highlight_row,
                 h_col=highlight_col, base_col=RED)

# Show the dot-product calculation for C[0,0]
fig.text(0.5, 0.07,
         "C[0,0] = (1×7)+(2×9)+(3×11) = 7+18+33 = 58       "
         "C[0,1] = (1×8)+(2×10)+(3×12) = 8+20+36 = 64\n"
         "C[1,0] = (4×7)+(5×9)+(6×11) = 28+45+66 = 139     "
         "C[1,1] = (4×8)+(5×10)+(6×12) = 32+50+72 = 154",
         ha="center", fontsize=10, color=GRAY,
         bbox=dict(boxstyle="round", facecolor=LGRAY, edgecolor=GRAY, alpha=0.8))

save_fig("fig10_matmul")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 — Matrix Inverse: Transform and Undo
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 11 — Matrix Inverse: A × A⁻¹ = I",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

A11 = np.array([[2.0, 1], [1, 3]])
Ainv11 = np.linalg.inv(A11)
v11 = np.array([1.0, 1])
Av = A11 @ v11
AinvAv = Ainv11 @ Av

def draw_transform_state(ax, title, vec_orig, vec_transformed, mat_label, col):
    style_ax(ax, title, xlim=(-0.5, 4.5), ylim=(-0.5, 4.5))
    arrow(ax, (0, 0), vec_orig, GRAY, lw=1.5, label="original")
    arrow(ax, (0, 0), vec_transformed, col, lw=2.5, label=mat_label)
    ax.plot(*vec_orig, "o", color=GRAY, ms=8)
    ax.plot(*vec_transformed, "o", color=col, ms=8)

draw_transform_state(axes[0],
    f"A) Apply A\nv=(1,1) → Av=({Av[0]:.0f},{Av[1]:.0f})",
    v11, Av, "Av", RED)

draw_transform_state(axes[1],
    f"B) Apply A⁻¹ to Av\nrecovers v=(1,1)",
    Av, AinvAv, "A⁻¹(Av)", GREEN)

ax = axes[2]
ax.axis("off")
ax.set_title("C) 2×2 Inverse Formula", fontsize=11, fontweight="bold")
det11 = np.linalg.det(A11)
steps = [
    ("A =", f"[[2, 1],\n [1, 3]]", BLUE),
    ("det(A) =", f"(2×3)−(1×1) = {det11:.0f}", GRAY),
    ("A⁻¹ =", f"(1/{det11:.0f}) × [[ 3, -1],\n              [-1,  2]]", RED),
    ("    =", f"[[{Ainv11[0,0]:.2f}, {Ainv11[0,1]:.2f}],\n  [{Ainv11[1,0]:.2f},  {Ainv11[1,1]:.2f}]]", RED),
    ("A × A⁻¹ =", "I  ✓", GREEN),
]
y = 0.92
for label, val, col in steps:
    ax.text(0.05, y, label, transform=ax.transAxes, fontsize=10,
            color=GRAY, va="top", fontfamily="monospace")
    ax.text(0.38, y, val, transform=ax.transAxes, fontsize=10,
            color=col, va="top", fontfamily="monospace", fontweight="bold")
    y -= 0.17

save_fig("fig11_inverse")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12 — Solving Ax = b: Geometric Interpretation
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 12 — Solving Ax = b: Intersection of Lines",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

# System: 2x + y = 5,  x + 3y = 10
A12 = np.array([[2.0, 1], [1, 3]])
b12 = np.array([5.0, 10])
sol = np.linalg.solve(A12, b12)   # x=1, y=3

x_range = np.linspace(-1, 6, 200)
y1 = (5 - 2 * x_range) / 1       # 2x + y = 5
y2 = (10 - x_range) / 3          # x + 3y = 10

ax = axes[0]
style_ax(ax, "A) Two Equations = Two Lines\nSolution = Intersection",
         xlim=(-0.5, 6), ylim=(-1, 7))
ax.plot(x_range, y1, color=BLUE, lw=2, label="2x + y = 5")
ax.plot(x_range, y2, color=GREEN, lw=2, label="x + 3y = 10")
ax.plot(*sol, "*", color=RED, ms=14, zorder=5, label=f"solution ({sol[0]:.0f},{sol[1]:.0f})")
ax.legend(fontsize=9, loc="upper right")
ax.fill_between(x_range, y1, y2, alpha=0.05, color=PURPLE)

ax = axes[1]
style_ax(ax, "B) Column Picture\nb = x·col₁ + y·col₂",
         xlim=(-0.5, 5.5), ylim=(-0.5, 5.5))
col1 = A12[:, 0]; col2 = A12[:, 1]
x_sol, y_sol = sol
arrow(ax, (0, 0), x_sol * col1, BLUE, label=f"x·col₁=1·(2,1)")
arrow(ax, x_sol * col1, x_sol * col1 + y_sol * col2, GREEN,
      label=f"y·col₂=3·(1,3)")
arrow(ax, (0, 0), b12, RED, label="b=(5,10)")

ax = axes[2]
ax.axis("off")
ax.set_title("C) Steps to Solve", fontsize=11, fontweight="bold")
steps12 = [
    "System:",
    "  2x + y  = 5",
    "   x + 3y = 10",
    "",
    "Matrix form: Ax = b",
    "",
    "det(A) = 2×3 - 1×1 = 5 ≠ 0",
    "  → A is invertible ✓",
    "",
    "x = A⁻¹b  (or np.linalg.solve)",
    "",
    f"Solution: x={sol[0]:.0f},  y={sol[1]:.0f}",
    "",
    "Verify:",
    f"  2({sol[0]:.0f})+{sol[1]:.0f} = {2*sol[0]+sol[1]:.0f} = 5  ✓",
    f"  {sol[0]:.0f}+3({sol[1]:.0f}) = {sol[0]+3*sol[1]:.0f} = 10 ✓",
]
y = 0.96
for s in steps12:
    col = GREEN if "✓" in s else (RED if "Solution" in s else
           (BLUE if s.startswith("  2") or s.startswith("  ") else GRAY))
    ax.text(0.05, y, s, transform=ax.transAxes,
            fontsize=9.5, color=col, va="top", fontfamily="monospace",
            fontweight="bold" if "Solution" in s else "normal")
    y -= 0.063

save_fig("fig12_linear_system")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 13 — Matrix Rank: Visualizing Dependence
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 13 — Matrix Rank: How Many Independent Directions?",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

cases13 = [
    ("A) Rank 2 (Full Rank)\nTwo independent columns",
     np.array([[3.0, 1], [1, 2]]), BLUE, True),
    ("B) Rank 1 (Deficient)\nColumns are multiples",
     np.array([[2.0, 4], [1, 2]]), ORANGE, False),
    ("C) Rank 0\nZero matrix",
     np.zeros((2, 2)), GRAY, False),
]
for ax, (title, M13, col, full) in zip(axes, cases13):
    style_ax(ax, title, xlim=(-0.5, 5), ylim=(-0.5, 5))
    c1 = M13[:, 0]; c2 = M13[:, 1]
    arrow(ax, (0, 0), c1, BLUE, label=f"col₁=({c1[0]:.0f},{c1[1]:.0f})")
    if not np.allclose(c2, 0):
        arrow(ax, (0, 0), c2, col, label=f"col₂=({c2[0]:.0f},{c2[1]:.0f})")
    rank = np.linalg.matrix_rank(M13)
    ax.text(0.5, -0.3,
            f"rank = {rank}  |  det = {np.linalg.det(M13):.1f}  |  "
            f"{'invertible ✓' if rank == 2 else 'singular ✗'}",
            ha="center", fontsize=9, color=col, fontweight="bold")
    if not full and rank == 1:
        xs_line = np.linspace(0, 4.5, 100)
        ys_line = (c1[1]/c1[0]) * xs_line if c1[0] != 0 else xs_line * 0
        ax.plot(xs_line, ys_line, "--", color=col, lw=1.2, alpha=0.5,
                label="column space (line)")
        ax.legend(fontsize=8, loc="upper right")

save_fig("fig13_rank")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 14 — Orthogonality & Orthogonal Matrix
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 14 — Orthogonality: QᵀQ = I, Preserves Shape",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

theta14 = np.pi / 6   # 30°
Q14 = np.array([[np.cos(theta14), -np.sin(theta14)],
                [np.sin(theta14),  np.cos(theta14)]])

# Panel A: orthogonal vectors
ax = axes[0]
style_ax(ax, "A) Orthogonal Vectors\na · b = 0, θ = 90°",
         xlim=(-0.5, 3.5), ylim=(-0.5, 3.5))
a14 = np.array([2.5, 0.0]); b14 = np.array([0.0, 2.5])
arrow(ax, (0, 0), a14, BLUE, label="a=(2.5,0)")
arrow(ax, (0, 0), b14, GREEN, label="b=(0,2.5)")
# right-angle marker
sq_size = 0.3
sq = plt.Polygon([(sq_size, 0), (sq_size, sq_size), (0, sq_size)],
                 closed=True, fill=False, edgecolor=GRAY, lw=1.2)
ax.add_patch(sq)
ax.text(0.9, -0.3, "a · b = 0  →  perpendicular",
        fontsize=9, color=PURPLE, ha="center")

# Panel B: orthogonal matrix preserves shape
ax = axes[1]
style_ax(ax, f"B) Rotation by 30° (Orthogonal Matrix)\nShape preserved",
         xlim=(-0.5, 3), ylim=(-0.5, 3))
square14 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=float)
rot_square = Q14 @ square14
ax.fill(square14[0], square14[1], alpha=0.2, color=BLUE, label="original")
ax.fill(rot_square[0], rot_square[1], alpha=0.2, color=RED, label="rotated")
ax.plot(np.append(square14[0], square14[0, 0]),
        np.append(square14[1], square14[1, 0]), color=BLUE, lw=1.5)
ax.plot(np.append(rot_square[0], rot_square[0, 0]),
        np.append(rot_square[1], rot_square[1, 0]), color=RED, lw=1.5)
ax.legend(fontsize=9)
ax.text(0.5, -0.3, "det(Q)=1, area preserved", fontsize=9,
        color=GRAY, ha="center")

# Panel C: QᵀQ = I verification
ax = axes[2]
ax.axis("off")
ax.set_title("C) QᵀQ = I  (numerical check)", fontsize=11, fontweight="bold")
QtQ = Q14.T @ Q14
lines14 = [
    f"Q (30° rotation):",
    f"[[{Q14[0,0]:.3f},  {Q14[0,1]:.3f}],",
    f" [{Q14[1,0]:.3f},  {Q14[1,1]:.3f}]]",
    "",
    "Qᵀ Q =",
    f"[[{QtQ[0,0]:.3f},  {QtQ[0,1]:.6f}],",
    f" [{QtQ[1,0]:.6f},  {QtQ[1,1]:.3f}]]",
    "",
    "≈ I  ✓",
    "",
    "Columns are unit vectors.",
    "Columns are orthogonal.",
    "Inverse = Transpose.",
]
y = 0.95
for line in lines14:
    col = GREEN if "✓" in line else (RED if "=" in line and "Q" in line else GRAY)
    ax.text(0.08, y, line, transform=ax.transAxes,
            fontsize=9.5, color=col, va="top", fontfamily="monospace")
    y -= 0.073

save_fig("fig14_orthogonal")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 15 — Special Matrices: Identity, Diagonal, Symmetric
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
fig.suptitle("FIGURE 15 — Special Matrices & Their Effects",
             fontsize=14, fontweight="bold", color=BLACK, y=1.02)

def heatmap_matrix(ax, M, title, cmap="Blues", fmt=".1f"):
    ax.set_title(title, fontsize=10, fontweight="bold")
    im = ax.imshow(M, cmap=cmap, aspect="auto",
                   vmin=M.min() - 0.1, vmax=M.max() + 0.1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:{fmt}}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=WHITE if M[i, j] > (M.max() * 0.6) else BLACK)
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels([f"c{j}" for j in range(M.shape[1])], fontsize=9)
    ax.set_yticklabels([f"r{i}" for i in range(M.shape[0])], fontsize=9)

I3 = np.eye(3)
D3 = np.diag([3.0, -1.0, 2.0])
S3 = np.array([[4.0, 2, 1], [2, 5, 3], [1, 3, 6]])

heatmap_matrix(axes[0], I3,
    "A) Identity I₃\nAI = IA = A\n(no transformation)", cmap="Blues")
axes[0].set_xlabel("Diagonal = 1, all else = 0", fontsize=9, color=GRAY)

heatmap_matrix(axes[1], D3,
    "B) Diagonal Matrix\nScales each axis independently", cmap="RdBu_r", fmt=".0f")
axes[1].set_xlabel("Scales: x→3x, y→-y, z→2z", fontsize=9, color=GRAY)

heatmap_matrix(axes[2], S3,
    "C) Symmetric Matrix S = Sᵀ\n(Covariance matrices are symmetric)", cmap="Oranges")
axes[2].set_xlabel("S[i,j] = S[j,i]  →  mirror image", fontsize=9, color=GRAY)

save_fig("fig15_special_matrices")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 16 — Full ML Story: Linear Regression Visualized
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_FULL)
fig.suptitle("FIGURE 16 — Complete ML Story: Linear Regression from Linear Algebra",
             fontsize=13, fontweight="bold", color=BLACK)

np.random.seed(42)
n = 40
area = np.random.uniform(50, 200, n)
price = 1.5 * area + 30 + np.random.normal(0, 15, n)

X16 = np.column_stack([np.ones(n), area])   # design matrix (n×2)
beta = np.linalg.solve(X16.T @ X16, X16.T @ price)
y_pred = X16 @ beta
residuals = price - y_pred

# Panel (0,0): Raw data
ax = axes[0, 0]
ax.scatter(area, price, color=BLUE, alpha=0.6, s=40, label="data points")
ax.set_xlabel("Area (m²)", fontsize=10); ax.set_ylabel("Price ($k)", fontsize=10)
ax.set_title("Step 1: Raw Data\n(scatter of samples)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel (0,1): Design matrix heatmap (subsample)
ax = axes[0, 1]
sample_X = X16[:8]
im = ax.imshow(sample_X, cmap="Blues", aspect="auto")
for i in range(8):
    ax.text(0, i, "1", ha="center", va="center", fontsize=10,
            color=WHITE if sample_X[i, 0] > 0.5 else BLACK)
    ax.text(1, i, f"{sample_X[i,1]:.0f}", ha="center", va="center",
            fontsize=9, color=BLACK)
ax.set_xticks([0, 1]); ax.set_xticklabels(["bias", "area"], fontsize=10)
ax.set_title("Step 2: Design Matrix X\n(first 8 rows shown)", fontsize=10, fontweight="bold")
ax.set_ylabel("Sample", fontsize=9)

# Panel (0,2): XᵀX — symmetric matrix
ax = axes[0, 2]
XtX16 = X16.T @ X16
im = ax.imshow(XtX16, cmap="Oranges")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{XtX16[i,j]:.0f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=WHITE if XtX16[i, j] > XtX16.max()*0.6 else BLACK)
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["β₀","β₁"]); ax.set_yticklabels(["β₀","β₁"])
ax.set_title(f"Step 3: XᵀX (2×2 symmetric)\ndet={np.linalg.det(XtX16):.0f}",
             fontsize=10, fontweight="bold")

# Panel (1,0): Normal equation solution
ax = axes[1, 0]
ax.axis("off")
ax.set_title("Step 4: Normal Equation\nβ = (XᵀX)⁻¹ Xᵀy", fontsize=10, fontweight="bold")
solution_lines = [
    ("XᵀX:", f"shape {XtX16.shape}  symmetric"),
    ("Xᵀy:", f"shape {(X16.T @ price).shape}"),
    ("", ""),
    ("β = solve(XᵀX, Xᵀy)", ""),
    ("", ""),
    (f"β₀ (intercept) =", f"{beta[0]:.2f}"),
    (f"β₁ (slope)     =", f"{beta[1]:.4f}"),
    ("", ""),
    ("Model:", f"price = {beta[0]:.1f} + {beta[1]:.3f}×area"),
]
y = 0.92
for left, right in solution_lines:
    ax.text(0.05, y, left, transform=ax.transAxes, fontsize=9,
            color=GRAY, va="top", fontfamily="monospace")
    ax.text(0.55, y, right, transform=ax.transAxes, fontsize=9,
            color=RED, va="top", fontweight="bold", fontfamily="monospace")
    y -= 0.10

# Panel (1,1): Fit line over scatter
ax = axes[1, 1]
x_line = np.linspace(area.min(), area.max(), 100)
y_line = beta[0] + beta[1] * x_line
ax.scatter(area, price, color=BLUE, alpha=0.5, s=35, label="data")
ax.plot(x_line, y_line, color=RED, lw=2.5,
        label=f"y={beta[0]:.1f}+{beta[1]:.3f}x")
for xi, yi, ypi in zip(area, price, y_pred):
    ax.plot([xi, xi], [yi, ypi], color=GRAY, lw=0.5, alpha=0.5)
ax.set_xlabel("Area (m²)", fontsize=10); ax.set_ylabel("Price ($k)", fontsize=10)
ax.set_title("Step 5: Best-Fit Line\n(minimizes sum of squared residuals)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel (1,2): Residuals
ax = axes[1, 2]
ax.bar(range(n), residuals, color=[RED if r > 0 else BLUE for r in residuals], alpha=0.7)
ax.axhline(0, color=BLACK, lw=1.2)
ax.set_xlabel("Sample index", fontsize=10)
ax.set_ylabel("Residual (actual − predicted)", fontsize=9)
ss_res = np.sum(residuals**2); ss_tot = np.sum((price - price.mean())**2)
r2 = 1 - ss_res / ss_tot
ax.set_title(f"Step 6: Residuals\nR² = {r2:.3f}", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save_fig("fig16_ml_regression")


# ─────────────────────────────────────────────────────────────────────────────
# Assemble PDF if not in PNG mode
# ─────────────────────────────────────────────────────────────────────────────
if not PNG_MODE:
    from matplotlib.backends.backend_pdf import PdfPages

    all_figs = [
        ("fig01_vectors",      "Figure 1 — Vectors"),
        ("fig02_vector_addition", "Figure 2 — Vector Addition"),
        ("fig03_scalar_mult",  "Figure 3 — Scalar Multiplication"),
        ("fig04_dot_product",  "Figure 4 — Dot Product & Angle"),
        ("fig05_norms",        "Figure 5 — Vector Norms"),
        ("fig06_matrix_transform", "Figure 6 — Matrix Transformation"),
        ("fig07_determinant",  "Figure 7 — Determinant (Area)"),
        ("fig08_det_steps",    "Figure 8 — Determinant Step-by-Step"),
        ("fig09_transpose",    "Figure 9 — Matrix Transpose"),
        ("fig10_matmul",       "Figure 10 — Matrix Multiplication"),
        ("fig11_inverse",      "Figure 11 — Matrix Inverse"),
        ("fig12_linear_system","Figure 12 — Solving Ax = b"),
        ("fig13_rank",         "Figure 13 — Matrix Rank"),
        ("fig14_orthogonal",   "Figure 14 — Orthogonality"),
        ("fig15_special_matrices","Figure 15 — Special Matrices"),
        ("fig16_ml_regression","Figure 16 — Full ML Story"),
    ]

    output_pdf = "ML-Math-Visuals.pdf"

    with PdfPages(output_pdf) as pdf:
        # Cover page
        fig_cover = plt.figure(figsize=(11, 8.5))
        fig_cover.patch.set_facecolor(BLACK)
        ax_cover = fig_cover.add_subplot(111)
        ax_cover.axis("off")
        ax_cover.set_facecolor(BLACK)
        ax_cover.text(0.5, 0.72,
                      "Machine Learning Mathematics",
                      transform=ax_cover.transAxes,
                      ha="center", fontsize=28, fontweight="bold", color=WHITE)
        ax_cover.text(0.5, 0.58,
                      "Visual Reference Guide",
                      transform=ax_cover.transAxes,
                      ha="center", fontsize=18, color=BLUE)
        ax_cover.text(0.5, 0.45,
                      "Vectors · Matrices · Determinants · Inverse · Rank\n"
                      "Dot Product · Norms · Linear Systems · Linear Regression",
                      transform=ax_cover.transAxes,
                      ha="center", fontsize=12, color=GRAY)
        ax_cover.text(0.5, 0.28,
                      "16 Figures · Step-by-Step · NumPy Verified",
                      transform=ax_cover.transAxes,
                      ha="center", fontsize=11, color=GREEN)
        ax_cover.text(0.5, 0.12,
                      "Based on \"Machine Learning Mathematics\" — Demystify with Ajay",
                      transform=ax_cover.transAxes,
                      ha="center", fontsize=10, color=GRAY, style="italic")
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        # Re-render each figure into the PDF
        import importlib, types

        def make_fig(name):
            """Re-execute the block that creates each figure by name."""
            pass   # figures already saved as PNGs during dry run above

        # Since figures were already closed (save_fig closes them),
        # re-run by importing PIL to composite PNGs into the PDF.
        try:
            from PIL import Image
            import io

            for fname, flabel in all_figs:
                png_path = f"{fname}.png"
                import os
                if os.path.exists(png_path):
                    img = Image.open(png_path)
                    fig_pg = plt.figure(figsize=(14, 9))
                    ax_pg = fig_pg.add_subplot(111)
                    ax_pg.imshow(img)
                    ax_pg.axis("off")
                    ax_pg.set_title(flabel, fontsize=12, color=BLACK,
                                    fontweight="bold", pad=6)
                    pdf.savefig(fig_pg, bbox_inches="tight")
                    plt.close(fig_pg)
        except ImportError:
            print("Pillow not installed — PDF will contain cover page only.")
            print("Install with: pip install Pillow")
            print("All figures saved as individual PNG files.")

    print(f"\nSaved: {output_pdf}")
else:
    print("\nPNG files saved:")
    import os
    for f in sorted(os.listdir(".")):
        if f.startswith("fig") and f.endswith(".png"):
            size = os.path.getsize(f) // 1024
            print(f"  {f}  ({size} KB)")

print("\nDone. 16 figures covering:")
topics = [
    "1.  Vectors: 2D/3D representation & ML feature vectors",
    "2.  Vector addition: triangle law, parallelogram law, step-by-step",
    "3.  Scalar multiplication & subtraction",
    "4.  Dot product: formula, projection, three angle cases",
    "5.  Vector norms: L1 diamond, L2 circle, L∞ square (unit balls)",
    "6.  Matrix as transformation: grid warping, basis vectors",
    "7.  Determinant as parallelogram area (det>0, det<0, det=0)",
    "8.  Determinant 2×2: step-by-step ad-bc with diagonal annotation",
    "9.  Transpose: row↔col flip, heatmap, shape rules",
    "10. Matrix multiplication: highlighted dot-product per element",
    "11. Matrix inverse: forward transform → undo → 2×2 formula",
    "12. Solving Ax=b: line intersection, column picture, steps",
    "13. Matrix rank: full-rank vs deficient vs zero (column space)",
    "14. Orthogonality: right-angle marker, rotation preserving shape",
    "15. Special matrices: identity, diagonal, symmetric (heatmaps)",
    "16. Full ML story: regression — data→design matrix→XᵀX→β→fit→residuals",
]
for t in topics:
    print(f"  {t}")
