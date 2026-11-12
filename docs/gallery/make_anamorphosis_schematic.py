"""Schematic: image slicer with anamorphosis, from the sky to the detector pixels.

Run: python make_anamorphosis_schematic.py
Writes anamorphosis_schematic.png next to this script (16:9, one page).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent

# ------------------------------------------------------------------ #
#  style                                                             #
# ------------------------------------------------------------------ #
INK = "#1d2433"
MUTED = "#6b7280"
GRID = "#c9ced8"
SPAXEL = "#e8853a"          # the highlighted spaxel
SPAXEL_LIGHT = "#f8dcc4"
SLICE_FILL = "#eef1f6"
WRONG = "#c0392b"
RIGHT = "#1e7a46"
LAMBDA = ["#3b6fb6", "#5aa0a8", "#8bbf5a"]   # wavelength bins i, i+1, i+2

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK})

fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.set_aspect("equal")
ax.axis("off")


def text(x, y, s, size=12, color=INK, ha="left", va="center", weight="normal", **kwargs):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, **kwargs)


def rect(x, y, w, h, fc="none", ec=INK, lw=1.0, **kwargs):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, **kwargs))


def arrow(x0, y0, x1, y1, color=INK, lw=1.6, style="-|>", ms=14, **kwargs):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, linewidth=lw, **kwargs))


def panel_title(x, y, number, title):
    text(x, y, f"{number}", size=15, weight="bold", color="white", ha="center",
         bbox=dict(boxstyle="circle,pad=0.25", fc=INK, ec="none"))
    text(x + 2.6, y, title, size=15, weight="bold")


# ------------------------------------------------------------------ #
#  title                                                             #
# ------------------------------------------------------------------ #
text(3, 86, "Image slicer with anamorphosis: from the sky to the detector pixels",
     size=21, weight="bold")
text(3, 82.2, "Lazuli narrow field: 58 slices × 116 spaxels of 40 mas  (2.3″ × 4.6″)",
     size=13, color=MUTED)

# ------------------------------------------------------------------ #
#  1. the sky                                                        #
# ------------------------------------------------------------------ #
panel_title(4.5, 76, "1", "The sky: slices and spaxels")

x0, y0, s = 4, 44, 3.0            # spaxel size (drawing units)
nx, ny = 12, 8                    # spaxels along the slice, slices shown
for j in range(ny):
    rect(x0, y0 + j * s, nx * s, s, fc=SLICE_FILL, ec=INK, lw=1.2)
    for i in range(1, nx):
        ax.plot([x0 + i * s] * 2, [y0 + j * s, y0 + (j + 1) * s], color=GRID, lw=0.8)
hl_i, hl_j = 7, 3                 # highlighted spaxel
rect(x0 + hl_i * s, y0 + hl_j * s, s, s, fc=SPAXEL, ec=INK, lw=1.4)
# highlighted slice outline
rect(x0, y0 + hl_j * s, nx * s, s, ec=SPAXEL, lw=2.4)

# dimensions
arrow(x0, y0 - 2.2, x0 + nx * s, y0 - 2.2, style="<|-|>", color=MUTED, lw=1.2, ms=10)
text(x0 + nx * s / 2, y0 - 4.3, "along the slice: 116 spaxels (4.6″)", size=11.5, color=MUTED, ha="center")
arrow(x0 - 1.8, y0, x0 - 1.8, y0 + ny * s, style="<|-|>", color=MUTED, lw=1.2, ms=10)
text(x0 - 3.0, y0 + ny * s / 2, "58 slices (2.3″)", size=11.5, color=MUTED, ha="center", rotation=90)

text(x0 + nx * s + 1.5, y0 + (hl_j + 0.5) * s, "one slice", size=11.5, color=SPAXEL, weight="bold")
text(x0 + nx * s + 1.5, y0 + (hl_j + 0.5) * s - 3.2, "40 mas wide", size=11, color=MUTED)
ax.annotate("one spaxel\n40 × 40 mas", xy=(x0 + (hl_i + 1) * s, y0 + (hl_j + 1) * s),
            xytext=(x0 + nx * s + 1.5, y0 + (hl_j + 3.6) * s), fontsize=11.5, color=SPAXEL,
            fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=SPAXEL, lw=1.2, shrinkA=2))

# ------------------------------------------------------------------ #
#  2. anamorphosis                                                   #
# ------------------------------------------------------------------ #
panel_title(56.5, 76, "2", "Anamorphic magnification")
text(56, 71.2, "×2 across the slice, ×1 along it", size=12, color=MUTED)

# square spaxel -> 1 x 2 rectangle
sq = 5.0
rect(58, 56, sq, sq, fc=SPAXEL, ec=INK, lw=1.4)
text(58 + sq / 2, 54.2, "spaxel", size=11, color=MUTED, ha="center", va="top")
arrow(65, 58.5, 71.5, 58.5, lw=2)
rect(73.5, 53.5, sq, 2 * sq, fc=SPAXEL, ec=INK, lw=1.4)
text(73.5 + sq + 1.2, 58.5, "1 × 2", size=13, weight="bold", color=SPAXEL)

# the slice becomes a pseudo-slit
text(56, 47.3, "each slice → one pseudo-slit", size=12)
sx0, sy0, ss = 57, 38.5, 2.2
for i in range(10):
    rect(sx0 + i * ss, sy0, ss, 2 * ss, fc=SPAXEL if i == 5 else SPAXEL_LIGHT, ec=INK, lw=0.9)
text(sx0 + 5 * ss, sy0 - 1.4, "slit length = slice length (116 spaxels)", size=10.5, color=MUTED, ha="center", va="top")
ax.annotate("", xy=(sx0 + 10 * ss + 1.2, sy0), xytext=(sx0 + 10 * ss + 1.2, sy0 + 2 * ss),
            arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.1, mutation_scale=9))
text(sx0 + 10 * ss + 2.2, sy0 + ss, "slit width\n→ 2 pixels", size=10.5, color=MUTED)

arrow(48, 56, 55, 56, lw=2, color=MUTED)       # 1 -> 2

# ------------------------------------------------------------------ #
#  3. the detector                                                   #
# ------------------------------------------------------------------ #
panel_title(98.5, 76, "3", "The detector: one slice trace")

# pixel grid: columns = along the slice, rows = dispersion
px = 3.4
gx0, gy0 = 99, 37
ncol, nrow = 6, 9
hl_col = 3
for c in range(ncol):
    for r in range(nrow):
        rect(gx0 + c * px, gy0 + r * px, px, px, fc="white", ec=GRID, lw=0.9)
rect(gx0 + hl_col * px, gy0, px, nrow * px, ec=SPAXEL, lw=2.2)
text(gx0 + ncol * px / 2, gy0 - 1.6, "along the slice: 1 spaxel = 1 pixel", size=11, color=MUTED, ha="center", va="top")
arrow(gx0 - 1.6, gy0 + nrow * px, gx0 - 1.6, gy0, color=MUTED, lw=1.2, ms=10)
text(gx0 - 2.8, gy0 + nrow * px / 2, "dispersion (λ)", size=11, color=MUTED, ha="center", rotation=90)
text(gx0 + (hl_col + 0.5) * px, gy0 + nrow * px + 1.4, "trace of the\nhighlighted spaxel",
     size=10.5, color=SPAXEL, ha="center", va="bottom", weight="bold")

# zoom: the pixels of the highlighted column and the wavelength bins
zx0, zy0, zp = 128, 37, 4.2      # zoom pixel size
nz = 6
for k in range(nz):
    rect(zx0, zy0 + (nz - 1 - k) * zp, zp, zp, fc="white", ec=INK, lw=1.1)
    text(zx0 - 0.8, zy0 + (nz - 1 - k + 0.5) * zp, f"p{'+' + str(k) if k else ''}", size=11, ha="right", color=MUTED)
ax.plot([gx0 + (hl_col + 1) * px, zx0], [gy0 + nrow * px, zy0 + nz * zp], color=SPAXEL, lw=0.8, ls=":")
ax.plot([gx0 + (hl_col + 1) * px, zx0], [gy0, zy0], color=SPAXEL, lw=0.8, ls=":")

# monochromatic images: 2 pixels, shifted by 1 pixel per wavelength bin
bw = 2.5
for b, color in enumerate(LAMBDA):
    xb = zx0 + zp + 1.2 + b * (bw + 0.8)
    top = zy0 + (nz - b) * zp
    rect(xb, top - 2 * zp, bw, 2 * zp, fc=color, ec=INK, lw=0.9, alpha=0.9)
    text(xb + bw / 2, top + 0.6, ["λ$_i$", "λ$_{i+1}$", "λ$_{i+2}$"][b], size=12, ha="center", va="bottom", color=color, weight="bold")
text(zx0 + zp + 1.2 + 3 * (bw + 0.8) + 0.4, zy0 + (nz - 1.5) * zp,
     "a monochromatic\nimage covers\n2 pixels", size=11, va="center")
text(zx0 + zp + 1.2 + 3 * (bw + 0.8) + 0.4, zy0 + (nz - 3.6) * zp,
     "the next bin is\nshifted by 1 pixel:\nbins share pixels", size=11, va="center")

arrow(83, 56, 96, 56, lw=2, color=MUTED)       # 2 -> 3

# ------------------------------------------------------------------ #
#  4. bottom: slicersim and the pixel count                          #
# ------------------------------------------------------------------ #
def box(x, y, w, h, ec=GRID, fc="white", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                facecolor=fc, edgecolor=ec, linewidth=lw))

panel_title(4.5, 27.5, "4", "In slicersim")
box(3, 3, 48, 20)
text(5, 19.5, "voxel = 1 spaxel × 1 wavelength bin", size=13, weight="bold")
text(5, 15.2, "• wavelength bins come from the dispersion law:", size=12)
text(7.5, 12.4, "1 bin = 1 detector pixel along λ", size=12, weight="bold")
text(5, 8.6, "• the 2-pixel slit image = the line spread function:", size=12)
text(7.5, 5.8, "2.3 bins (= pixels) per resolution element", size=12, weight="bold")

# pixel count
panel_title(56.5, 27.5, "5", "Counting pixels over K wavelength bins")
box(55, 3, 47, 20)
cx0, cy0, cp = 58, 12.5, 3.0
for k in range(4):                                 # pixels p .. p+3
    rect(cx0 + k * cp, cy0, cp, cp, fc="white", ec=INK, lw=1)
for b, color in enumerate(LAMBDA):                 # bins i .. i+2 over 2 pixels each
    rect(cx0 + b * cp + 0.15, cy0 + cp + 0.5 + b * 1.25, 2 * cp - 0.3, 1.0, fc=color, ec="none")
text(cx0 + 4 * cp + 1.5, cy0 + cp / 2, "K = 3 bins → 4 pixels", size=12, weight="bold")
text(57, 8.8, "on the detector:", size=12)
text(76, 8.8, "K + 1 pixels", size=13, weight="bold", color=RIGHT)
text(57, 5.2, "2 pixels per voxel:", size=12)
text(76, 5.2, "2K pixels", size=13, weight="bold", color=WRONG)
text(88, 5.2, "shared pixels\ncounted twice", size=10.5, color=WRONG)

# variance per bin
panel_title(108.5, 27.5, "6", "Variance of one wavelength bin")
box(107, 3, 50, 20)
text(109, 18.5, "read noise is paid once per pixel:", size=12)
text(109, 13.6, r"1 pixel per voxel:   $\sigma^2_\mathrm{RON} + \alpha f$", size=14, color=RIGHT)
text(109, 9.2, r"2 pixels per voxel:  $2\,\sigma^2_\mathrm{RON} + \alpha f$", size=14, color=WRONG)
text(109, 4.8, "slicersim ≥ 1.5: npix_per_voxel = 1  (was 2)", size=11, color=RIGHT)

for ext in ["png"]:
    fig.savefig(HERE / f"anamorphosis_schematic.{ext}", dpi=200)
print("written", sorted(str(p.name) for p in HERE.glob("anamorphosis_schematic.*")))
