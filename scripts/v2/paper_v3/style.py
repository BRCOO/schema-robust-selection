"""Shared figure style for the PVLDB v3 paper (Nature teal-green family).

Palette checked with the dataviz validator (2026-11-04): the three categorical slots teal / deep blue / sky blue pass the
CVD and normal-vision separation checks; grey is the neutral slot for "other baselines" only. Sky blue has low contrast
against white, so it is always paired with a marker or a direct label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/v2"
OUT_FIG = ROOT / "paper/v3_assets/figures"
OUT_TAB = ROOT / "paper/v3_assets/tables"
OUT_SRC = ROOT / "paper/v3_assets/source_data"
for _d in (OUT_FIG, OUT_TAB, OUT_SRC):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------- palette
TEAL = "#008C76"        # SRS+T (proposed)
TEAL_DARK = "#00594B"
TEAL_MID = "#3FAE9A"    # SRS (source-only), always with a dashed line or hollow marker
TEAL_LIGHT = "#A9DCD1"
TEAL_PALE = "#E6F4F0"
BLUE = "#1F5A9E"        # NMIFS (strongest baseline)
SKY = "#56B4E9"         # mRMR
GREY = "#7A7A7A"        # L1 + schema dropout / other named baselines
GREY_LIGHT = "#BDBDBD"
GREY_PALE = "#E6E6E6"
WARM = "#C8553D"        # degradation / worse only
SAND = "#B9A27E"        # filter-then-mRMR (knows targets), neutral warm-grey
INK = "#1A1A1A"
INK2 = "#555555"

METHOD_COLOR = {
    "srs_ap_t": TEAL, "srs_ap": TEAL_MID, "nmifs": BLUE, "mrmr": SKY, "l1_dropout": GREY, "mrmr_filter_known": SAND,
    "srs_c_val": "#7CC7B6", "srs_ap_unaware": "#8FD1C3", "srs_ap_t_nosur": "#9BB9B2",
}
LABEL = {
    "srs_ap_t": "SRS+T", "srs_ap": "SRS", "nmifs": "NMIFS", "mrmr": "mRMR", "mrmr_filter_known": "Filter+mRMR",
    "mmrfs": "MMRFS", "jmi": "JMI", "cmim": "CMIM", "l1_logistic": "L1", "l1_dropout": "L1+dropout",
    "mmr_ig": "MMR", "mms_xin_ig": "MMS", "dpp_ig": "DPP", "mrmr_stratified": "mRMR-strat.", "best_single_srs": "Best single gen.",
    "srs_c_val": "SRS+T (likelihood)", "srs_c_unknown_val": "SRS (likelihood)", "srs_ap_unaware": "w/o scenarios",
    "srs_ap_t_nosur": "w/o surrogates in obj.", "fs_cv": "schema-unaware (lik.)",
}
#: the twelve established selectors of the paper, in table order
BASELINES = ("nmifs", "mrmr", "mrmr_filter_known", "mmrfs", "jmi", "cmim", "l1_logistic", "l1_dropout", "mmr_ig", "mms_xin_ig",
             "dpp_ig", "mrmr_stratified")

# sequential teal ramp and a diverging warm-teal map centred on 0
SEQ = LinearSegmentedColormap.from_list("teal_seq", ["#F4FBF9", "#BFE6DC", "#5DBAA6", "#008C76", "#00463B"])
DIV = LinearSegmentedColormap.from_list("warm_teal", ["#A63A26", "#E39B85", "#F5F3F0", "#7FC6B6", "#00594B"])

# ----------------------------------------------------------------------------------------------- geometry (inches)
FULL_W = 7.0     # acmart two-column text width
COL_W = 3.33     # one column


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.5, "legend.frameon": False, "axes.edgecolor": "#333333",
        "axes.labelcolor": INK, "xtick.color": "#333333", "ytick.color": "#333333", "text.color": INK,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "axes.titleweight": "bold", "axes.titlepad": 3, "grid.color": "#E3E3E3", "grid.linewidth": 0.5,
        # math text in Arial too (symbols such as omega, eta, rho and x100 otherwise come from DejaVu), grids behind the data
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "axes.axisbelow": True,
    })


def panel_label(ax, letter: str, dx: float = -0.02, dy: float = 1.02) -> None:
    """Small bold lowercase panel letter just outside the upper-left corner of the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right", color=INK)


def title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=6.8, fontweight="bold", color=INK, pad=3)


def save(fig, name: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "svg", "png"):
        p = OUT_FIG / f"{name}.{ext}"
        # axis labels are passed explicitly: matplotlib 3.10 counts only their height (x) / width (y) in the tight box,
        # which clips a long x label of the right-most panel
        extra = fig.get_default_bbox_extra_artists() + [a.xaxis.label for a in fig.axes] + [a.yaxis.label for a in fig.axes]
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, dpi=300 if ext == "png" else None, bbox_extra_artists=extra)
        paths.append(p)
    plt.close(fig)
    return paths


def save_exact(fig, name: str) -> list[Path]:
    """Save at exactly the figure size (no tight box), so that a figure made FULL_W or COL_W wide prints at 100% and its
    7 pt text stays 7 pt. Layouts must keep every artist inside the canvas; ``check_inside`` reports any that leave it."""
    check_inside(fig, name)
    paths = []
    for ext in ("pdf", "svg", "png"):
        p = OUT_FIG / f"{name}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def check_inside(fig, name: str, tol: float = 0.5) -> None:
    """Print the text artists whose extent leaves the canvas (tolerance in pixels at the figure dpi)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer)
        except Exception:  # noqa: BLE001
            continue
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol:
            print(f"  [{name}] outside the canvas: {t.get_text()[:40]!r} x {bb.x0:.0f}..{bb.x1:.0f} of {W:.0f}, y {bb.y0:.0f}..{bb.y1:.0f} of {H:.0f}")
