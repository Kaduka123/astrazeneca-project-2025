# cho_analysis/core/visualization_utils.py
"""Shared Visualization Constants for the CHO Analysis Pipeline.

Defines common aesthetic defaults (styles, DPI, colormaps) used across
different visualization modules to ensure consistency. These provide
code-level defaults that can be overridden by runtime configurations
fetched via `config.py`.
"""

from typing import Any, Dict, Final, List

# ===================================================
#  === Shared Visualization Constants ===
# ===================================================

# Font Configuration
# ----------------------------------------------------
DEFAULT_FONT_FAMILY: Final[str] = "DejaVu Sans"
FONT_SIZE_FIGURE_TITLE: Final[int] = 18
FONT_SIZE_AXES_TITLE: Final[int] = 16
FONT_SIZE_LABEL: Final[int] = 12
FONT_SIZE_TICK: Final[int] = 10
FONT_SIZE_LEGEND: Final[int] = 10
FONT_SIZE_ANNOTATION: Final[int] = 9
FONT_WEIGHT_BOLD: Final[str] = "bold"
FONT_WEIGHT_NORMAL: Final[str] = "normal"

# Matplotlib rcParams
# ----------------------------------------------------
BASE_RC_PARAMS: Final[Dict[str, Any]] = {
    "font.family": DEFAULT_FONT_FAMILY,
    "font.sans-serif": [DEFAULT_FONT_FAMILY, "sans-serif"],
    "font.size": FONT_SIZE_TICK,
    "axes.titlesize": FONT_SIZE_AXES_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.titlesize": FONT_SIZE_FIGURE_TITLE,
    "axes.titleweight": FONT_WEIGHT_BOLD,
    "axes.labelweight": FONT_WEIGHT_NORMAL,
    "figure.titleweight": FONT_WEIGHT_BOLD,
    "axes.titlepad": 15.0,
    "axes.labelpad": 10.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
}

# General Style and Resolution Defaults
# ----------------------------------------------------
DEFAULT_STYLE: Final[str] = "seaborn-v0_8-whitegrid"
DEFAULT_DPI: Final[int] = 300
SAVE_DPI: Final[int] = 300

# Shared Colormaps & Palettes
# ----------------------------------------------------
CMAP_SEQUENTIAL: Final[str] = "viridis"  # Default for heatmaps, continuous data where order matters
CMAP_DIVERGING: Final[str] = "RdBu_r"  # Default for correlations, centered data
PALETTE_QUALITATIVE: Final[str] = "tab10"  # Default for distinct categories (e.g., platforms)

# ===================================================
#  === Task-Specific Visualization Constants ===
# ===================================================

# Task 1 Specific Colors
# ----------------------------------------------------
NODE_COLOR_POS: Final[str] = "#377eb8"  # Blueish
NODE_COLOR_NEG: Final[str] = "#e41a1c"  # Reddish
EDGE_COLOR_POS: Final[str] = "#4daf4a"  # Greenish
EDGE_COLOR_NEG: Final[str] = "#984ea3"  # Purplish

# Task 2 Specific Colors
# ----------------------------------------------------
PALETTE_DISTRIBUTION: Final[List[str]] = ["#377eb8", "#e41a1c", "#4daf4a"]  # Blue, Red, Green
PALETTE_BAR: Final[str] = "Blues_d"
