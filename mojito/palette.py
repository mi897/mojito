"""The one plot palette: every chart and animation in scripts/ and mojito/sim/gait_plots.py takes its colours from here.

Series colours in order of use: BLUE, ORANGE, AQUA, PURPLE (also the leg colours FL, FR, RL, RR). Text in neutral ink,
one y-axis per chart (CLAUDE.md). Plain constants: importing this needs neither NumPy nor Matplotlib.
"""
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1fa6a0"
PURPLE = "#8a5cc2"
INK = "#333333"   # text, axis labels and reference lines in the physics plots
GREY = "#9a9a9a"  # secondary reference marks

# The panel style of the learning and body plots (scripts/learning/evaluate.py, scripts/body/make_body_demos.py, make_demo.py).
TITLE_INK = "#0b0b0b"
LABEL_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e6e5e0"


def panel_style(font_size: float, title_size: float, line_width: float) -> dict:
    """Matplotlib rcParams for the panel style: light surface, no top or right spine, a light grid, titles on the left."""
    return {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": font_size,
        "axes.edgecolor": GRID, "axes.labelcolor": LABEL_INK, "text.color": TITLE_INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": LABEL_INK, "ytick.labelcolor": LABEL_INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlesize": title_size, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "lines.linewidth": line_width,
    }
