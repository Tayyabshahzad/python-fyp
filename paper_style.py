"""
paper_style.py
==============

Shared plotting style for reproducing the figures of

    M. Zubair, M. Tahir, P. Vasilopoulos and K. Sabeeh,
    Phys. Rev. B 96, 045405 (2017).

Nothing here touches the physics.  Every number in this file is a
typographic or geometric property measured off the published figures
(300 dpi renders of the PDF pages), so that our output can be compared
against the paper frame-for-frame.

MEASURED CONVENTIONS OF THE PUBLISHED FIGURES
---------------------------------------------
* Type is Times.  STIX is used here: it is metric-compatible with Times
  and covers the mathematics as well, so labels such as E^{up,+}_{++}
  match the paper's shapes.
* No figure carries a title inside its frame.  All captions live in the
  body text of the paper, so no suptitle / set_title is ever used.
* Ticks point INWARD and appear on all four sides, with unlabelled minor
  ticks between the labelled majors.
* Frames are a plain rectangle on all four sides, ~1.2 pt.
* Legends are hand-placed inside the axes as a short colour-coded line
  sample followed by BLACK text - never matplotlib's boxed legend.
"""

import matplotlib.pyplot as plt

# Line weights measured off the published curves.
LW_MAIN = 1.25       # principal curves (Figs. 1, 2)
LW_FAN = 0.80        # dense Landau-level fans (Figs. 3, 5, 6, 7)
LW_TRACE = 0.85      # single traces (Figs. 8, 9, 12, 13, 14)
LW_EF = 1.25         # the magenta Fermi-energy curve

# The paper's dotted spin-down style.  matplotlib's ":" is too sparse at
# these line widths; this dash pattern matches the printed dot spacing.
DOTS = (1, 1.4)


def apply():
    """Install the paper's typography.  Call once at import time."""
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["savefig.dpi"] = 300


def frame(ax, labelsize=10.5, minor=True, labelbottom=True):
    """Give one axes the paper's frame and tick treatment."""
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   bottom=True, left=True, labelsize=labelsize,
                   length=5, width=1.0, labelbottom=labelbottom)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       bottom=True, left=True, length=2.6)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)


def dashed(style):
    """Keyword arguments giving the paper's dotted line style."""
    return {"dashes": DOTS} if style == ":" else {}


def legend_entry(ax, x, y, style, color, text, fontsize=8.0, sample=0.105,
                 gap=0.030, backing=False, width=0.215, height=0.072):
    """One hand-placed legend row, as the paper draws them.

    A short line sample carries the colour and the line style; the label
    itself is always black.  x, y are axes fractions.

    backing=True paints an opaque white rectangle behind the row first.
    The paper does this in the dense Landau-level figures (verified by
    sampling the PDF: pure white pixels appear where fan lines would
    otherwise cross behind the legend text).  width and height are the
    rectangle's size in axes fractions - keep them just large enough to
    cover the row, or neighbouring rows merge into one white block and
    hide the curves the figure is meant to show.
    """
    if backing:
        ax.add_patch(plt.Rectangle(
            (x - 0.012, y - height / 2), width, height,
            transform=ax.transAxes, facecolor="white",
            edgecolor="none", zorder=5))
    ax.plot([x, x + sample], [y, y], transform=ax.transAxes, color=color,
            linestyle=style, lw=1.3, clip_on=False, zorder=6,
            **dashed(style))
    ax.text(x + sample + gap, y, text, transform=ax.transAxes, color="black",
            fontsize=fontsize, va="center", ha="left", zorder=6)


def E_label(mu, spin, tau, n=False):
    """The paper's curve labels.

    Figures 1-2 use   E^{s,tau}_{mu}
    Figures 3-7 use   E^{s,tau}_{n,mu}   (n=True)
    """
    arrow = r"\uparrow" if spin > 0 else r"\downarrow"
    valley = "+" if tau > 0 else "-"
    sub = f"n,{mu}" if n else mu
    return fr"$\mathrm{{E}}^{{{arrow},{valley}}}_{{{sub}}}$"


def save(fig, name):
    """Save at the paper's resolution and report."""
    fig.savefig(name, dpi=300)
    print(f"Saved: {name}")
