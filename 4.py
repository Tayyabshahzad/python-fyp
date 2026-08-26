# =============================================================================
# FIGURE 4  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 4 caption, verbatim:
#   "LL spectrum of bilayer MoS2 at B = 30 T and V = 0 labeled by (n,mu,s)
#    with s the spin index s = +-1(up,down) and mu the layer index [see text
#    after Eq. (4)] mu = (mu1 mu2). The upper panels are for the conduction
#    band and the lower ones are for the valence band. Further, the left
#    panels are for Mz = Mv = 0 and the right ones for Mz != Mv != 0. For
#    simplicity we do not show the valence band levels for the second layer."
#
# EQUATIONS USED (all in paper_equations.py / landau_levels.py):
#   Eq. (5)  p.3  - quartic for the n >= 1 Landau levels
#   Eq. (4)  p.3  - E = hbar*omega_c*epsilon
#   Eq. (8)  p.4  - the single n = -1 level
#   Eq. (10) p.4  - cubic for the three n = 0 levels
#
# This is a LEVEL DIAGRAM, not a plot: a plain box, no axis numbers, an
# "E" arrow at the left, and two columns of short horizontal segments -
# one per valley - each tagged (n, mu, s).
#
# LABEL PLACEMENT
#   The published labels are hand-placed, so they are transcribed here as
#   an explicit table rather than derived.  In the Mz = Mv = 0 panels the
#   two valleys are exactly degenerate, so a label describes BOTH columns
#   and only ONE set of labels is drawn; they alternate between the
#   far-left position and the middle gap so that neighbouring rows do not
#   collide.  Energies are still computed from the equations - only the
#   text position is transcribed.
#
# STATUS: all four panels transcribed and matched against the published
#   figure.  Each panel has its own label table above its draw function.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import landau_levels as ll

ps.apply()

B_FIELD = 30.0     # caption: "at B = 30 T"
V = 0.0            # caption: "and V = 0"

# Geometry measured off the published upper-left panel.
COL1 = (0.314, 0.531)      # tau = +1 segments
COL2 = (0.758, 0.976)      # tau = -1 segments
X_LEFT = 0.304             # far-left labels, right-aligned here
X_MID = 0.541              # middle-gap labels, left-aligned here
# The levels occupy this fractional band of the panel height.
LEVEL_BAND = (0.052, 0.836)

UP, DN = +1, -1


def arrow(s):
    return r"$\uparrow$" if s > 0 else r"$\downarrow$"


# -----------------------------------------------------------------------------
# UPPER-LEFT PANEL: conduction band, Mz = Mv = 0
#
# Transcribed from the published panel, top to bottom.  Each entry is
#     (side, [(n, mu, s), ...])
# The energy is taken from the FIRST (n, mu, s); any further entries are
# levels the paper judges degenerate with it and lists in the same label.
# -----------------------------------------------------------------------------
UL_LABELS = [
    ("left", [(6, "+-", DN)]),
    ("mid",  [(5, "++", UP)]),
    ("left", [(6, "+-", UP)]),
    ("mid",  [(5, "++", DN)]),
    ("left", [(5, "+-", DN)]),
    ("mid",  [(4, "++", UP)]),
    ("left", [(5, "+-", UP)]),
    ("mid",  [(4, "++", DN)]),
    ("mid",  [(4, "+-", DN), (3, "++", UP)]),
    ("left", [(4, "+-", UP), (3, "++", DN)]),
    ("mid",  [(3, "+-", DN), (2, "++", UP)]),
    ("left", [(3, "+-", UP), (2, "++", DN)]),
    ("mid",  [(2, "+-", DN), (1, "++", UP)]),
    ("left", [(2, "+-", UP), (1, "++", DN)]),
    ("mid",  [(1, "+-", DN), (0, "++", UP)]),
    ("left", [(1, "+-", UP), (0, "++", DN)]),
    ("left", [(-1, "++", UP), (-1, "++", DN)]),
]

MU_COLOR = {"+-": "red", "++": "black", "-+": "red", "--": "black"}

# -----------------------------------------------------------------------------
# UPPER-RIGHT PANEL: conduction band, Mz, Mv != 0
#
# Here the valleys are no longer degenerate, so each column is labelled
# separately.  The tau = -1 column carries labels on BOTH sides.
# Entries are (side, [(printed_text, colour, lookup_key), ...]).
#
# NOTE on the n = 0 labels.  The panel prints "(0,++,s)" in the tau = +1
# column and "(0,+-,s)" in the tau = -1 column.  By the paper's own
# reservation rule (p.4) it is the other way round: mu = (+,+) is reserved
# for n = -1 at K, so the n = 0 conduction root at K carries mu = (+,-),
# and at K' it carries mu = (+,+).  The printed text is reproduced exactly
# as published; the lookup key points at the level that text actually
# refers to.  Recorded rather than silently corrected.
# -----------------------------------------------------------------------------
RED, BLACK, BLUE, GREEN = "red", "black", "blue", "green"


def _t(n, mu, s):
    return f"({n},{mu},{arrow(s)})"


UR_LEFT_COL = [                      # tau = +1, red (+-) and black (++)
    ("left", [(_t(4, "+-", UP), RED, (4, "+-", UP)),
              (_t(4, "+-", DN), RED, (4, "+-", DN)),
              (_t(3, "++", UP), BLACK, (3, "++", UP))]),
    ("left", [(_t(3, "++", DN), BLACK, (3, "++", DN))]),
    ("left", [(_t(3, "+-", UP), RED, (3, "+-", UP)),
              (_t(3, "+-", DN), RED, (3, "+-", DN))]),
    ("left", [(_t(2, "++", DN), BLACK, (2, "++", DN))]),
    ("mid1", [(_t(2, "++", UP), BLACK, (2, "++", UP))]),
    ("left", [(_t(2, "+-", UP), RED, (2, "+-", UP)),
              (_t(2, "+-", DN), RED, (2, "+-", DN))]),
    ("left", [(_t(1, "++", DN), BLACK, (1, "++", DN))]),
    ("mid1", [(_t(1, "++", UP), BLACK, (1, "++", UP))]),
    ("left", [(_t(1, "+-", UP), RED, (1, "+-", UP)),
              (_t(1, "+-", DN), RED, (1, "+-", DN))]),
    ("left", [(_t(0, "++", UP), BLACK, (0, "+-", UP))]),
    ("left", [(_t(0, "++", DN), BLACK, (0, "+-", DN))]),
    ("left", [(_t(-1, "++", UP), BLACK, (-1, "++", UP))]),
    ("left", [(_t(-1, "++", DN), BLACK, (-1, "++", DN))]),
]

UR_RIGHT_COL = [                     # tau = -1, blue (+-) and green (++)
    ("mid2", [(_t(4, "+-", DN), BLUE, (4, "+-", DN))]),
    ("right", [(_t(4, "+-", UP), BLUE, (4, "+-", UP))]),
    ("right", [(_t(3, "++", UP), GREEN, (3, "++", UP)),
               (_t(3, "++", DN), GREEN, (3, "++", DN))]),
    ("mid2", [(_t(3, "+-", UP), BLUE, (3, "+-", UP))]),
    ("right", [(_t(3, "+-", DN), BLUE, (3, "+-", DN))]),
    ("right", [(_t(2, "++", UP), GREEN, (2, "++", UP)),
               (_t(2, "++", DN), GREEN, (2, "++", DN))]),
    ("mid2", [(_t(2, "+-", UP), BLUE, (2, "+-", UP))]),
    ("right", [(_t(2, "+-", DN), BLUE, (2, "+-", DN))]),
    ("right", [(_t(1, "++", UP), GREEN, (1, "++", UP)),
               (_t(1, "++", DN), GREEN, (1, "++", DN))]),
    ("mid2", [(_t(1, "+-", UP), BLUE, (1, "+-", UP))]),
    ("right", [(_t(1, "+-", DN), BLUE, (1, "+-", DN))]),
    ("right", [(_t(0, "+-", DN), BLUE, (0, "++", DN))]),
    ("right", [(_t(0, "+-", UP), BLUE, (0, "++", UP))]),
    ("right", [(_t(-1, "+-", DN), BLUE, (-1, "+-", DN))]),
    ("right", [(_t(-1, "+-", UP), BLUE, (-1, "+-", UP))]),
]

# x anchors for the four label positions of the upper-right panel
UR_X = {"left": 0.228, "mid1": 0.442, "mid2": 0.550, "right": 0.812}
UR_COL1 = (0.235, 0.430)
UR_COL2 = (0.560, 0.800)

# Per-label nudges, keyed by the row's first lookup key.
#   dx : axes fractions, negative = further left
#   dy : fractions of the panel height, negative = downwards
# Two rows in the published panel need them: the three-part red label at
# the top is long enough to need extra room (the paper lets it run over
# the segments), and (3,++,down) sits so close beneath it that the two
# collide unless it is dropped clear.
UR_NUDGE = {
    (4, "+-", UP): (+0.085, +0.024),
    (3, "++", DN): (0.0, -0.032),
}


def level_table(zeeman, taus=(+1, -1), nmax=8):
    """{(n, mu, s): energy} at B = 30 T, gathered from the given valleys.

    Both valleys are needed even for the Mz = Mv = 0 panel.  The paper
    reserves mu = (+,+) for the n = -1 level at K and mu = (+,-) for it at
    K' (p.4), so the three n = 0 roots are labelled (--, -+, +-) at K but
    (--, -+, ++) at K'.  A label such as "(0,++,down)" therefore refers to
    a K' level, while "(1,+-,up)" beside it refers to a K level - they are
    exactly degenerate when Mz = Mv = 0, which is why the published panel
    lists them on one row.
    """
    out = {}
    for tau in taus:
        for s in (UP, DN):
            for L, name in ll.labelled(
                    ll.spectrum(B_FIELD, s, tau, V, zeeman, nmax=nmax), tau):
                out.setdefault((L.n, name, s), L.E)
    return out


def draw_box(ax):
    """Plain frame plus the E arrow - no ticks, no numbers."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)
    ax.annotate("", xy=(0.052, 0.885), xytext=(0.052, 0.075),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                mutation_scale=14))
    ax.text(0.045, 0.955, "E", transform=ax.transAxes, fontsize=15,
            ha="center", va="center")


def label_text(ax, side, parts, energy, fontsize=10.5):
    """Draw one hand-placed label.

    The paper colours each bracket by its own mu: red for mu = +-, black
    for mu = ++.  A combined label therefore mixes colours, which is done
    here by drawing the pieces left to right and measuring as we go.
    """
    pieces = [(f"({n},{mu},{arrow(s)})", MU_COLOR[mu]) for n, mu, s in parts]
    joined = " ;".join(p[0] for p in pieces)
    if side == "left":
        # right-aligned block: draw the whole string in the first colour,
        # then overprint the tail in black when the colours differ
        ax.text(X_LEFT, energy, joined, color=pieces[0][1], fontsize=fontsize,
                ha="right", va="center", transform=ax.get_yaxis_transform())
        if len(pieces) > 1 and pieces[1][1] != pieces[0][1]:
            tail = " ;" + " ;".join(p[0] for p in pieces[1:])
            ax.text(X_LEFT, energy, tail, color=pieces[1][1],
                    fontsize=fontsize, ha="right", va="center",
                    transform=ax.get_yaxis_transform())
    else:
        ax.text(X_MID, energy, joined, color=pieces[0][1], fontsize=fontsize,
                ha="left", va="center", transform=ax.get_yaxis_transform())
        if len(pieces) > 1 and pieces[1][1] != pieces[0][1]:
            head = pieces[0][0]
            ax.text(X_MID, energy, head + " ;", color=pieces[0][1],
                    fontsize=fontsize, ha="left", va="center",
                    transform=ax.get_yaxis_transform())


def draw_upper_left(ax):
    """Conduction band, Mz = Mv = 0."""
    tbl = level_table(zeeman=False)
    # Draw exactly the levels the published panel labels - nothing else.
    # This is what stops stray higher levels (e.g. n = 6 of mu = ++) from
    # appearing above the top labelled row.
    wanted = [key for _, parts in UL_LABELS for key in parts]
    shown = {k: tbl[k] for k in wanted if k in tbl}

    lo, hi = min(shown.values()), max(shown.values())
    span = hi - lo
    f0, f1 = LEVEL_BAND
    full = span / (f1 - f0)
    ax.set_ylim(lo - f0 * full, lo - f0 * full + full)
    ax.set_xlim(0, 1)
    draw_box(ax)

    # both columns carry the same levels: the valleys are degenerate here
    for key, E in shown.items():
        n, mu, s = key
        kw = {} if s > 0 else {"dashes": (5, 2.4)}
        for x0, x1 in (COL1, COL2):
            ax.plot([x0, x1], [E, E], color=MU_COLOR[mu], lw=1.6,
                    linestyle="-" if s > 0 else "--", **kw)

    for side, parts in UL_LABELS:
        key = parts[0]
        if key in shown:
            label_text(ax, side, parts, shown[key])

    # header, all on one line, and B=30 T at the bottom centre
    for x, txt, fs in [(0.235, r"$\mathrm{M_z}$ , $\mathrm{M_v}$ = 0", 14),
                       (0.425, r"$\tau$=+1", 14),
                       (0.635, "V=0 meV", 14),
                       (0.868, r"$\tau$=$-$1", 14)]:
        ax.text(x, 0.955, txt, transform=ax.transAxes, fontsize=fs,
                ha="center", va="center")
    ax.text(0.638, 0.045, "B=30 T", transform=ax.transAxes, fontsize=14,
            ha="center", va="center")


# =============================================================================
def draw_upper_right(ax):
    """Conduction band, Mz, Mv != 0 - the valleys are no longer degenerate."""
    tblK = level_table(zeeman=True, taus=(+1,))
    tblKp = level_table(zeeman=True, taus=(-1,))

    def rows(table, entries):
        out = []
        for side, parts in entries:
            key = parts[0][2]
            if key in table:
                out.append((side, parts, table[key]))
        return out

    left_rows = rows(tblK, UR_LEFT_COL)
    right_rows = rows(tblKp, UR_RIGHT_COL)

    allE = [E for _, _, E in left_rows + right_rows]
    lo, hi = min(allE), max(allE)
    span = hi - lo
    f0, f1 = 0.075, 0.815
    full = span / (f1 - f0)
    ax.set_ylim(lo - f0 * full, lo - f0 * full + full)
    ax.set_xlim(0, 1)
    draw_box(ax)

    for col, table, entries in ((UR_COL1, tblK, UR_LEFT_COL),
                                (UR_COL2, tblKp, UR_RIGHT_COL)):
        for side, parts in entries:
            for _, colour, key in parts:
                if key not in table:
                    continue
                s = key[2]
                kw = {} if s > 0 else {"dashes": (5, 2.4)}
                ax.plot(col, [table[key]] * 2, color=colour, lw=1.6,
                        linestyle="-" if s > 0 else "--", **kw)

    yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
    for side, parts, E in left_rows + right_rows:
        text = " ;".join(p[0] for p in parts)
        dx, dy = UR_NUDGE.get(parts[0][2], (0.0, 0.0))
        # "left"  : right-aligned, outside the left edge of column 1
        # "mid1"  : left-aligned,  just right of column 1
        # "mid2"  : right-aligned, just left of column 2
        # "right" : left-aligned,  outside the right edge of column 2
        ha = "right" if side in ("left", "mid2") else "left"
        ax.text(UR_X[side] + dx, E + dy * yspan, text, color=parts[0][1],
                fontsize=9.0, ha=ha, va="center",
                transform=ax.get_yaxis_transform())

    ax.text(0.545, 0.965, r"$\mathrm{M_z}$ , $\mathrm{M_v}$ $\neq$ 0",
            transform=ax.transAxes, fontsize=14, ha="center", va="center")
    ax.text(0.885, 0.885, "B=30 T", transform=ax.transAxes, fontsize=14,
            ha="center", va="center")
    ax.text(0.332, 0.855, r"$\tau$=+1", transform=ax.transAxes, fontsize=14,
            ha="center", va="center")
    ax.text(0.640, 0.855, r"$\tau$=$-$1", transform=ax.transAxes, fontsize=14,
            ha="center", va="center")
    ax.text(0.545, 0.035, "V=0 meV", transform=ax.transAxes, fontsize=14,
            ha="center", va="center")


# -----------------------------------------------------------------------------
# LOWER-LEFT PANEL: valence band, Mz = Mv = 0
#
# Only mu = -+ is drawn - the caption says "For simplicity we do not show
# the valence band levels for the second layer".  Everything is red.
# The valleys are degenerate here, so both columns carry the same levels
# and only one set of labels is drawn.
#
# The published panel pairs each (n,-+,up) solid line with the
# (n+1,-+,down) dashed line just beneath it; the spin-up labels sit on the
# far left and the spin-down ones in the middle gap.  Energies decrease
# downwards, so n increases downwards.
# -----------------------------------------------------------------------------
LL_LABELS = [
    ("left", (0, "-+", DN)),
    ("left", (0, "-+", UP)),
    ("mid",  (1, "-+", DN)),
    ("left", (1, "-+", UP)),
    ("mid",  (2, "-+", DN)),
    ("left", (2, "-+", UP)),
    ("mid",  (3, "-+", DN)),
    ("left", (3, "-+", UP)),
    ("mid",  (4, "-+", DN)),
    ("left", (4, "-+", UP)),
    ("mid",  (5, "-+", DN)),
    ("left", (5, "-+", UP)),
    ("mid",  (6, "-+", DN)),
    ("left", (6, "-+", UP)),
]

LL_COL1 = (0.240, 0.461)
LL_COL2 = (0.725, 0.946)
LL_X = {"left": 0.230, "mid": 0.529}
LL_BAND = (0.039, 0.781)


def draw_lower_left(ax):
    """Valence band, Mz = Mv = 0."""
    tbl = level_table(zeeman=False)
    rows = [(side, key, tbl[key]) for side, key in LL_LABELS if key in tbl]

    allE = [E for _, _, E in rows]
    lo, hi = min(allE), max(allE)
    f0, f1 = LL_BAND
    full = (hi - lo) / (f1 - f0)
    ax.set_ylim(lo - f0 * full, lo - f0 * full + full)
    ax.set_xlim(0, 1)
    draw_box(ax)

    for _, key, E in rows:
        s = key[2]
        kw = {} if s > 0 else {"dashes": (5, 2.4)}
        for col in (LL_COL1, LL_COL2):
            ax.plot(col, [E, E], color="red", lw=1.6,
                    linestyle="-" if s > 0 else "--", **kw)

    for side, key, E in rows:
        n, mu, s = key
        ax.text(LL_X[side], E, f"({n},{mu},{arrow(s)})", color="red",
                fontsize=10.0, ha="right" if side == "left" else "left",
                va="center", transform=ax.get_yaxis_transform())

    for x, txt in [(0.348, r"$\tau$=+1"), (0.588, "V=0 meV"),
                   (0.833, r"$\tau$=$-$1")]:
        ax.text(x, 0.945, txt, transform=ax.transAxes, fontsize=14,
                ha="center", va="center")
    ax.text(0.588, 0.855, "B=30 T", transform=ax.transAxes, fontsize=14,
            ha="center", va="center")


# -----------------------------------------------------------------------------
# LOWER-RIGHT PANEL: valence band, Mz, Mv != 0
#
# Again only mu = -+ is drawn.  The Zeeman terms lift the valley
# degeneracy, so the two columns sit at different energies: red for K on
# the left, blue for K' on the right.  Both columns carry the SAME
# sequence of (n, mu, s) labels, each on its own outer side.
# -----------------------------------------------------------------------------
LR_SEQUENCE = [
    (0, "-+", DN), (1, "-+", DN), (0, "-+", UP), (2, "-+", DN),
    (1, "-+", UP), (3, "-+", DN), (2, "-+", UP), (4, "-+", DN),
    (3, "-+", UP), (5, "-+", DN), (4, "-+", UP), (6, "-+", DN),
]

LR_COL1 = (0.188, 0.384)
LR_COL2 = (0.532, 0.736)
LR_X_LEFT = 0.179
LR_X_RIGHT = 0.778
LR_BAND = (0.056, 0.855)


def draw_lower_right(ax):
    """Valence band, Mz, Mv != 0."""
    tblK = level_table(zeeman=True, taus=(+1,))
    tblKp = level_table(zeeman=True, taus=(-1,))

    left = [(k, tblK[k]) for k in LR_SEQUENCE if k in tblK]
    right = [(k, tblKp[k]) for k in LR_SEQUENCE if k in tblKp]

    allE = [E for _, E in left + right]
    lo, hi = min(allE), max(allE)
    f0, f1 = LR_BAND
    full = (hi - lo) / (f1 - f0)
    ax.set_ylim(lo - f0 * full, lo - f0 * full + full)
    ax.set_xlim(0, 1)
    draw_box(ax)

    for rows, col, colour, x, ha in (
            (left, LR_COL1, "red", LR_X_LEFT, "right"),
            (right, LR_COL2, "blue", LR_X_RIGHT, "left")):
        for (n, mu, s), E in rows:
            kw = {} if s > 0 else {"dashes": (5, 2.4)}
            ax.plot(col, [E, E], color=colour, lw=1.6,
                    linestyle="-" if s > 0 else "--", **kw)
            ax.text(x, E, f"({n},{mu},{arrow(s)})", color=colour,
                    fontsize=10.0, ha=ha, va="center",
                    transform=ax.get_yaxis_transform())

    for x, txt in [(0.278, r"$\tau$=+1"), (0.463, "V=0 meV"),
                   (0.639, r"$\tau$=$-$1"), (0.843, "B=30 T")]:
        ax.text(x, 0.945, txt, transform=ax.transAxes, fontsize=14,
                ha="center", va="center")


# =============================================================================
if __name__ == "__main__":
    print("Fig. 4 - LL spectrum at B = 30 T, all four panels")

    # The published figure is a single 2x2 block: conduction on top,
    # valence below; Mz = Mv = 0 on the left, Mz, Mv != 0 on the right.
    fig = plt.figure(figsize=(13.8, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.055, wspace=0.045,
                          left=0.007, right=0.993, top=0.993, bottom=0.007)

    print("  upper-left  : conduction, Mz = Mv = 0")
    draw_upper_left(fig.add_subplot(gs[0, 0]))
    print("  upper-right : conduction, Mz, Mv != 0")
    draw_upper_right(fig.add_subplot(gs[0, 1]))
    print("  lower-left  : valence, Mz = Mv = 0")
    draw_lower_left(fig.add_subplot(gs[1, 0]))
    print("  lower-right : valence, Mz, Mv != 0")
    draw_lower_right(fig.add_subplot(gs[1, 1]))

    ps.save(fig, "bilayer_MoS2_fig4.png")
    plt.show()
