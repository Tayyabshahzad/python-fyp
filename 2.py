# =============================================================================
# FIGURE 2  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 2 caption, verbatim:
#   "Band structure of bilayer MoS2 for different electric fields E_z. The
#    left panel is for the conduction band and the right one for the
#    valence band."
#
# EQUATIONS USED (all in paper_equations.py):
#   Eq. (3)  p.2 - the fourth-degree equation whose roots are epsilon
#   Eq. (2)  p.2 - E = hbar*v_F*epsilon, converts those roots into eV
#
# The figure is an oblique "waterfall": the same band structure of Fig. 1
# drawn at four electric-field values, stacked along a V axis.
#
# LAYOUT read off the published panel:
#   * three hand-drawn axes only - mplot3d's own bounding box is switched
#     off, because the paper's axes are anchored on the data corner
#     (ka/pi = -0.1, V = 0) rather than floating around the box
#   * ka/pi axis along the bottom front, ticks -0.1, 0.0, 0.1, plus a
#     dense comb of unlabelled minor ticks
#   * E axis vertical at the front-left corner, ticks 0.85, 0.90, 0.95
#     (conduction) / -1.0, -0.9, -0.8 (valence), same minor-tick comb
#   * V axis diagonal along the top, ticks 0, 0.005, 0.01, 0.015
#   * one blue dashed "elbow" per non-zero V: a vertical drop from the V
#     axis to that slice's own band-edge level, then a horizontal run
#     along ka/pi at that level
#   * black = mu(++) / mu(--), red = mu(+-) / mu(-+); solid = spin up,
#     dotted = spin down.  At V = 0 the spins are degenerate, so the
#     dotted curve hides under the solid one exactly as in Fig. 1.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers "3d")

import paper_equations as pe
import paper_style as ps

ps.apply()
P = pe.P

# Fig. 2 plots a WIDER k range than Fig. 1 but labels only -0.1, 0.0, 0.1.
# Measured off the published panel: each slice's curves run off the TOP of
# the box rather than turning over inside it, which only happens if the
# range extends past the labelled ticks.  The lattice constant stays at
# the standard MoS2 value used everywhere else - it is the plotted range
# that differs, not the scale.
#
# The exact half-range is fixed by the published panel: its V = 0 curve
# meets the top-left corner of the box exactly, i.e. E(k_min, V=0) equals
# the top of the E axis.  Solving that for the conduction panel
# (E axis top 0.962 eV) gives ka/pi = 0.1436.
#
# The valence panel uses the SAME k range - it is the same figure - so its
# E limits follow from it rather than being chosen: the lower (black)
# band reaches -1.0496 eV at k_min when V = 0, which fixes the bottom of
# its box, and the upper (red) band peaks at -0.7294 eV at V = 15 meV,
# which sets the top.  Ticks then fall at -0.8, -0.9, -1.0 as published.
Q = np.linspace(-0.1436, 0.1436, 301)
K = (np.pi / P.A_LATTICE) * Q
I0 = np.argmin(np.abs(Q))

# The four field values the paper stacks along the V axis.
V_SLICES = [0.0, 0.005, 0.010, 0.015]

TAU = +1          # Fig. 2 shows one valley, as Fig. 1's K panel does
SIGN = "eq1"      # see discrepancy (D1) in paper_equations.py

# Band index -> colour, the same mu convention as Fig. 1.
BAND_COLOR = {0: "black", 1: "red", 2: "red", 3: "black"}
CONDUCTION, VALENCE = (2, 3), (0, 1)

K_LEFT, K_RIGHT = Q[0], Q[-1]


def bands(V):
    """Eq. (3) -> Eq. (2) at every k, for both spins."""
    up = np.empty((len(K), 4))
    dn = np.empty((len(K), 4))
    for i, k in enumerate(K):
        up[i] = pe.eq2_eq3_band_energies(k, +1, TAU, V, sign_convention=SIGN)
        dn[i] = pe.eq2_eq3_band_energies(k, -1, TAU, V, sign_convention=SIGN)
    return up, dn


def draw_axes(ax, e_lo, e_hi, e_ticks, e_fmt, label_gap=0.062):
    """The paper's three hand-drawn axes, anchored at the data corner."""
    lw = 1.7
    span = e_hi - e_lo

    # --- ka/pi axis: bottom front, at V = 0 -------------------------------
    ax.plot([K_LEFT, K_RIGHT], [0, 0], [e_lo, e_lo], color="black", lw=lw)
    for kt in np.linspace(K_LEFT, K_RIGHT, 31):          # minor comb
        ax.plot([kt, kt], [0, 0], [e_lo, e_lo - 0.007 * span],
                color="black", lw=0.9)
    for kt in (-0.1, 0.0, 0.1):
        ax.plot([kt, kt], [0, 0], [e_lo, e_lo - 0.014 * span],
                color="black", lw=lw)
        ax.text(kt, -0.0007, e_lo - 0.045 * span, f"{kt:.1f}",
                ha="center", va="top", fontsize=11)
    ax.text(0.012, -0.0012, e_lo - 0.115 * span, r"ka/$\pi$",
            ha="center", va="top", fontsize=13)

    # --- E axis: vertical at the front-left corner ------------------------
    ax.plot([K_LEFT, K_LEFT], [0, 0], [e_lo, e_hi], color="black", lw=lw)
    for et in np.linspace(e_lo, e_hi, 21):
        ax.plot([K_LEFT, K_LEFT - 0.0028], [0, 0], [et, et],
                color="black", lw=0.9)
    for et in e_ticks:
        ax.plot([K_LEFT, K_LEFT - 0.0055], [0, 0], [et, et],
                color="black", lw=lw)
        ax.text(K_LEFT - 0.013, 0, et, e_fmt(et), ha="right", va="center",
                fontsize=11)
    ax.text(K_LEFT - label_gap, 0, e_lo + 0.42 * span, "E (eV)",
            ha="center", va="center", fontsize=13)

    # --- V axis: diagonal along the top -----------------------------------
    ax.plot([K_LEFT, K_LEFT], [0, V_SLICES[-1]], [e_hi, e_hi],
            color="black", lw=lw)
    for vt in V_SLICES:
        gap = 0.030 * span if vt else 0.014 * span
        ax.text(K_LEFT - 0.016, vt, e_hi + gap, f"{vt:g}" if vt else "0",
                ha="right", va="bottom", fontsize=11)
    ax.text(K_LEFT - 0.085, V_SLICES[-1] * 0.55, e_hi + 0.085 * span,
            "V (eV)", ha="center", va="bottom", fontsize=13)


def clip(z, e_lo, e_hi):
    """Blank out the parts of a curve that leave the box.

    mplot3d does NOT clip lines to set_zlim the way a 2-D axes clips to
    set_ylim, so without this the arms of each parabola keep going and
    cross the V axis and the top of the frame.  The published panel shows
    them stopping at the edge, so the data is masked instead.
    """
    out = np.asarray(z, dtype=float).copy()
    out[(out > e_hi) | (out < e_lo)] = np.nan
    return out


def draw_waterfall(ax, band_indices, e_lo, e_hi):
    """One V slice per entry in V_SLICES, plus the blue elbow guides."""
    # black first, then red, matching the published draw order
    ordered = sorted(band_indices, key=lambda b: BAND_COLOR[b] != "black")
    for V in V_SLICES:
        up, dn = bands(V)
        y = np.full_like(Q, V)
        for b in ordered:
            colour = BAND_COLOR[b]
            ax.plot(Q, y, clip(up[:, b], e_lo, e_hi), color=colour, lw=1.05)
            if V != 0.0:                    # spins degenerate at V = 0
                ax.plot(Q, y, clip(dn[:, b], e_lo, e_hi), color=colour,
                        lw=0.85, linestyle=":", dashes=ps.DOTS)
        if V != 0.0:
            floor = up[I0, ordered[0]]
            ax.plot([K_LEFT, K_LEFT], [V, V], [e_hi, floor],
                    color="blue", lw=1.9, linestyle="--", dashes=(3.2, 2.2))
            ax.plot([K_LEFT, K_RIGHT], [V, V], [floor, floor],
                    color="blue", lw=1.9, linestyle="--", dashes=(3.2, 2.2))


def build_panel(ax, band_indices, e_lo, e_hi, e_ticks, e_fmt,
                label_gap=0.062):
    ax.set_axis_off()
    draw_waterfall(ax, band_indices, e_lo, e_hi)
    draw_axes(ax, e_lo, e_hi, e_ticks, e_fmt, label_gap)
    ax.set_xlim(K_LEFT, K_RIGHT)
    ax.set_ylim(0, V_SLICES[-1])
    ax.set_zlim(e_lo, e_hi)
    ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=-55)
    # elev/azim measured against the published panel: V must recede up
    # AND to the right, with ka/pi running along the bottom front and E
    # vertical at the left.  An azimuth on the other side of -90 mirrors
    # the V axis and stacks the slices to the left instead.
    ax.set_box_aspect((1.15, 2.0, 1.45))


# =============================================================================
def render(ax_target, which):
    """Draw one panel.  which is "conduction" or "valence"."""
    if which == "conduction":
        build_panel(ax_target, CONDUCTION, 0.818, 0.962,
                    [0.85, 0.90, 0.95], lambda v: f"{v:.2f}")
    else:
        # the valence tick labels carry a minus sign, so they are wider
        # than the conduction ones and the axis label sits further out
        build_panel(ax_target, VALENCE, -1.0496, -0.7150,
                    [-1.0, -0.9, -0.8], lambda v: f"{v:.1f}", label_gap=0.088)


def single(which, filename):
    """Save one panel on its own, for checking against the PDF."""
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    render(ax, which)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ps.save(fig, filename)
    plt.close(fig)


if __name__ == "__main__":
    print("Fig. 2 - solving Eq. (3) and applying Eq. (2) at four fields")

    # --- the published figure: both panels side by side ------------------
    fig = plt.figure(figsize=(12.4, 6.4))
    print("  conduction panel")
    render(fig.add_subplot(1, 2, 1, projection="3d"), "conduction")
    print("  valence panel")
    render(fig.add_subplot(1, 2, 2, projection="3d"), "valence")
    fig.subplots_adjust(left=0.04, right=0.97, top=0.98, bottom=0.04,
                        wspace=0.10)
    ps.save(fig, "bilayer_MoS2_fig2.png")

    # --- the same two panels on their own, for panel-by-panel checking
    #     against the PDF (identical drawing code, just one per file)
    print("  writing the individual panels")
    single("conduction", "bilayer_MoS2_fig2_conduction.png")
    single("valence", "bilayer_MoS2_fig2_valence.png")

    plt.show()
