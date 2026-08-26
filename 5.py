# =============================================================================
# FIGURE 5  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 5 caption, verbatim:
#   "As in Fig. 3 but for V = 15 meV."
#
# So the physics, panels and colour scheme are exactly those of Fig. 3
# (see 3.py); only V and the axis ranges change.
#
# EQUATIONS USED (all in paper_equations.py / landau_levels.py):
#   Eq. (5)  p.3  - quartic for the n >= 1 Landau levels
#   Eq. (4)  p.3  - E = hbar*omega_c*epsilon
#   Eq. (8)  p.4  - the single n = -1 level
#   Eq. (10) p.4  - cubic for the three n = 0 levels
#   Eq. (17) p.6  - Fermi energy at fixed electron density (magenta curve)
#
# LAYOUT measured off the published figure (PDF page 5, lower half):
#   two panels, each 603 x 370 px  ->  aspect 1.628
#   column gap / panel width = 0.406
#   left  panel: y 8.0 .. 8.6 exactly, ticks every 0.1 ; x 0..40 step 10
#   right panel: y ticks -9.5 .. -7.5 step 0.5         ; x 0..40 step 10
#   left-panel legend: "V = 15 meV" ~8.43 and E_F ~8.385 at top-left;
#   two rows of four at ~8.085 and ~8.032, columns at x = 0.022, 0.276,
#   0.516, 0.827 (red | black | green | blue).
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import landau_levels as ll

ps.apply()

B_GRID = np.linspace(0.4, 40.0, 240)
N_MAX = 24
V = 0.015          # Fig. 5 is V = 15 meV - the one change from Fig. 3
ZEEMAN = True      # "As in Fig. 3", i.e. Mz, Mv != 0

COND_COLOR = {+1: {"+-": "red",  "++": "black"},
              -1: {"+-": "blue", "++": "green"}}
VAL_COLOR = {+1: {"-+": "red",  "--": "black"},
             -1: {"-+": "blue", "--": "green"}}

COND_YLIM = (8.0, 8.6)
VAL_YLIM = (-9.56, -7.29)


def yf(value, ylim):
    """Axes fraction of a data value, so legend rows can be placed at the
    energies they occupy in the published figure."""
    return (value - ylim[0]) / (ylim[1] - ylim[0])


def draw_fan(ax, order, colors):
    """Draw the Landau fan in an explicit (valley, mu) sequence.

    order is a list of (tau, mu) drawn front-to-back, so the LAST entry
    ends up on top where the curves converge near B = 0.  Fig. 5's
    Measured off the published panels:
      conduction  mu = ++ : black (K) on top - the wedge at B ~ 0 is black,
                            green emerges over it only from B ~ 2 onwards
                  mu = +- : red (K) on top   - the wedge at 8.15 is red
                  the +- fan is drawn last, so its red/blue lines cross
                  visibly over the ++ fan in the upper part of the panel
      valence     mu = -- : black (K) sits BEHIND, green (K') rides over
                            it as the wave pattern near -9.0
                  mu = -+ : BLUE (K') on top - the beating pattern at the
                            top of the panel has blue dotted lines lying
                            over the red, which is what produces the
                            interference look there.
    """
    mus = sorted({mu for _, mu in order})
    cache = {}
    for tau in (+1, -1):
        for s in (+1, -1):
            print(f"    branch tau={tau:+d} s={s:+d}")
            cache[(s, tau)] = ll.branch_curves(B_GRID, s, tau, V, ZEEMAN,
                                               mus, N_MAX)
    for tau, mu in order:
        for s in (+1, -1):
            style = "-" if s > 0 else ":"
            for arr in cache[(s, tau)][mu]:
                ax.plot(B_GRID, arr * 10, color=colors[tau][mu],
                        lw=ps.LW_FAN, linestyle=style, **ps.dashed(style))


def setup(ax, ylim, yticks, ylabels):
    ax.set_xlim(0, 40)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(r"$B$ (T)", fontsize=13, labelpad=2)
    ax.set_ylabel(r"E ($10^{-1}$ eV)", fontsize=12)
    ps.frame(ax)


# =============================================================================
if __name__ == "__main__":
    print("Fig. 5 - as Fig. 3 but V = 15 meV")

    fig = plt.figure(figsize=(9.6, 4.15))
    gs = fig.add_gridspec(1, 2, wspace=0.406 * 0.62,
                          left=0.075, right=0.995, top=0.985, bottom=0.145)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])

    print("  conduction panel:")
    draw_fan(ax_c, [(-1, "++"), (+1, "++"), (-1, "+-"), (+1, "+-")],
             COND_COLOR)
    setup(ax_c, COND_YLIM, [8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6],
          ["8.0", "8.1", "8.2", "8.3", "8.4", "8.5", "8.6"])

    print("  Fermi energy, Eq. (17):")
    B_ef = B_GRID[B_GRID >= 1.0]
    EF = np.array([ll.eq17_fermi_energy(B, V, ZEEMAN) for B in B_ef])
    ax_c.plot(B_ef, EF * 10, color="magenta", lw=ps.LW_EF)

    print("  valence panel:")
    draw_fan(ax_v, [(+1, "--"), (-1, "--"), (+1, "-+"), (-1, "-+")],
             VAL_COLOR)
    setup(ax_v, VAL_YLIM, [-9.5, -9.0, -8.5, -8.0, -7.5],
          [r"$-9.5$", r"$-9.0$", r"$-8.5$", r"$-8.0$", r"$-7.5$"])

    # ---- conduction legend, read off the published panel -----------------
    C = COND_YLIM
    ax_c.text(0.020, yf(8.430, C), "V = 15 meV", transform=ax_c.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ps.legend_entry(ax_c, 0.020, yf(8.385, C), "-", "magenta",
                    r"$\mathrm{E_F}$", fontsize=9.5, backing=True,
                    width=0.185, height=0.058)

    COLUMNS = [(0.022, "red", "+-", +1), (0.276, "black", "++", +1),
               (0.516, "green", "++", -1), (0.827, "blue", "+-", -1)]
    for energy, spin, style in [(8.085, +1, "-"), (8.032, -1, ":")]:
        for x, colour, mu, tau in COLUMNS:
            ps.legend_entry(ax_c, x, yf(energy, C), style, colour,
                            ps.E_label(mu, spin, tau, n=True), backing=True,
                            width=0.235, height=0.052)

    # ---- valence legend --------------------------------------------------
    # The caption says "As in Fig. 3", and the published panel repeats
    # Fig. 3's arrangement: K in the left column, K' in the right.
    W = VAL_YLIM
    for energy, style, colour, mu, spin, tau in [
            (-7.93, "-", "red",   "-+", +1, +1),
            (-8.25, ":", "red",   "-+", -1, +1),
            (-8.57, "-", "black", "--", +1, +1),
            (-8.90, ":", "black", "--", -1, +1)]:
        ps.legend_entry(ax_v, 0.030, yf(energy, W), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.215, height=0.080)

    for energy, style, colour, mu, spin, tau in [
            (-8.10, "-", "blue",  "-+", +1, -1),
            (-8.41, ":", "blue",  "-+", -1, -1),
            (-8.72, "-", "green", "--", +1, -1),
            (-9.03, ":", "green", "--", -1, -1)]:
        ps.legend_entry(ax_v, 0.252, yf(energy, W), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.215, height=0.080)

    ax_v.text(0.507, yf(-8.93, W), "V = 15 meV", transform=ax_v.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    ps.save(fig, "bilayer_MoS2_fig5.png")
    plt.show()
