# =============================================================================
# FIGURE 6  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 6 caption, verbatim:
#   "LLs in bilayer MoS2, in the conduction band, versus magnetic field B
#    for V = 0 meV. The left panel is for Mz = Mv = 0 and the right one
#    for Mz != Mv != 0. The magenta curve shows the Fermi energy E_F
#    versus B."
#
# EQUATIONS USED (all in paper_equations.py / landau_levels.py):
#   Eq. (5)  p.3  - quartic for the n >= 1 Landau levels
#   Eq. (4)  p.3  - E = hbar*omega_c*epsilon
#   Eq. (8)  p.4  - the single n = -1 level
#   Eq. (10) p.4  - cubic for the three n = 0 levels
#   Eq. (17) p.6  - Fermi energy at fixed electron density (magenta curve)
#
# WHY THE TWO PANELS DIFFER
#   Left  (Mz = Mv = 0): the Zeeman terms are off, so K and K' are exactly
#     degenerate.  Only ONE set of curves exists and the published legend
#     records this with equalities, e.g. E^{up,+}_{n,+-} = E^{up,-}_{n,+-}.
#     Two colours only: red for mu = +-, black for mu = ++.
#   Right (Mz, Mv != 0): the degeneracy is lifted and all four colours
#     appear - this panel is the same content as Fig. 3's conduction panel.
#
# LAYOUT read off the published panels:
#   left  : y 8.285 .. 8.555, ticks 8.30 .. 8.55 step 0.05 ; x 0..40
#   right : y 8.222 .. 8.556, ticks 8.25 .. 8.55 step 0.05 ; x 0..40
#   Legend rows are placed by the ENERGY they sit at in the figure, via
#   yf(), rather than by guessed axes fractions.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import landau_levels as ll

ps.apply()

B_GRID = np.linspace(0.4, 40.0, 240)
N_MAX = 24
V = 0.0            # Fig. 6 is V = 0 meV

# Colour convention, identical to Figs. 3 and 5.
COLOR = {+1: {"+-": "red",  "++": "black"},
         -1: {"+-": "blue", "++": "green"}}

LEFT_YLIM = (8.285, 8.555)
RIGHT_YLIM = (8.222, 8.556)


def yf(value, ylim):
    """Axes fraction of a data value, so legend rows can be placed at the
    energies they occupy in the published figure."""
    return (value - ylim[0]) / (ylim[1] - ylim[0])


def draw_fan(ax, order, zeeman):
    """Draw the conduction fan in an explicit (valley, mu) sequence.

    The LAST entry ends up on top where the curves converge near B = 0.
    Measured off the published panels, not chosen for looks.
    """
    mus = sorted({mu for _, mu in order})
    cache = {}
    for tau in sorted({t for t, _ in order}):
        for s in (+1, -1):
            print(f"    branch tau={tau:+d} s={s:+d}")
            cache[(s, tau)] = ll.branch_curves(B_GRID, s, tau, V, zeeman,
                                               mus, N_MAX)
    for tau, mu in order:
        for s in (+1, -1):
            style = "-" if s > 0 else ":"
            for arr in cache[(s, tau)][mu]:
                ax.plot(B_GRID, arr * 10, color=COLOR[tau][mu],
                        lw=ps.LW_FAN, linestyle=style, **ps.dashed(style))


def setup(ax, ylim, yticks):
    ax.set_xlim(0, 40)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks])
    ax.set_xlabel(r"$B$ (T)", fontsize=13, labelpad=2)
    ax.set_ylabel(r"E ($10^{-1}$ eV)", fontsize=12)
    ps.frame(ax)


def fermi_curve(ax, zeeman):
    """Eq. (17): the magenta E_F(B) curve."""
    print("    Fermi energy, Eq. (17)")
    B_ef = B_GRID[B_GRID >= 1.0]
    EF = np.array([ll.eq17_fermi_energy(B, V, zeeman) for B in B_ef])
    ax.plot(B_ef, EF * 10, color="magenta", lw=ps.LW_EF)


def eq_label(mu, spin):
    """The left panel's labels carry the valley equality explicitly,
    e.g. E^{up,+}_{n,+-} = E^{up,-}_{n,+-}."""
    return (ps.E_label(mu, spin, +1, n=True) + " = "
            + ps.E_label(mu, spin, -1, n=True))


# =============================================================================
if __name__ == "__main__":
    print("Fig. 6 - conduction LLs vs B at V = 0")

    fig = plt.figure(figsize=(9.6, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.26,
                          left=0.075, right=0.995, top=0.985, bottom=0.155)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # ---------------- LEFT: Mz = Mv = 0 -----------------------------------
    # Only K is drawn: K' is exactly degenerate with it, which is what the
    # published legend's equalities state.
    print("  left panel (Mz = Mv = 0):")
    draw_fan(ax_l, [(+1, "+-"), (+1, "++")], zeeman=False)
    setup(ax_l, LEFT_YLIM, [8.30, 8.35, 8.40, 8.45, 8.50, 8.55])
    fermi_curve(ax_l, zeeman=False)

    L = LEFT_YLIM
    ps.legend_entry(ax_l, 0.030, yf(8.535, L), "-", "magenta",
                    r"$\mathrm{E_F}$", fontsize=9.5, backing=True,
                    width=0.175, height=0.042)
    ax_l.text(0.720, yf(8.500, L), "V = 0 meV", transform=ax_l.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ps.legend_entry(ax_l, 0.600, yf(8.412, L), "-", "red",
                    eq_label("+-", +1), backing=True,
                    width=0.400, height=0.042)
    ps.legend_entry(ax_l, 0.600, yf(8.347, L), ":", "red",
                    eq_label("+-", -1), backing=True,
                    width=0.400, height=0.042)
    ps.legend_entry(ax_l, 0.025, yf(8.315, L), ":", "black",
                    eq_label("++", -1), backing=True,
                    width=0.400, height=0.042)
    ps.legend_entry(ax_l, 0.525, yf(8.315, L), "-", "black",
                    eq_label("++", +1), backing=True,
                    width=0.400, height=0.042)

    # ---------------- RIGHT: Mz, Mv != 0 ----------------------------------
    # Same content as Fig. 3's conduction panel, so the same draw order and
    # the same legend arrangement are used.
    print("  right panel (Mz, Mv != 0):")
    draw_fan(ax_r, [(+1, "+-"), (+1, "++"), (-1, "+-"), (-1, "++")],
             zeeman=True)
    setup(ax_r, RIGHT_YLIM, [8.25, 8.30, 8.35, 8.40, 8.45, 8.50, 8.55])
    fermi_curve(ax_r, zeeman=True)

    R = RIGHT_YLIM
    ps.legend_entry(ax_r, 0.020, yf(8.528, R), "-", "magenta",
                    r"$\mathrm{E_F}$", fontsize=9.5, backing=True,
                    width=0.175, height=0.042)
    ps.legend_entry(ax_r, 0.720, yf(8.340, R), "-", "blue",
                    ps.E_label("+-", +1, -1, n=True), backing=True,
                    width=0.270, height=0.045)
    ps.legend_entry(ax_r, 0.720, yf(8.300, R), ":", "blue",
                    ps.E_label("+-", -1, -1, n=True), backing=True,
                    width=0.270, height=0.045)
    for x, style, colour, mu, spin, tau in [
            (0.030, "-", "red",   "+-", +1, +1),
            (0.460, ":", "green", "++", -1, -1),
            (0.730, "-", "green", "++", +1, -1)]:
        ps.legend_entry(ax_r, x, yf(8.270, R), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.245, height=0.045)
    for x, style, colour, mu, spin, tau in [
            (0.030, ":", "red",   "+-", -1, +1),
            (0.250, "-", "black", "++", +1, +1),
            (0.480, ":", "black", "++", -1, +1)]:
        ps.legend_entry(ax_r, x, yf(8.238, R), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.230, height=0.045)
    ax_r.text(0.720, yf(8.238, R), "V = 0 meV", transform=ax_r.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    ps.save(fig, "bilayer_MoS2_fig6.png")

    # the two panels on their own, for checking against the PDF
    plt.show()
