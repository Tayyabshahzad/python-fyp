# =============================================================================
# FIGURE 3  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 3 caption, verbatim:
#   "Energy spectrum of bilayer MoS2 versus magnetic field B for Mz, Mv != 0,
#    and V = 0. The left (right) panel is for the conduction (valence) band.
#    The magenta curve shows the Fermi energy E_F versus B for an electron
#    density n_e = 1.9 x 10^13 cm^-2."
#
# EQUATIONS USED (all in paper_equations.py / landau_levels.py):
#   Eq. (5)  p.3  - quartic for the n >= 1 Landau levels
#   Eq. (4)  p.3  - E = hbar*omega_c*epsilon
#   Eq. (8)  p.4  - the single n = -1 level
#   Eq. (10) p.4  - cubic for the three n = 0 levels
#   Eq. (17) p.6  - Fermi energy at fixed electron density (magenta curve)
#
# LAYOUT measured off the published figure (300 dpi render of PDF page 4):
#   two panels, each 583 x 305 px  ->  aspect 1.911
#   column gap / panel width = 0.458
#   left  panel: y ticks 8.25 .. 8.55 step 0.05 , x ticks 0..40 step 10
#   right panel: y ticks -9.5 .. -7.5 step 0.5  , x ticks 0..40 step 10
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import landau_levels as ll

ps.apply()

B_GRID = np.linspace(0.4, 40.0, 240)
N_MAX = 24
V = 0.0            # Fig. 3 is V = 0
ZEEMAN = True      # caption: "for Mz, Mv != 0"

# Colour convention read off the published figure.  Valley sets the colour
# pair, mu picks within it; spin is carried by the line style.
COND_COLOR = {+1: {"+-": "red",  "++": "black"},
              -1: {"+-": "blue", "++": "green"}}
VAL_COLOR = {+1: {"-+": "red",  "--": "black"},
             -1: {"-+": "blue", "--": "green"}}

# Measured axis limits (energies are plotted in units of 10^-1 eV).
COND_YLIM = (8.222, 8.556)
VAL_YLIM = (-9.56, -7.29)


def yf(value, ylim):
    """Axes fraction of a data value - lets the legend rows below be
    placed at the ENERGIES they occupy in the published figure rather
    than at guessed fractions."""
    return (value - ylim[0]) / (ylim[1] - ylim[0])


def draw_fan(ax, mu_keys, colors, tau_order=(+1, -1)):
    """Every Landau level carrying one of mu_keys, over the field sweep.

    tau_order fixes which valley is drawn LAST, i.e. which colour ends up
    on top where the curves converge near B = 0.  The published panels
    differ: the conduction panel shows the K' colours (green/blue) on top,
    the valence panel shows the K colours (red/black) on top.  Measured
    off the figure, not chosen for looks.
    """
    for tau in tau_order:
        for s in (+1, -1):
            print(f"    branch tau={tau:+d} s={s:+d}")
            curves = ll.branch_curves(B_GRID, s, tau, V, ZEEMAN,
                                      mu_keys, N_MAX)
            style = "-" if s > 0 else ":"
            for mu in mu_keys:
                for arr in curves[mu]:
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
    print("Fig. 3 - Landau levels from Eq. (5), (8), (10); E_F from Eq. (17)")

    fig = plt.figure(figsize=(9.6, 3.65))
    gs = fig.add_gridspec(1, 2, wspace=0.458 * 0.62,
                          left=0.075, right=0.995, top=0.985, bottom=0.155)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])

    print("  conduction panel:")
    draw_fan(ax_c, ["+-", "++"], COND_COLOR)
    setup(ax_c, COND_YLIM,
          [8.25, 8.30, 8.35, 8.40, 8.45, 8.50, 8.55],
          ["8.25", "8.30", "8.35", "8.40", "8.45", "8.50", "8.55"])

    print("  Fermi energy, Eq. (17):")
    B_ef = B_GRID[B_GRID >= 1.0]
    EF = np.array([ll.eq17_fermi_energy(B, V, ZEEMAN) for B in B_ef])
    ax_c.plot(B_ef, EF * 10, color="magenta", lw=ps.LW_EF)

    print("  valence panel:")
    draw_fan(ax_v, ["-+", "--"], VAL_COLOR, tau_order=(-1, +1))
    setup(ax_v, VAL_YLIM, [-9.5, -9.0, -8.5, -8.0, -7.5],
          [r"$-9.5$", r"$-9.0$", r"$-8.5$", r"$-8.0$", r"$-7.5$"])

    # ---- conduction legend ----------------------------------------------
    # Positions read directly off the published panel.  The rows are placed
    # by ENERGY, using yf(), because that is how they are identifiable in
    # the figure:
    #     E_F                     magenta, top-left,      ~8.528
    #     E^{up,-}_{n,+-}         blue solid,  right,     ~8.340
    #     E^{dn,-}_{n,+-}         blue dotted, right,     ~8.300
    #     row 1 (~8.270): red solid | green dotted | green solid
    #     row 2 (~8.238): red dotted | black solid | black dotted | V = 0 meV
    C = COND_YLIM
    ps.legend_entry(ax_c, 0.020, yf(8.528, C), "-", "magenta",
                    r"$\mathrm{E_F}$", fontsize=9.5, backing=True,
                    width=0.185, height=0.070)

    ps.legend_entry(ax_c, 0.720, yf(8.340, C), "-", "blue",
                    ps.E_label("+-", +1, -1, n=True), backing=True,
                    width=0.275, height=0.078)
    ps.legend_entry(ax_c, 0.720, yf(8.300, C), ":", "blue",
                    ps.E_label("+-", -1, -1, n=True), backing=True,
                    width=0.275, height=0.078)

    for x, style, colour, mu, spin, tau in [
            (0.030, "-", "red",   "+-", +1, +1),
            (0.460, ":", "green", "++", -1, -1),
            (0.730, "-", "green", "++", +1, -1)]:
        ps.legend_entry(ax_c, x, yf(8.270, C), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.250, height=0.078)

    for x, style, colour, mu, spin, tau in [
            (0.030, ":", "red",   "+-", -1, +1),
            (0.250, "-", "black", "++", +1, +1),
            (0.480, ":", "black", "++", -1, +1)]:
        ps.legend_entry(ax_c, x, yf(8.238, C), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.235, height=0.078)

    ax_c.text(0.720, yf(8.238, C), "V = 0 meV", transform=ax_c.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    # ---- valence legend --------------------------------------------------
    # Two columns, K on the left and K' on the right, each row placed at
    # the ENERGY it occupies in the published panel:
    #     left  column x = 0.030 : -7.93, -8.25, -8.57, -8.90
    #     right column x = 0.252 : -8.10, -8.41, -8.72, -9.03
    #     V = 0 meV at x = 0.507, -8.93
    W = VAL_YLIM
    for energy, style, colour, mu, spin, tau in [
            (-7.93, "-", "red",   "-+", +1, +1),
            (-8.25, ":", "red",   "-+", -1, +1),
            (-8.57, "-", "black", "--", +1, +1),
            (-8.90, ":", "black", "--", -1, +1)]:
        ps.legend_entry(ax_v, 0.030, yf(energy, W), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.215, height=0.090)

    for energy, style, colour, mu, spin, tau in [
            (-8.10, "-", "blue",  "-+", +1, -1),
            (-8.41, ":", "blue",  "-+", -1, -1),
            (-8.72, "-", "green", "--", +1, -1),
            (-9.03, ":", "green", "--", -1, -1)]:
        ps.legend_entry(ax_v, 0.252, yf(energy, W), style, colour,
                        ps.E_label(mu, spin, tau, n=True), backing=True,
                        width=0.215, height=0.090)

    ax_v.text(0.507, yf(-8.93, W), "V = 0 meV", transform=ax_v.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    ps.save(fig, "bilayer_MoS2_fig3.png")
    plt.show()
