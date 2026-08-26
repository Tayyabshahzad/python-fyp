# =============================================================================
# FIGURE 7  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 7 caption, verbatim:
#   "As in Fig. 5 but for V = 15 meV."
#
# NOTE ON THE CAPTION.  It points at Fig. 5, but Fig. 5 is already the
# V = 15 meV version of Fig. 3 and shows the conduction AND valence bands.
# Fig. 7's actual content - conduction band only, left panel Mz = Mv = 0,
# right panel Mz, Mv != 0, magenta E_F on both - is the layout of Fig. 6.
# So the printed cross-reference appears to be a slip for "As in Fig. 6".
# The figure is reproduced from what is drawn, not from the cross-reference.
#
# EQUATIONS USED (all in paper_equations.py / landau_levels.py):
#   Eq. (5)  p.3  - quartic for the n >= 1 Landau levels
#   Eq. (4)  p.3  - E = hbar*omega_c*epsilon
#   Eq. (8)  p.4  - the single n = -1 level
#   Eq. (10) p.4  - cubic for the three n = 0 levels
#   Eq. (17) p.6  - Fermi energy at fixed electron density (magenta curve)
#
# THE LEFT-PANEL EQUALITIES DIFFER FROM FIG. 6's
#   Fig. 6 is at V = 0, where the valleys are degenerate spin by spin:
#       E^{up,+}_{n,mu} = E^{up,-}_{n,mu}      (same spin)
#   Fig. 7 is at V = 15 meV, where that is broken but a weaker symmetry
#   survives, pairing OPPOSITE spins across the valleys:
#       E^{up,+}_{n,mu} = E^{down,-}_{n,mu}    (opposite spin)
#   Both are transcribed from the published legends.
#
# LAYOUT read off the published panels:
#   left  : y 8.100 .. 8.605, ticks 8.1 .. 8.6 step 0.1 ; x 0..40
#   right : y 8.000 .. 8.605, ticks 8.0 .. 8.6 step 0.1 ; x 0..40
#   Both fans start from two flat levels: mu = ++ at 8.45 and mu = +- at
#   8.15, which is the 2V conduction splitting of Fig. 1 seen at B -> 0.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import landau_levels as ll

ps.apply()

B_GRID = np.linspace(0.4, 40.0, 240)
N_MAX = 24
V = 0.015          # Fig. 7 is V = 15 meV

COLOR = {+1: {"+-": "red",  "++": "black"},
         -1: {"+-": "blue", "++": "green"}}

LEFT_YLIM = (8.100, 8.605)
RIGHT_YLIM = (8.000, 8.605)


def yf(value, ylim):
    """Axes fraction of a data value, so legend rows sit at the energies
    they occupy in the published figure."""
    return (value - ylim[0]) / (ylim[1] - ylim[0])


def draw_fan(ax, order, zeeman):
    """Draw the conduction fan in an explicit (valley, mu) sequence; the
    LAST entry ends up on top where the curves converge near B = 0."""
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
    ax.set_yticklabels([f"{t:.1f}" for t in yticks])
    ax.set_xlabel(r"$B$ (T)", fontsize=13, labelpad=2)
    ax.set_ylabel(r"E ($10^{-1}$ eV)", fontsize=12)
    ps.frame(ax)


def fermi_curve(ax, zeeman):
    """Eq. (17): the magenta E_F(B) curve."""
    print("    Fermi energy, Eq. (17)")
    B_ef = B_GRID[B_GRID >= 1.0]
    EF = np.array([ll.eq17_fermi_energy(B, V, zeeman) for B in B_ef])
    ax.plot(B_ef, EF * 10, color="magenta", lw=ps.LW_EF)


def flipped_label(mu, spin):
    """The left panel's labels pair OPPOSITE spins across the valleys:
    E^{s,+}_{n,mu} = E^{-s,-}_{n,mu}.  See the note in the file header."""
    return (ps.E_label(mu, spin, +1, n=True) + " = "
            + ps.E_label(mu, -spin, -1, n=True))


# =============================================================================
if __name__ == "__main__":
    print("Fig. 7 - conduction LLs vs B at V = 15 meV")

    fig = plt.figure(figsize=(9.6, 4.05))
    gs = fig.add_gridspec(1, 2, wspace=0.26,
                          left=0.075, right=0.995, top=0.985, bottom=0.150)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # ---------------- LEFT: Mz = Mv = 0 -----------------------------------
    # Only K is drawn; K' repeats it with the spins exchanged, which is
    # what the published legend's equalities record.
    print("  left panel (Mz = Mv = 0):")
    draw_fan(ax_l, [(+1, "++"), (+1, "+-")], zeeman=False)
    setup(ax_l, LEFT_YLIM, [8.1, 8.2, 8.3, 8.4, 8.5, 8.6])
    fermi_curve(ax_l, zeeman=False)

    L = LEFT_YLIM
    ps.legend_entry(ax_l, 0.030, yf(8.400, L), "-", "magenta",
                    r"$\mathrm{E_F}$", fontsize=9.5, backing=True,
                    width=0.180, height=0.034, sample=0.070)
    ax_l.text(0.030, yf(8.118, L), "V = 15 meV", transform=ax_l.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ps.legend_entry(ax_l, 0.600, yf(8.280, L), "-", "red",
                    flipped_label("+-", +1), backing=True,
                    width=0.385, height=0.034)
    ps.legend_entry(ax_l, 0.600, yf(8.210, L), ":", "red",
                    flipped_label("+-", -1), backing=True,
                    width=0.385, height=0.034)
    ps.legend_entry(ax_l, 0.300, yf(8.118, L), ":", "black",
                    flipped_label("++", -1), backing=True,
                    width=0.385, height=0.034)
    ps.legend_entry(ax_l, 0.725, yf(8.118, L), "-", "black",
                    flipped_label("++", +1), backing=True,
                    width=0.385, height=0.034)

    # ---------------- RIGHT: Mz, Mv != 0 ----------------------------------
    # Same content and draw order as Fig. 5's conduction panel: black over
    # green for mu = ++, red over blue for mu = +-.
    print("  right panel (Mz, Mv != 0):")
    draw_fan(ax_r, [(-1, "++"), (+1, "++"), (-1, "+-"), (+1, "+-")],
             zeeman=True)
    setup(ax_r, RIGHT_YLIM, [8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6])
    fermi_curve(ax_r, zeeman=True)

    R = RIGHT_YLIM
    ax_r.text(0.030, yf(8.430, R), "V = 15 meV", transform=ax_r.transAxes,
              fontsize=9.5, va="center", zorder=6,
              bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ps.legend_entry(ax_r, 0.050, yf(8.375, R), "-", "magenta",
                    r"$\mathrm{E_F}$", fontsize=9.5, backing=True,
                    width=0.180, height=0.034, sample=0.070)

    COLUMNS = [(0.025, "red", "+-", +1), (0.275, "black", "++", +1),
               (0.525, "green", "++", -1), (0.775, "blue", "+-", -1)]
    for energy, spin, style in [(8.112, +1, "-"), (8.042, -1, ":")]:
        for x, colour, mu, tau in COLUMNS:
            ps.legend_entry(ax_r, x, yf(energy, R), style, colour,
                            ps.E_label(mu, spin, tau, n=True), backing=True,
                            width=0.195, height=0.034)

    ps.save(fig, "bilayer_MoS2_fig7.png")
    plt.show()
