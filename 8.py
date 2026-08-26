# =============================================================================
# FIGURE 8  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 8 caption, verbatim:
#   "Fermi energy E_F versus magnetic field B at T = 1 K. The upper panels
#    are for V = 0 meV and the lower ones for V = 15 meV. The panels differ
#    only in the range of B."
#
# EQUATIONS USED (all in paper_equations.py / landau_levels.py):
#   Eq. (17) p.6  - E_F from the fixed electron density n_e = 1.9e13 cm^-2
#   Eq. (4), (5), (8), (10)  - the Landau levels that Eq. (17) sums over
#
# WHAT THE FIGURE SHOWS
#   As B rises each Landau level empties in turn, so E_F saws up and down.
#   Black is Mz = Mv = 0, red is Mz, Mv != 0.  Switching the Zeeman terms
#   on splits each level, which doubles the number of teeth and halves
#   their height - that is the whole point of the comparison.
#
# LAYOUT read off the four published panels:
#   (a) V=0    B  2..13 : y 8.5118 .. 8.5207, ticks 8.512 .. 8.520 step .002
#   (b) V=0    B 13..40 : y 8.4855 .. 8.5455, ticks 8.49 .. 8.54  step .01
#   (c) V=15   B  2..13 : y 8.5068 .. 8.5208, ticks 8.508 .. 8.520 step .002
#   (d) V=15   B 13..40 : y 8.4965 .. 8.5405, ticks 8.50 .. 8.54  step .01
#   Legend: two rows top-left, a short colour sample then BLACK text.
#   The V label sits bottom-left in every panel EXCEPT (d), where the
#   published panel puts it bottom-right to clear the curves.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import landau_levels as ll

ps.apply()

# The published split: a fine low-field range and a coarser high-field one.
# The low-field panels resolve oscillations only a few tenths of a meV
# apart, so they need a much denser B grid than the fan figures did.
B_LOW = np.linspace(2.0, 13.0, 800)
B_HIGH = np.linspace(13.0, 40.0, 500)

PANELS = [
    # (V, B grid, ylim, yticks, tick format, V-label position)
    (0.000, B_LOW,  (8.5118, 8.5207),
     [8.512, 8.514, 8.516, 8.518, 8.520], "{:.3f}", "left"),
    (0.000, B_HIGH, (8.4855, 8.5455),
     [8.49, 8.50, 8.51, 8.52, 8.53, 8.54], "{:.2f}", "left"),
    (0.015, B_LOW,  (8.5068, 8.5208),
     [8.508, 8.510, 8.512, 8.514, 8.516, 8.518, 8.520], "{:.3f}", "left"),
    (0.015, B_HIGH, (8.4965, 8.5405),
     [8.50, 8.51, 8.52, 8.53, 8.54], "{:.2f}", "right"),
]

XTICKS = {id(B_LOW): [2, 4, 6, 8, 10, 12],
          id(B_HIGH): [15, 20, 25, 30, 35, 40]}


def fermi_curve(B_values, V, zeeman):
    """Eq. (17) evaluated at every B of the grid, in units of 1e-1 eV."""
    out = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 200 == 0:
            print(f"      B {i}/{len(B_values)}  ({B:.2f} T)")
        out[i] = ll.eq17_fermi_energy(B, V, zeeman)
    return out * 10


def draw_panel(ax, V, B_values, ylim, yticks, fmt, v_side):
    print(f"  panel: V = {V*1000:.0f} meV, B = {B_values[0]:.0f}"
          f"..{B_values[-1]:.0f} T")
    print("    Mz = Mv = 0")
    off = fermi_curve(B_values, V, zeeman=False)
    print("    Mz, Mv != 0")
    on = fermi_curve(B_values, V, zeeman=True)

    ax.plot(B_values, off, color="black", lw=ps.LW_TRACE)
    ax.plot(B_values, on, color="red", lw=ps.LW_TRACE)

    ax.set_xlim(B_values[0], B_values[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(XTICKS[id(B_values)])
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(t) for t in yticks])
    ax.set_xlabel(r"B (T)", fontsize=14, labelpad=2)
    ax.set_ylabel(r"$E_F$ ($10^{-1}$ eV)", fontsize=13)
    ps.frame(ax, labelsize=11.5)

    # legend: two rows at the top left, colour in the sample only
    ps.legend_entry(ax, 0.040, 0.930, "-", "black",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ = 0", fontsize=11,
                    sample=0.090, gap=0.030)
    ps.legend_entry(ax, 0.040, 0.838, "-", "red",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ $\neq$ 0", fontsize=11,
                    sample=0.090, gap=0.030)

    # the V label: bottom-left everywhere except the V=15 meV high-field
    # panel, where the published figure moves it to the bottom right
    x, ha = (0.075, "left") if v_side == "left" else (0.960, "right")
    ax.text(x, 0.075, f"V= {V*1000:.0f} meV", transform=ax.transAxes,
            fontsize=12, va="center", ha=ha)


# =============================================================================
if __name__ == "__main__":
    print("Fig. 8 - Fermi energy from Eq. (17) at T = 1 K")

    fig = plt.figure(figsize=(10.4, 6.9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                          left=0.085, right=0.985, top=0.985, bottom=0.095)

    for idx, (V, B_values, ylim, yticks, fmt, v_side) in enumerate(PANELS):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        draw_panel(ax, V, B_values, ylim, yticks, fmt, v_side)

    ps.save(fig, "bilayer_MoS2_fig8.png")
    plt.show()
