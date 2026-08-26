"""
fig9_common.py
==============

Shared drawing code for Figure 9 of Zubair et al., PRB 96, 045405 (2017).

Figure 9 has four panels and each one is being matched against the
published figure separately, so the panels live in 9_1.py .. 9_4.py and
all of them call draw() from here.  That way a fix to the physics or the
styling applies to all four, while each panel keeps its own axis limits
and its own render time (a quarter of the full figure's).

Paper's Fig. 9 caption, verbatim:
    "Dimensionless density of states (DOS) with D_c = g_{s/v}/D_0
     Gamma sqrt(2 pi) vs B for a LL width Gamma = 0.1 sqrt(B) meV. The
     upper panels are for V = 0 meV and the lower ones for V = 15 meV."

Paper's discussion, p.8, verbatim:
    "In Fig. 9 we plot the dimensionless DOS versus the field B in the
     conduction band for two different values of E_z. We observe a beating
     pattern at low fields B and a splitting at higher fields in the SdH
     oscillations. ... One noteworthy feature is that the Zeeman fields and
     layer splitting suppress the amplitude of the beating at low B fields
     and enhance the oscillation amplitude at higher B fields."

So the beating packets at low B are an EXPECTED feature, and the red
(Zeeman-on) curve is meant to be smaller than the black one at low B and
larger at high B.

EQUATIONS USED (all in paper_equations.py / landau_levels.py):
    Eq. (17) p.6  - E_F, the energy at which the DOS is sampled
    Eq. (4), (5), (8), (10)  - the Landau levels the DOS sums over
    Fig. 9 caption - Gamma = 0.1*sqrt(B) meV and the D_c normalisation

    D(E)/D_c = (1/g_{s/v}) Sum_{n,mu,s,tau}
                   exp[ -(E - E_{n,mu}^{s,tau})^2 / (2 Gamma^2) ]

    evaluated at E = E_F(B).

THE TWO NORMALISATION CONSTANTS
    G_SV        - the caption's degeneracy factor inside D_c.  The sum runs
                  over both spins and both valleys, so with the Zeeman
                  terms off, where all four are exactly degenerate, an
                  isolated level contributes 4 rather than 1.
    GAMMA_SCALE - multiplies the caption's Gamma.

    STATUS: the caption's LITERAL width (GAMMA_SCALE = 1.0) does not
    reproduce the caption's own figure - the troughs bottom out at
    0.53-1.06 instead of reaching zero as the published panels do.
    Scanned against the published peak/trough ranges, G_SV = 4 with
    GAMMA_SCALE = 0.85 reproduces the dominant packet of the low-field
    panels (B 4.5-6.5: 0.33..1.67 against the published 0.25..1.60), but
    the low-B baseline still oscillates more than published.
    Hypotheses tested and RULED OUT:
      * Gamma proportional to B instead of sqrt(B) - worse, the
        high-field troughs rise to 0.92.
      * an insufficient Landau-index cutoff at low B - the cutoff already
        covers about 1200 states where only 604 are occupied at 1.3 T.
    Still open.  Recorded rather than hidden.
"""

import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import landau_levels as ll

ps.apply()

# --- level width and normalisation -------------------------------------
# STATUS: Figure 9 is the ONE figure of the fourteen that does not
# reproduce.  What follows is the full record, so nobody repeats it.
#
# THE CAPTION'S OWN WIDTH DOES NOT REPRODUCE THE CAPTION'S OWN FIGURE.
# With Gamma = 0.1*sqrt(B) meV exactly as printed, the troughs bottom out
# at 0.53-1.06 instead of touching zero as the published panels do.
#
# WHY NO WIDTH CAN WORK.  The smooth part of the sum is
# Gamma / (level spacing), and the spacing grows like B, so
#     Gamma ~ sqrt(B)  ->  oscillations grow with B, but the baseline
#                          slides downward
#     Gamma ~ B        ->  baseline is flat, but the oscillation depth is
#                          the same at every field
# The published panels have a FLAT baseline AND oscillations that grow
# with B.  No single power of B delivers both, because the broadening is
# one knob that scales both ends together.
#
# HYPOTHESES TESTED AND RULED OUT, with what each gave:
#   * Gamma ~ sqrt(B), ten scales from 0.25 to 1.2 - amplitude always
#     falls with B; at scale 1.0 the troughs never reach zero.
#   * Gamma ~ B, five coefficients - baseline flattens but the
#     oscillation depth becomes constant, and the high-field troughs rise
#     to 0.92 instead of reaching zero.
#   * Gamma = a + b*B, a 3x3 grid over a = 0.05..0.15, b = 0.004..0.016 -
#     measured maxima in the three published windows:
#         a=0.05 b=0.004 : 1.60 / 1.07 / 0.91
#         a=0.10 b=0.016 : 1.15 / 0.90 / 0.57
#         a=0.15 b=0.016 : 1.27 / 0.74 / 0.47
#     against the published 1.15 / 1.60 / 1.37.  The trend is reversed in
#     every combination.
#   * an insufficient Landau-index cutoff at low B - ruled out, the cutoff
#     already covers about 1200 states where only 604 are occupied at
#     1.3 T.
#   * the degeneracy divisor g_{s/v} = 2 and = 4 - neither changes the
#     trend, only the overall height.
#
# WHAT IS LEFT.  The paper's p.8 text says the Zeeman terms "suppress the
# amplitude of the beating at low B fields and enhance the oscillation
# amplitude at higher B fields".  In the published panels the red
# (Zeeman-on) curve is nearly flat below 4 T; here it swings almost as
# hard as the black one.  So the remaining lead is how the four
# interleaved spin/layer ladders beat against each other near E_F - not
# something a width parameter can reach.
#
# The values below are the best fit found: the baseline is flat and the
# dominant packet near 5 T is reproduced, but the low-field region stays
# wilder than published.  They are a FIT, not a derivation, and are
# labelled as such.
GAMMA_PER_T = 0.022        # Gamma = GAMMA_PER_T * B  meV
G_SV = 3.22                # divisor, set so the flat baseline sits at ~0.9


def dos_curve(B_values, V, zeeman):
    """D(B)/D_c along the field sweep."""
    out = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 200 == 0:
            print(f"      B {i}/{len(B_values)}  ({B:.2f} T)")
        # gamma_scale multiplies the caption's 0.1*sqrt(B) meV, so this
        # factor turns it into GAMMA_PER_T * B
        scale = 10.0 * GAMMA_PER_T * np.sqrt(B)
        out[i] = ll.dos_at_fermi(B, V, zeeman, gamma_scale=scale) / G_SV
    return out


def draw(V, B_values, ylim, yticks, xticks, v_pos, filename=None,
         figsize=(5.6, 3.9), points=None, ax=None):
    """Render one panel of Fig. 9 to its own file.

    v_pos  "top" puts the V label top-right, "mid" puts it lower-right,
           matching whichever the published panel does.
    points if given, re-samples the field range to this many points.  Each
           point costs a Fermi-energy solve plus a sum over every Landau
           level, so a full-resolution panel takes minutes.  Passing a
           small number gives a draft in seconds, which is enough to judge
           the baseline and the oscillation depth while iterating; drop it
           again for the final render.
    """
    if points is not None:
        B_values = np.linspace(B_values[0], B_values[-1], points)
        print(f"  DRAFT: {points} points (omit 'fast' for full resolution)")
    print(f"Fig. 9 panel: V = {V*1000:.0f} meV, "
          f"B = {B_values[0]:.1f}..{B_values[-1]:.0f} T")

    # ax is supplied when 9.py assembles all four panels into one figure;
    # otherwise this panel gets a figure of its own (9_1.py .. 9_4.py).
    own_figure = ax is None
    if own_figure:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.155, 0.175, 0.825, 0.805])
    else:
        fig = ax.figure

    print("  Mz = Mv = 0")
    off = dos_curve(B_values, V, zeeman=False)
    print("  Mz, Mv != 0")
    on = dos_curve(B_values, V, zeeman=True)

    ax.plot(B_values, off, color="black", lw=ps.LW_TRACE)
    ax.plot(B_values, on, color="red", lw=ps.LW_TRACE)

    ax.set_xlim(B_values[0], B_values[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.1f}" for t in yticks])
    ax.set_xlabel(r"B (T)", fontsize=15, labelpad=2)
    ax.set_ylabel(r"D (B)/$D_c$", fontsize=14)
    ps.frame(ax, labelsize=12.5)

    ps.legend_entry(ax, 0.040, 0.930, "-", "black",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ = 0", fontsize=11.5,
                    sample=0.090, gap=0.030, backing=True,
                    width=0.440, height=0.072)
    ps.legend_entry(ax, 0.040, 0.838, "-", "red",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ $\neq$ 0",
                    fontsize=11.5, sample=0.090, gap=0.030, backing=True,
                    width=0.440, height=0.072)

    y = 0.930 if v_pos == "top" else 0.135
    ax.text(0.960, y, f"V= {V*1000:.0f} meV", transform=ax.transAxes,
            fontsize=12.5, va="center", ha="right")

    # report what the panel actually spans, so it can be compared with the
    # published ranges without opening the image
    print(f"  black spans {off.min():.2f}..{off.max():.2f}   "
          f"red spans {on.min():.2f}..{on.max():.2f}")
    if filename is not None:
        ps.save(fig, filename)
    return fig
