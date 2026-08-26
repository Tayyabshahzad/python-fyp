# =============================================================================
# FIGURE 13  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 13 caption, verbatim:
#   "Spin Ps and valley Pv polarizations versus magnetic field B at T = 1 K.
#    The parameters are the same as in Fig. 11 for Mz != Mv != 0."
#
# EQUATIONS USED (in sigma_xx_common.py / paper_equations.py):
#   Eq. (29),(30) p.11 - the spin and valley polarisations
#   Eq. (28)  p.10 - the branch conductivities they are built from
#   Eq. (17)  p.6  - E_F at the paper's fixed electron density
#   Eq. (4),(5),(8),(10),(6),(7) - the Landau levels and their coefficients
#
# WHICH V?  THE PAPER CONTRADICTS ITSELF
#   The caption says the parameters are "the same as in Fig. 11", and
#   Fig. 11 is V = 15 meV.  But the body text on p.11 says, of this very
#   figure, verbatim:
#       "We plot the spin Ps (black solid curve) and Pv (red dotted curve)
#        polarization versus magnetic field at T = 1 K, V = 0 meV, and
#        finite Zeeman fields in Fig. 13."
#   The two disagree.  V below selects which is used; the sentence that
#   describes the plotted curves directly is taken as the primary reading,
#   with the caption's alternative one edit away.
#
# NO FITTED PARAMETER
#   Ps and Pv are RATIOS of Eq. (28) contributions, so its unknown
#   prefactor A cancels exactly.
#
# LAYOUT measured off the published figure (PDF page 11):
#   x 1..30 , ticks 5,10,15,20,25,30
#   y -1..1 , ticks -1.0,-0.5,0.0,0.5,1.0
#   black SOLID  = p_s , red DASHED = p_v
#   a flat BLUE line at zero: with Mz = Mv = 0 the four branch
#   conductivities are equal in pairs, both numerators vanish, and the two
#   polarisations are identically zero.
#   legend: "p_s" top-left, "p_v" bottom-left.
# =============================================================================
import sys

import numpy as np
import matplotlib.pyplot as plt

import paper_style as ps
import sigma_xx_common as sx

ps.apply()

V = 0.0            # see "WHICH V?" above; 0.015 follows the caption instead


if __name__ == "__main__":
    draft = "fast" in sys.argv
    npts = 300 if draft else 1400
    if draft:
        print("DRAFT resolution - omit 'fast' for the full render")
    print("Fig. 13 - spin and valley polarisation, Eqs. (29) and (30)")

    B = np.linspace(1.0, 30.0, npts)
    Ps = np.empty(npts)
    Pv = np.empty(npts)
    for i, b in enumerate(B):
        if i % 150 == 0:
            print(f"    B {i}/{npts}  ({b:.2f} T)")
        Ps[i], Pv[i] = sx.eq29_eq30_polarisations(b, V, zeeman=True)

    fig = plt.figure(figsize=(6.6, 4.6))
    ax = fig.add_axes([0.145, 0.155, 0.835, 0.825])

    # Mz = Mv = 0 gives Ps = Pv = 0 identically - the published blue line
    ax.axhline(0.0, color="blue", lw=1.2)
    ax.plot(B, Ps, color="black", lw=0.8)
    ax.plot(B, Pv, color="red", lw=0.8, linestyle="--", dashes=(4.5, 2.2))

    ax.set_xlim(1, 30)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels([r"$-1.0$", r"$-0.5$", "0.0", "0.5", "1.0"])
    ax.set_xlabel(r"B (T)", fontsize=15, labelpad=2)
    ax.set_ylabel(r"$p_s$ , $p_v$", fontsize=15)
    ps.frame(ax, labelsize=13)

    ps.legend_entry(ax, 0.020, 0.955, "-", "black", r"$\mathrm{p_s}$",
                    fontsize=12, sample=0.085, gap=0.028)
    ps.legend_entry(ax, 0.020, 0.045, "--", "red", r"$\mathrm{p_v}$",
                    fontsize=12, sample=0.085, gap=0.028)

    print(f"  Ps spans {Ps.min():.2f}..{Ps.max():.2f}   "
          f"Pv spans {Pv.min():.2f}..{Pv.max():.2f}")
    ps.save(fig, "bilayer_MoS2_fig13.png")
    plt.show()
