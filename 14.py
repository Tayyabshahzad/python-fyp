# =============================================================================
# FIGURE 14  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 14 caption, verbatim:
#   "Longitudinal (black) and Hall (red) resistivities versus magnetic field
#    B at T = 1 K and finite spin and valley Zeeman fields. The upper panels
#    are for V = 0 meV and the lower ones for V = 15 meV. The left and right
#    panels differ only in the range of B and rho_0 = A^-1 x 10^-35."
#
# Paper's text, p.11, verbatim:
#   "we evaluate the magnetoresistivity rho_munu using the conductivity
#    tensor via the well-known relations rho_xx = sigma_xx/S and
#    rho_xy = sigma_xy/S, with S = sigma_xx sigma_yy - sigma_xy sigma_yx
#    approx n_e^2 e^2 / B^2 where n_e is the electron concentration."
#
# EQUATIONS USED:
#   Eq. (28) p.10 - sigma_xx           (via sigma_xx_common.py)
#   Eq. (22) p.8  - sigma_xy           (via hall_common.py)
#   Eq. (17) p.6  - E_F at fixed density
#   plus the relations above for rho_xx, rho_xy and S.
#
# THE rho_0 NORMALISATION IN THE CAPTION CANNOT SERVE BOTH CURVES
#   sigma_xy is a pure number times e^2/h - it carries NO factor of A.
#   sigma_xx from Eq. (28) is A times a dimensionless sum.  With
#   rho_0 = A^-1 x 10^-35 that makes
#       rho_xy / rho_0  proportional to  A       (linear)
#       rho_xx / rho_0  proportional to  A^2     (quadratic)
#   so one value of A cannot place both on the published axis.  Checked
#   numerically: fitting A so that rho_xy(B=2) equals the published 0.0022
#   gives A = 3.3e-40, and that same A predicts a rho_xx peak of 1.5e-34
#   where the figure shows about 0.005 - wrong by 32 orders of magnitude.
#
#   The SHAPES are fully determined by the physics and are not adjusted:
#     rho_xy = sigma_xy / S comes out exactly B/(n_e e) in the classical
#     limit, i.e. a straight line through the origin carrying the Hall
#     staircase as small steps - which is what the published red curve is.
#     rho_xx = sigma_xx / S carries Eq. (28)'s SdH peaks, growing with B
#     because B^2/S outpaces the decay of the peaks.
#
#   So each curve is placed on the published axis with its own constant,
#   both stated below and both anchored to a published value rather than
#   tuned by eye.  This is a limitation of the paper's stated rho_0, not a
#   free choice: no single constant exists that works for both.
#
# LAYOUT measured off the published figure (PDF page 11):
#   left  panels: B 2..15  , rho 0..0.020 , y ticks every 0.005
#   right panels: B 15..40 , rho 0..0.05  , y ticks every 0.01
#   red = rho_xy , black = rho_xx ; V label upper-left, legend mid-right.
#
# Run "python 14.py fast" for a quick draft at reduced resolution.
# =============================================================================
import sys

import numpy as np
import matplotlib.pyplot as plt

import paper_equations as pe
import paper_style as ps
import hall_common as hc
import sigma_xx_common as sx

ps.apply()
P = pe.P

E_CH = pe.E_CHARGE
H_PLANCK = 2.0 * np.pi * pe.HBAR_J

# Display constants - see the rho_0 note above.  Each is anchored to a
# value read off the published axis, not tuned by eye.
#   rho_xy is a straight line; the published slope is 0.019 at B = 15 T.
#   rho_xx is anchored so its left-panel peak reaches the published ~0.005.
RHO_XY_ANCHOR = (15.0, 0.019)
RHO_XX_ANCHOR = 0.005


def raw_rho(B, V):
    """rho_xx and rho_xy from the paper's relations, before display scaling.

    S = n_e^2 e^2 / B^2 is the paper's stated approximation.  Fig. 14 is at
    finite spin and valley Zeeman fields, per its caption.
    """
    S = P.N_E ** 2 * E_CH ** 2 / B ** 2
    sigma_xy = hc.sigma_xy(B, V, True, True) * E_CH ** 2 / H_PLANCK
    sigma_xx = sx.sigma_xx(B, V, True) * 1e5      # undo Fig. 12's 10^5
    return sigma_xx / S, sigma_xy / S


def curves(B_values, V):
    rxx = np.empty(len(B_values))
    rxy = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 100 == 0:
            print(f"      B {i}/{len(B_values)}  ({B:.2f} T)")
        rxx[i], rxy[i] = raw_rho(B, V)
    return rxx, rxy


def draw_panel(ax, V, B_values, ylim, yticks, xticks, fmt, scales):
    print(f"  panel V = {V*1000:.0f} meV, B = {B_values[0]:.0f}"
          f"..{B_values[-1]:.0f} T")
    rxx, rxy = curves(B_values, V)
    rxx = rxx * scales["xx"]
    rxy = rxy * scales["xy"]

    ax.plot(B_values, rxy, color="red", lw=0.8)
    ax.plot(B_values, rxx, color="black", lw=0.7)

    ax.set_xlim(B_values[0], B_values[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(t) for t in yticks])
    ax.set_xlabel(r"B (T)", fontsize=14, labelpad=2)
    ax.set_ylabel(r"$\rho_{xy}$, $\rho_{xx}$ ($\rho_0$)", fontsize=13)
    ps.frame(ax, labelsize=11.5)

    ax.text(0.075, 0.870, f"V = {V*1000:.0f} meV", transform=ax.transAxes,
            fontsize=12, va="center")
    ps.legend_entry(ax, 0.610, 0.560, "-", "red", r"$\rho_{xy}$",
                    fontsize=12, sample=0.100, gap=0.030)
    ps.legend_entry(ax, 0.610, 0.400, "-", "black", r"$\rho_{xx}$",
                    fontsize=12, sample=0.100, gap=0.030)
    print(f"    rho_xx {rxx.min():.4f}..{rxx.max():.4f}   "
          f"rho_xy {rxy.min():.4f}..{rxy.max():.4f}")


if __name__ == "__main__":
    draft = "fast" in sys.argv
    nl, nr = (200, 160) if draft else (900, 700)
    if draft:
        print("DRAFT resolution - omit 'fast' for the full render")
    print("Fig. 14 - resistivities from rho = sigma/S")

    # Fix the two display constants from the anchors, once, up front.
    Ba, target = RHO_XY_ANCHOR
    rxx_a, rxy_a = raw_rho(Ba, 0.0)
    scale_xy = target / rxy_a
    probe = np.linspace(3.0, 15.0, 90)
    peak = max(raw_rho(b, 0.0)[0] for b in probe)
    scale_xx = RHO_XX_ANCHOR / peak
    print(f"  display constants: rho_xy x {scale_xy:.4e},"
          f"  rho_xx x {scale_xx:.4e}")
    scales = {"xx": scale_xx, "xy": scale_xy}

    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32,
                          left=0.095, right=0.985, top=0.985, bottom=0.090)

    for row, V in enumerate((0.0, 0.015)):
        draw_panel(fig.add_subplot(gs[row, 0]), V,
                   np.linspace(2.0, 15.0, nl), (0, 0.020),
                   [0.000, 0.005, 0.010, 0.015, 0.020],
                   [2, 4, 6, 8, 10, 12, 14], "{:.3f}", scales)
        draw_panel(fig.add_subplot(gs[row, 1]), V,
                   np.linspace(15.0, 40.0, nr), (0, 0.05),
                   [0.00, 0.01, 0.02, 0.03, 0.04, 0.05],
                   [15, 20, 25, 30, 35, 40], "{:.2f}", scales)

    ps.save(fig, "bilayer_MoS2_fig14.png")
    plt.show()
