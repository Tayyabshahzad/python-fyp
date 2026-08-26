# =============================================================================
# FIGURE 12  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 12 caption, verbatim:
#   "Longitudinal conductivity versus magnetic field B at T = 1 K. The upper
#    (lower) panels are for V = 0 meV (V = 15 meV). The left and right
#    panels differ only in the range of B."
#
# EQUATIONS USED (all in paper_equations.py):
#   Eq. (28) p.10 - the collisional (longitudinal) conductivity
#   Eq. (17) p.6  - E_F at the paper's fixed electron density
#   Eq. (4),(5),(8),(10) - the Landau levels
#   Eq. (6),(7)   - the eigenvector coefficients rho and k that Eq. (28) uses
#
# Eq. (28), verbatim from p.10:
#
#   sigma_xx = A Sum_{n,mu,s,tau} (rho^{s,tau}_{n,mu})^4
#              [ (2n+1)(1 + (k^{s,tau}_{n,mu})^2)^2
#                + (2n-1) n^2 / eps^4_{n,d2}
#                + (2n+3)(n+1)^2 (k^{s,tau}_{n,mu})^4 / eps^4_{n,d4} ]
#              f(E^{s,tau}_{n,mu}) [1 - f(E^{s,tau}_{n,mu})]
#
#   "where A = (e^2/h)(beta N_I |U_0|^2 / pi l_B^2 k_s^2) and Gamma is the
#    level width."
#
# THE PREFACTOR NEEDS NO FITTING
#   A contains the impurity density N_I, the screened potential strength
#   U_0 and the screening wavevector k_s, none of which the paper gives
#   numerically.  That looks like a dead end, but the published y axis is
#   labelled "sigma_xx (A x 10^5)" - the figure is plotted IN UNITS OF A,
#   so the unknown constants cancel and nothing has to be fitted.
#
#   One factor does have to be carried explicitly: beta = 1/k_B T.  The
#   paper's own text (p.11) works with the combination
#       "beta f(E^{s,tau}_{n,mu}) [1 - f(E^{s,tau}_{n,mu})]
#        approx delta(E_F - E^{s,tau}_{n,mu})"
#   so beta multiplies the Fermi factors in the sum.  Including it puts
#   every window on the published scale, with no free parameter:
#
#       window        this code    published
#       B 3.0-3.5       9.5          ~9
#       B 6.0-6.5       2.6          ~2-3
#       B 20-21         0.81         ~0.8
#       B 35-36         0.58         ~0.6
#
#   An earlier version of this project instead fitted an overall constant
#   (A_SCALE = 5200) to the peak height.  That is no longer needed.
#
# WHAT THE PAPER SAYS THE FIGURE SHOWS (p.10, verbatim)
#   "Fig. 12 shows a beating pattern of the SdH oscillations for B fields
#    up to 9 T when Ez is absent (V = 0) and for B fields up to 7 T when a
#    finite Ez is present (V = 15 meV). For high B fields the beating
#    pattern is absent and the longitudinal conductivity peaks are split."
#
# LAYOUT measured off the published figure (PDF page 10):
#   left  panels: B 3..15  , sigma 0..10  , y ticks every 2
#   right panels: B 15..40 , sigma 0..1.0 , y ticks every 0.2
#   legend top-right in every panel: the V label, then black, then red.
#
# Run "python 12.py fast" for a quick draft at reduced resolution.
# =============================================================================
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import paper_equations as pe
import paper_style as ps

ps.apply()
P = pe.P

BETA = 1.0 / (pe.K_B * P.T)        # 1/k_B T, see the prefactor note above


def nmax_for(B):
    """More Landau levels are occupied at low field, so the cutoff grows."""
    return 150 if B < 4.0 else 60


def branch(B, V, s, tau, zeeman, nmax):
    """Conduction levels of one (s, tau) branch with everything Eq. (28)
    needs: the energy, the index n, and the coefficients rho and k."""
    hw = pe.eq4_hbar_omega_c(B)
    Mz = pe.eq_zeeman_Mz(B) if zeeman else 0.0
    Mv = pe.eq_zeeman_Mv(B) if zeeman else 0.0
    kappa_tau = (P.DELTA + tau * V) / hw
    alpha_tau = (P.DELTA - tau * V) / hw
    lam_hw = P.LAMBDA / hw
    t = P.GAMMA / hw
    Z = tau * (s * Mz - tau * Mv) / hw
    d1 = kappa_tau + s * lam_hw + Z
    d2 = alpha_tau - Z
    d3 = alpha_tau - s * lam_hw - Z
    d4 = kappa_tau + Z

    rows = []
    xi = pe.eq1_xi_terms(s, tau, V, Mz, Mv)              # Eq. (8), n = -1
    rows.append((xi[3] if tau == 1 else xi[1], -1, 0.0, 0.0, d2, d4))

    for eps in pe.eq10_epsilon_roots(d1, d3, d4, t):     # Eq. (10), n = 0
        if eps * hw > 0.5:
            k_c = pe.eq7_k_coefficient(eps, 0, d1, d2, t)
            rho = pe.eq7_rho_normalisation(eps, 0, k_c, d2, d4)
            rows.append((eps * hw, 0, rho, k_c, d2, d4))

    for n in range(1, nmax + 1):                         # Eq. (5), n >= 1
        for eps in pe.eq5_epsilon_roots(n, d1, d2, d3, d4, t):
            if eps * hw > 0.5:
                k_c = pe.eq7_k_coefficient(eps, n, d1, d2, t)
                rho = pe.eq7_rho_normalisation(eps, n, k_c, d2, d4)
                rows.append((eps * hw, n, rho, k_c, d2, d4))
    return rows, hw


def sigma_xx(B, V, zeeman):
    """Eq. (28) in units of A, i.e. what the published y axis plots.

    The spectrum is built ONCE and reused for both the Fermi energy and
    the conductivity sum; building it twice per field point is what made
    earlier scans of this figure so slow.
    """
    nmax = nmax_for(B)
    rows = []
    hw = None
    for tau in (+1, -1):
        for s in (+1, -1):
            r, hw = branch(B, V, s, tau, zeeman, nmax)
            rows.extend(r)

    E = np.array([r[0] for r in rows])
    D0 = 2.0 * np.pi * pe.HBAR_J / (pe.E_CHARGE * B)
    filling = P.N_E * D0
    kT = pe.K_B * P.T

    def occ(ef):
        x = np.clip((E - ef) / kT, -500, 500)
        return np.sum(1.0 / (1.0 + np.exp(x))) - filling

    EF = brentq(occ, E.min() - 0.05, E.max() + 0.05, xtol=1e-12)

    total = 0.0
    for energy, n, rho, k_c, d2, d4 in rows:
        x = np.clip((energy - EF) / kT, -500, 500)
        f = 1.0 / (1.0 + np.exp(x))
        weight = f * (1.0 - f)
        if weight < 1e-14:
            continue
        if n < 0:                       # Appendix B: the n = -1 term
            total += weight
            continue
        eps = energy / hw
        bracket = ((2 * n + 1) * (1.0 + k_c ** 2) ** 2
                   + (2 * n - 1) * n ** 2 / (eps - d2) ** 4
                   + (2 * n + 3) * (n + 1) ** 2 * k_c ** 4 / (eps - d4) ** 4)
        total += rho ** 4 * bracket * weight
    return BETA * total / 1e5          # units of A x 10^5, as published


def curve(B_values, V, zeeman):
    out = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 150 == 0:
            print(f"      B {i}/{len(B_values)}  ({B:.2f} T)")
        out[i] = sigma_xx(B, V, zeeman)
    return out


def draw_panel(ax, V, B_values, ylim, yticks, xticks, fmt):
    print(f"  panel V = {V*1000:.0f} meV, B = {B_values[0]:.0f}"
          f"..{B_values[-1]:.0f} T")
    print("    Mz = Mv = 0")
    off = curve(B_values, V, False)
    print("    Mz, Mv != 0")
    on = curve(B_values, V, True)

    ax.plot(B_values, off, color="black", lw=0.7)
    ax.plot(B_values, on, color="red", lw=0.7)

    ax.set_xlim(B_values[0], B_values[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(t) for t in yticks])
    ax.set_xlabel(r"B (T)", fontsize=14, labelpad=2)
    ax.set_ylabel(r"$\sigma_{xx}$ (A$\times 10^5$)", fontsize=13)
    ps.frame(ax, labelsize=11.5)

    ax.text(0.965, 0.930, f"V = {V*1000:.0f} meV", transform=ax.transAxes,
            fontsize=11.5, ha="right", va="center")
    ps.legend_entry(ax, 0.560, 0.835, "-", "black",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ = 0", fontsize=11,
                    sample=0.090, gap=0.030)
    ps.legend_entry(ax, 0.560, 0.745, "-", "red",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ $\neq$ 0", fontsize=11,
                    sample=0.090, gap=0.030)
    print(f"    black {off.min():.2f}..{off.max():.2f}   "
          f"red {on.min():.2f}..{on.max():.2f}")


if __name__ == "__main__":
    draft = "fast" in sys.argv
    nl, nr = (260, 200) if draft else (1100, 750)
    if draft:
        print("DRAFT resolution - omit 'fast' for the full render")

    print("Fig. 12 - longitudinal conductivity, Eq. (28) in units of A")
    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          left=0.085, right=0.985, top=0.985, bottom=0.090)

    for row, V in enumerate((0.0, 0.015)):
        draw_panel(fig.add_subplot(gs[row, 0]), V,
                   np.linspace(3.0, 15.0, nl), (0, 10),
                   [0, 2, 4, 6, 8, 10], [4, 6, 8, 10, 12, 14], "{:.0f}")
        draw_panel(fig.add_subplot(gs[row, 1]), V,
                   np.linspace(15.0, 40.0, nr), (0, 1.0),
                   [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   [15, 20, 25, 30, 35, 40], "{:.1f}")

    ps.save(fig, "bilayer_MoS2_fig12.png")
    plt.show()
