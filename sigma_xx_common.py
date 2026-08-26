"""
sigma_xx_common.py
==================

Shared machinery for Figures 12 and 13 of Zubair et al.,
Phys. Rev. B 96, 045405 (2017).

Figure 12 plots Eq. (28) itself (the collisional / longitudinal
conductivity).  Figure 13 plots Eqs. (29) and (30), the spin and valley
polarisations, which the paper says are "extracted from Eq. (28)" - they
are ratios of the SAME sum split by valley and spin.  So both figures need
the per-branch contributions, and they live here.

Eq. (28), verbatim from p.10:

    sigma_xx = A Sum_{n,mu,s,tau} (rho^{s,tau}_{n,mu})^4
               [ (2n+1)(1 + (k^{s,tau}_{n,mu})^2)^2
                 + (2n-1) n^2 / eps^4_{n,d2}
                 + (2n+3)(n+1)^2 (k^{s,tau}_{n,mu})^4 / eps^4_{n,d4} ]
               f(E^{s,tau}_{n,mu}) [1 - f(E^{s,tau}_{n,mu})]

    "where A = (e^2/h)(beta N_I |U_0|^2 / pi l_B^2 k_s^2) and Gamma is the
     level width."

Eq. (29) and Eq. (30), verbatim from p.11:

    Ps = [ (sigma^{K,up}_xx + sigma^{K',down}_xx)
         - (sigma^{K,down}_xx + sigma^{K',up}_xx) ]
       / [ (sigma^{K,up}_xx + sigma^{K',down}_xx)
         + (sigma^{K,down}_xx + sigma^{K',up}_xx) ]

    Pv = [ (sigma^{K,up}_xx + sigma^{K,down}_xx)
         - (sigma^{K',up}_xx + sigma^{K',down}_xx) ]
       / [ (sigma^{K,up}_xx + sigma^{K,down}_xx)
         + (sigma^{K',up}_xx + sigma^{K',down}_xx) ]

THE PREFACTOR NEEDS NO FITTING
    A contains the impurity density N_I, the screened potential strength
    U_0 and the screening wavevector k_s, none of which the paper gives
    numerically.  It never has to be known:
      * Fig. 12's published y axis is labelled "sigma_xx (A x 10^5)", i.e.
        the figure is plotted IN UNITS OF A, so the unknowns cancel.
      * Figs. 13's Ps and Pv are RATIOS of Eq. (28) contributions, so A
        cancels there exactly.

    One factor is carried explicitly: beta = 1/k_B T.  The paper's own text
    (p.11) works with the combination
        "beta f(E^{s,tau}_{n,mu}) [1 - f(E^{s,tau}_{n,mu})]
         approx delta(E_F - E^{s,tau}_{n,mu})"
    so beta multiplies the Fermi factors in the sum.  Including it puts
    Fig. 12 on the published scale with no free parameter:

        window        this code    published
        B 3.0-3.5       9.5          ~9
        B 6.0-6.5       2.6          ~2-3
        B 20-21         0.81         ~0.8
        B 35-36         0.58         ~0.6

    An earlier version of this project instead fitted an overall constant
    (A_SCALE = 5200) to the peak height.  That is no longer needed.
"""

import numpy as np
from scipy.optimize import brentq

import paper_equations as pe

P = pe.P
BETA = 1.0 / (pe.K_B * P.T)        # 1/k_B T, see the prefactor note above


def nmax_for(B):
    """More Landau levels are occupied at low field, so the cutoff grows."""
    return 150 if B < 4.0 else 60


def branch(B, V, s, tau, zeeman, nmax):
    """Conduction levels of one (s, tau) branch with everything Eq. (28)
    needs: the energy, the index n, and the coefficients rho and k.

    Returns (rows, hbar_omega_c) where each row is
        (energy_eV, n, rho, k, d2, d4)
    and n = -1 marks the Eq. (8) level, which Appendix B handles separately.
    """
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


def all_branches(B, V, zeeman):
    """Build every (s, tau) branch once, and the Fermi energy they share.

    The spectrum is built ONCE per field point and reused for both the
    Fermi energy and the conductivity sums; building it separately for each
    was what made earlier scans of these figures so slow.
    """
    nmax = nmax_for(B)
    branches = {}
    hw = None
    for tau in (+1, -1):
        for s in (+1, -1):
            branches[(s, tau)], hw = branch(B, V, s, tau, zeeman, nmax)

    E = np.array([r[0] for br in branches.values() for r in br])
    D0 = 2.0 * np.pi * pe.HBAR_J / (pe.E_CHARGE * B)
    filling = P.N_E * D0
    kT = pe.K_B * P.T

    def occ(ef):
        x = np.clip((E - ef) / kT, -500, 500)
        return np.sum(1.0 / (1.0 + np.exp(x))) - filling

    EF = brentq(occ, E.min() - 0.05, E.max() + 0.05, xtol=1e-12)
    return branches, hw, EF


def branch_sum(rows, hw, EF):
    """Eq. (28)'s sum for ONE (s, tau) branch, in units of A."""
    kT = pe.K_B * P.T
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
    return BETA * total


def sigma_xx(B, V, zeeman):
    """Eq. (28) summed over every branch, in units of A x 10^5 - which is
    exactly what the published Fig. 12 y axis plots."""
    branches, hw, EF = all_branches(B, V, zeeman)
    return sum(branch_sum(rows, hw, EF)
               for rows in branches.values()) / 1e5


def eq29_eq30_polarisations(B, V, zeeman=True):
    """Eqs. (29) and (30): the spin and valley polarisations.

    Both are ratios of the same four branch sums, so the unknown prefactor
    A cancels exactly and nothing needs to be fitted.

    With the Zeeman terms switched off the four branches are equal in
    pairs, so both numerators vanish identically and Ps = Pv = 0 - which is
    the flat blue line the published figure draws.
    """
    branches, hw, EF = all_branches(B, V, zeeman)
    s_up_K = branch_sum(branches[(+1, +1)], hw, EF)
    s_dn_K = branch_sum(branches[(-1, +1)], hw, EF)
    s_up_Kp = branch_sum(branches[(+1, -1)], hw, EF)
    s_dn_Kp = branch_sum(branches[(-1, -1)], hw, EF)

    ps_num = (s_up_K + s_dn_Kp) - (s_dn_K + s_up_Kp)
    ps_den = (s_up_K + s_dn_Kp) + (s_dn_K + s_up_Kp)
    pv_num = (s_up_K + s_dn_K) - (s_up_Kp + s_dn_Kp)
    pv_den = (s_up_K + s_dn_K) + (s_up_Kp + s_dn_Kp)

    ps = ps_num / ps_den if abs(ps_den) > 1e-300 else 0.0
    pv = pv_num / pv_den if abs(pv_den) > 1e-300 else 0.0
    return ps, pv
