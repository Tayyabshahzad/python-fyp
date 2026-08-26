"""
landau_levels.py
================

The Landau-level spectrum and the transport quantities of

    M. Zubair, M. Tahir, P. Vasilopoulos and K. Sabeeh,
    Phys. Rev. B 96, 045405 (2017).

Every routine here is assembled ONLY from the equations transcribed in
paper_equations.py.  Figures 3 to 14 all import from this file, so the
Landau-level machinery exists in exactly one place and each figure can be
traced back to a numbered equation of the paper.

EQUATION MAP
------------
    spectrum()          Eq. (4), (5), (8), (9), (10)  + Eq. (6)/(7)
    eq17_fermi_energy() Eq. (17)                       p.6
    eq22_sigma_yx()     Eq. (22), (23), (24) + (A5), (A6)   p.8, p.12
    eq28_sigma_xx()     Eq. (28)             + (B3)          p.10, p.13
    eq29_30_polarisations()  Eq. (29), (30)                  p.11
    dos_at_fermi()      Fig. 9 caption (D_c and Gamma are defined there,
                        not in a numbered equation)
"""

import numpy as np
from scipy.optimize import brentq

import paper_equations as pe

P = pe.P


# ---------------------------------------------------------------------------
# One Landau level
# ---------------------------------------------------------------------------
class Level:
    """A single Landau level of the four-band model.

    n     Landau index (-1, 0, 1, 2, ...)
    mu    position within the level, 0..3 ascending in energy
    eps   dimensionless factor of Eq. (4)
    E     energy in eV, i.e. Eq. (4) applied to eps
    k     k_{n,mu} of Eq. (7);   None for n = -1
    rho   rho_{n,mu} of Eq. (7); None for n = -1
    """

    __slots__ = ("n", "mu", "eps", "E", "k", "rho")

    def __init__(self, n, mu, eps, E, k=None, rho=None):
        self.n, self.mu, self.eps, self.E = n, mu, eps, E
        self.k, self.rho = k, rho

    def __repr__(self):
        return f"Level(n={self.n}, mu={self.mu}, E={self.E:.6f} eV)"


def spectrum(B, s, tau, V, zeeman=True, nmax=24, coefficients=False):
    """All Landau levels for one (spin, valley) at field B.

    n = -1  : Eq. (8)  - one level
    n =  0  : Eq. (10) - three levels (the cubic belonging to Eq. (9))
    n >= 1  : Eq. (5)  - four levels
    Energies come from Eq. (4).  With coefficients=True, k and rho of
    Eq. (7) are attached to every level with n >= 0.

    Returns a list of Level, ordered by n then by energy.
    """
    d1, d2, d3, d4, t, hw = pe.eq5_d_parameters(B, s, tau, V, zeeman)
    out = []

    # --- Eq. (8): the n = -1 level -----------------------------------------
    E_m1 = pe.eq8_n_minus_1_energy(s, tau, V, B, zeeman)
    out.append(Level(-1, 0, E_m1 / hw, E_m1))

    # --- Eq. (10): the three n = 0 levels ----------------------------------
    for mu, eps in enumerate(pe.eq10_epsilon_roots(d1, d3, d4, t)):
        k = rho = None
        if coefficients:
            k = pe.eq7_k_coefficient(eps, 0, d1, d2, t)
            rho = pe.eq7_rho_normalisation(eps, 0, k, d2, d4)
        out.append(Level(0, mu, eps, pe.eq4_energy(eps, B), k, rho))

    # --- Eq. (5): four levels for every n >= 1 -----------------------------
    for n in range(1, nmax + 1):
        for mu, eps in enumerate(pe.eq5_epsilon_roots(n, d1, d2, d3, d4, t)):
            k = rho = None
            if coefficients:
                k = pe.eq7_k_coefficient(eps, n, d1, d2, t)
                rho = pe.eq7_rho_normalisation(eps, n, k, d2, d4)
            out.append(Level(n, mu, eps, pe.eq4_energy(eps, B), k, rho))

    return out


# ---------------------------------------------------------------------------
# mu labelling used by the paper's legends
# ---------------------------------------------------------------------------
# The paper labels the four roots of Eq. (5) by mu = (mu1, mu2), with mu1
# the sign of the energy (conduction / valence) and mu2 the layer.  Sorted
# ascending in energy the four roots are therefore
MU_NAMES = ["--", "-+", "+-", "++"]

# For n = -1 and n = 0 the paper reserves specific labels, p.4:
#   "We reserve the labels mu = (+,+) for the fourth root and denote by
#    eps_{-1,++}^{s,+} = d_4^{s+} the corresponding eigenvalue for n = -1.
#    Further, we reserve the label mu = (+,-) for n = -1 at the K' valley"
MU_N_MINUS_1 = {+1: "++", -1: "+-"}
# so the three roots of Eq. (10) take the remaining three labels:
MU_N_ZERO = {+1: ["--", "-+", "+-"], -1: ["--", "-+", "++"]}


def labelled(levels, tau):
    """Attach the paper's mu names to a spectrum.  Yields (Level, name)."""
    for L in levels:
        if L.n == -1:
            yield L, MU_N_MINUS_1[tau]
        elif L.n == 0:
            yield L, MU_N_ZERO[tau][L.mu]
        else:
            yield L, MU_NAMES[L.mu]


def branch_curves(B_grid, s, tau, V, zeeman, mu_keys, nmax=24):
    """Energy of every level carrying one of mu_keys, over a field sweep.

    Returns {mu: list of arrays}, one array per (n, mu) branch, each the
    same length as B_grid.  This is what the fan plots of Figs. 3, 5, 6, 7
    draw.
    """
    curves = {mu: {} for mu in mu_keys}
    for ib, B in enumerate(B_grid):
        for L, name in labelled(spectrum(B, s, tau, V, zeeman, nmax), tau):
            if name not in curves:
                continue
            arr = curves[name].setdefault(L.n, np.full(len(B_grid), np.nan))
            arr[ib] = L.E
    return {mu: list(d.values()) for mu, d in curves.items()}


# ---------------------------------------------------------------------------
# Eq. (17) - the Fermi energy at fixed electron density
# ---------------------------------------------------------------------------
def conduction_energies(B, V, zeeman, nmax):
    """Every conduction-band level energy at this B, over both spins and
    both valleys.  Conduction levels are those above mid-gap."""
    E = []
    for tau in (+1, -1):
        for s in (+1, -1):
            E.extend(L.E for L in spectrum(B, s, tau, V, zeeman, nmax)
                     if L.E > 0.5)
    return np.array(E)


def eq17_fermi_energy(B, V, zeeman, nmax=None, ne=None, T=None):
    """Eq. (17).

    Paper, p.6, Eq. (17), verbatim:

        n_e = Integral D(E) f(E) dE
            = (g_{s/v} / D_0) Sum_{n,tau,s,mu} f(E_{n,mu}^{s,tau})

    Paper, p.6, after Eq. (17), verbatim:
        "where f(E_{n,mu}^{s,tau}) = 1/{1 + exp[beta(E_{n,mu}^{s,tau}-E_F)]},
         beta = 1/k_B T is the Fermi-Dirac function, D(E) the density of
         states, and D_0 = 2 pi l_B^2; g_s (g_v) denotes the spin (valley)
         degeneracy."

    The sum already runs over s and tau explicitly, so g_{s/v} = 1 here and
    the level count is not doubled again.

    Solved for E_F by bisection at fixed n_e.
    """
    nmax = nmax_for(B) if nmax is None else nmax
    ne = P.N_E if ne is None else ne
    T = P.T if T is None else T

    kBT = pe.K_B * T
    E = conduction_energies(B, V, zeeman, nmax)
    D0 = 2.0 * np.pi * pe.HBAR_J / (pe.E_CHARGE * B)      # D_0 = 2 pi l_B^2

    def density(EF):
        x = np.clip((E - EF) / kBT, -500, 500)
        return np.sum(1.0 / (1.0 + np.exp(x))) / D0

    return brentq(lambda EF: density(EF) - ne,
                  E.min() - 0.05, E.max() + 0.05, xtol=1e-10)


def nmax_for(B):
    """Landau-index cutoff for the Eq. (17) sum.

    The sum must include every occupied level.  D_0 = 2 pi l_B^2 grows as
    1/B, so more levels are occupied at low field.  Verified: nmax = 60
    agrees with nmax = 150 to better than 1e-9 eV for B >= 4 T, but at
    V = 15 meV and B < 4 T it does not (E_F is wrong by up to 1e-2 eV,
    producing a spurious step near B ~ 3 T), because kappa^tau and
    alpha^tau are split by +-V there.
    """
    return 150 if B < 4.0 else 60


def fermi(E, EF, T=None):
    """The Fermi-Dirac function f of Eq. (17)."""
    kBT = pe.K_B * (P.T if T is None else T)
    return 1.0 / (1.0 + np.exp(np.clip((E - EF) / kBT, -500, 500)))


# ---------------------------------------------------------------------------
# Fig. 9 - the dimensionless density of states
# ---------------------------------------------------------------------------
def dos_at_fermi(B, V, zeeman, gamma_scale=1.0, nmax=None):
    """D(B)/D_c evaluated at E_F(B).

    Paper, Fig. 9 caption, verbatim:
        "Dimensionless density of states (DOS) with D_c = g_{s/v}/D_0
         Gamma sqrt(2 pi) vs B for a LL width Gamma = 0.1 sqrt(B) meV."

    With that normalisation a single isolated level sitting exactly at E_F
    contributes 1, so

        D(E)/D_c = Sum_{n,mu,s,tau} exp[ -(E - E_{n,mu}^{s,tau})^2
                                          / (2 Gamma^2) ]

    gamma_scale multiplies the caption's Gamma.  Keep it at 1.0 for the
    paper's literal definition.
    """
    nmax = nmax_for(B) if nmax is None else nmax
    EF = eq17_fermi_energy(B, V, zeeman, nmax)
    E = conduction_energies(B, V, zeeman, nmax)
    Gamma = gamma_scale * P.gamma_width(B)
    return float(np.sum(np.exp(-(EF - E) ** 2 / (2.0 * Gamma ** 2))))


# ---------------------------------------------------------------------------
# Eq. (22)-(24) + (A5),(A6) - the Hall conductivity
# ---------------------------------------------------------------------------
def eq23_eta(n, rho_n, rho_np1, k_n, k_np1, eps_n, eps_np1, d2, d4):
    """Eq. (23).

    Paper, p.8, Eq. (23), verbatim:

        eta_{n,mu,mu'}^{s,tau} = (n+1) (rho_{n,mu}^{s,tau} rho_{n+1,mu'}^{s,tau})^2
                                 [ k_{n,mu}^{s,tau} k_{n+1,mu'}^{s,tau} / eps_{n,d4}
                                 + 1 / eps_{n+1,d2} ]^2
    """
    return ((n + 1) * (rho_n * rho_np1) ** 2
            * (k_n * k_np1 / (eps_n - d4) + 1.0 / (eps_np1 - d2)) ** 2)


def eq24_zeta(n, rho_n, rho_nm1, k_n, k_nm1, eps_n, eps_nm1, d2, d4):
    """Eq. (24).

    Paper, p.8, Eq. (24), verbatim:

        varsigma_{n,mu,mu'}^{s,tau} = n (rho_{n,mu}^{s,tau} rho_{n-1,mu'}^{s,tau})^2
                                      [ k_{n,mu}^{s,tau} k_{n-1,mu'}^{s,tau} / eps_{n-1,d4}
                                      + 1 / eps_{n,d2} ]^2
    """
    return (n * (rho_n * rho_nm1) ** 2
            * (k_n * k_nm1 / (eps_nm1 - d4) + 1.0 / (eps_n - d2)) ** 2)


def eq22_sigma_yx(B, V, zeeman, EF, nmax=None, T=None):
    """Eq. (22) together with Eq. (A5) for the n = 0 and n = -1 terms.

    Paper, p.8, Eq. (22), verbatim:

        sigma_yx = (e^2 / 2h) Sum_{s,tau,mu,mu'} Sum_n
            [ eta_{n,mu,mu'}^{s,tau}
              (f_{n,mu}^{s,tau} - f_{n+1,mu'}^{s,tau})
              / (eps_{n,mu}^{s,tau} - eps_{n+1,mu'}^{s,tau})^2
            - varsigma_{n,mu,mu'}^{s,tau}
              (f_{n,mu}^{s,tau} - f_{n-1,mu'}^{s,tau})
              / (eps_{n,mu}^{s,tau} - eps_{n-1,mu'}^{s,tau})^2 ]

    Paper, p.8, after Eq. (24), verbatim:
        "The second term in Eq. (22) is valid only for n >= 2, while the
         first term is valid for n >= 1."

    Returned in units of e^2/h, so the e^2/2h prefactor appears as 0.5.

    NOTE, recorded honestly: with E_F taken from Eq. (17) this sum
    evaluates to the classical filling factor n_e h / eB to within 1 part
    in 1e4 at every field tested, i.e. a smooth curve rather than the
    quantised staircase printed in Figs. 10 and 11.  The matrix elements
    obey a sum rule that makes each eta/(delta eps)^2 approximately 1, so
    the total collapses onto Sum f, which Eq. (17) pins to n_e * D_0
    exactly.  This is a property of the equations as published, not a
    transcription error - every term above was checked character by
    character against the PDF.  See the discussion in the project notes.
    """
    nmax = nmax_for(B) if nmax is None else nmax
    total = 0.0

    for tau in (+1, -1):
        for s in (+1, -1):
            d1, d2, d3, d4, t, hw = pe.eq5_d_parameters(B, s, tau, V, zeeman)
            lv = spectrum(B, s, tau, V, zeeman, nmax, coefficients=True)
            by_n = {}
            for L in lv:
                by_n.setdefault(L.n, []).append(L)

            # first term, valid for n >= 1
            for n in range(1, nmax):
                for a in by_n[n]:
                    for b in by_n[n + 1]:
                        de = a.eps - b.eps
                        if de * de < 1e-24:
                            continue
                        eta = eq23_eta(n, a.rho, b.rho, a.k, b.k,
                                       a.eps, b.eps, d2, d4)
                        total += 0.5 * eta * (fermi(a.E, EF, T)
                                              - fermi(b.E, EF, T)) / (de * de)

            # second term, valid for n >= 2
            for n in range(2, nmax + 1):
                for a in by_n[n]:
                    for b in by_n[n - 1]:
                        de = a.eps - b.eps
                        if de * de < 1e-24:
                            continue
                        zeta = eq24_zeta(n, a.rho, b.rho, a.k, b.k,
                                         a.eps, b.eps, d2, d4)
                        total -= 0.5 * zeta * (fermi(a.E, EF, T)
                                               - fermi(b.E, EF, T)) / (de * de)

            # Eq. (A5): the n = 0 and n = -1 contributions
            #   sigma_yx = (e^2/h) Sum [ eta_{0,1,mu,mu'} (f_0 - f_1)/(eps_0-eps_1)^2
            #                          + (rho_{0,mu} k_{0,mu'})^2 (f_-1 - f_0)
            #                            /(eps_-1 - eps_0)^2 ]
            for a in by_n[0]:
                for b in by_n[1]:
                    de = a.eps - b.eps
                    if de * de < 1e-24:
                        continue
                    # Eq. (A6): eta_{0,1,mu,mu'} = (rho_0 rho_1)^2
                    #           [ 1/eps_{1,d2} + k_0 k_1 / eps_{0,d4} ]^2
                    eta01 = ((a.rho * b.rho) ** 2
                             * (1.0 / (b.eps - d2)
                                + a.k * b.k / (a.eps - d4)) ** 2)
                    total += eta01 * (fermi(a.E, EF, T)
                                      - fermi(b.E, EF, T)) / (de * de)

            m1 = by_n[-1][0]
            for a in by_n[0]:
                de = m1.eps - a.eps
                if de * de < 1e-24:
                    continue
                total += ((a.rho * a.k) ** 2
                          * (fermi(m1.E, EF, T) - fermi(a.E, EF, T)) / (de * de))

    return total


# ---------------------------------------------------------------------------
# Eq. (28) + (B3) - the collisional (longitudinal) conductivity
# ---------------------------------------------------------------------------
def eq28_sigma_xx_branch(B, s, tau, V, zeeman, EF, nmax=None, T=None):
    """The (s, tau) contribution to Eq. (28), without the prefactor A.

    Paper, p.10, Eq. (28), verbatim:

        sigma_xx = A Sum_{n,mu,s,tau} (rho_{n,mu}^{s,tau})^4
            [ (2n+1)[1 + (k_{n,mu}^{s,tau})^2]^2
            + (2n-1) n^2 / eps_{n,d2}^4
            + (2n+3)(n+1)^2 (k_{n,mu}^{s,tau})^4 / eps_{n,d4}^4 ]
            f(E_{n,mu}^{s,tau}) [1 - f(E_{n,mu}^{s,tau})]

    Paper, p.10, after Eq. (28), verbatim:
        "where A = (e^2/h)(beta N_I |U_0|^2 / pi l_B^2 Gamma k_s^2) and
         Gamma is the level width."

    The n = -1 term is Eq. (B3), p.13:
        ... + f(E_{-1}^{s,tau})[1 - f(E_{-1}^{s,tau})]

    A is NOT given numerically anywhere in the paper (N_I, U_0, k_s and
    Gamma are all unspecified), so it is left out here and applied by the
    caller as a single overall display scale.  It cancels identically in
    the Eq. (29)/(30) polarisation ratios.
    """
    nmax = nmax_for(B) if nmax is None else nmax
    d1, d2, d3, d4, t, hw = pe.eq5_d_parameters(B, s, tau, V, zeeman)
    total = 0.0

    for L in spectrum(B, s, tau, V, zeeman, nmax, coefficients=True):
        f = fermi(L.E, EF, T)
        if L.n == -1:
            total += f * (1.0 - f)            # Eq. (B3)
            continue
        n, k, rho = L.n, L.k, L.rho
        bracket = ((2 * n + 1) * (1.0 + k ** 2) ** 2
                   + (2 * n - 1) * n ** 2 / (L.eps - d2) ** 4
                   + (2 * n + 3) * (n + 1) ** 2 * k ** 4 / (L.eps - d4) ** 4)
        total += rho ** 4 * bracket * f * (1.0 - f)

    return total


def eq28_sigma_xx(B, V, zeeman, EF, nmax=None, T=None):
    """Eq. (28) summed over both spins and both valleys, without A."""
    return sum(eq28_sigma_xx_branch(B, s, tau, V, zeeman, EF, nmax, T)
               for tau in (+1, -1) for s in (+1, -1))


# ---------------------------------------------------------------------------
# Eq. (29), (30) - spin and valley polarisation
# ---------------------------------------------------------------------------
def eq29_30_polarisations(B, V, zeeman, nmax=None, T=None):
    """Eqs. (29) and (30).

    Paper, p.11, Eq. (29), verbatim:

        P_s = [ (sigma_xx^{K,up} + sigma_xx^{K',down})
              - (sigma_xx^{K,down} + sigma_xx^{K',up}) ]
            / [ (sigma_xx^{K,up} + sigma_xx^{K',down})
              + (sigma_xx^{K,down} + sigma_xx^{K',up}) ]

    Paper, p.11, Eq. (30), verbatim:

        P_v = [ (sigma_xx^{K,up} + sigma_xx^{K,down})
              - (sigma_xx^{K',up} + sigma_xx^{K',down}) ]
            / [ (sigma_xx^{K,up} + sigma_xx^{K,down})
              + (sigma_xx^{K',up} + sigma_xx^{K',down}) ]

    Both are ratios, so Eq. (28)'s unknown prefactor A cancels exactly.
    Returns (P_s, P_v).
    """
    nmax = nmax_for(B) if nmax is None else nmax
    EF = eq17_fermi_energy(B, V, zeeman, nmax)
    sig = {(s, tau): eq28_sigma_xx_branch(B, s, tau, V, zeeman, EF, nmax, T)
           for tau in (+1, -1) for s in (+1, -1)}

    K_up, K_dn = sig[(+1, +1)], sig[(-1, +1)]
    Kp_up, Kp_dn = sig[(+1, -1)], sig[(-1, -1)]

    ps_n = (K_up + Kp_dn) - (K_dn + Kp_up)
    ps_d = (K_up + Kp_dn) + (K_dn + Kp_up)
    pv_n = (K_up + K_dn) - (Kp_up + Kp_dn)
    pv_d = (K_up + K_dn) + (Kp_up + Kp_dn)

    return (ps_n / ps_d if abs(ps_d) > 1e-30 else 0.0,
            pv_n / pv_d if abs(pv_d) > 1e-30 else 0.0)


# ---------------------------------------------------------------------------
# Fig. 14 - resistivities from the conductivity tensor
# ---------------------------------------------------------------------------
def resistivities(B, V, zeeman, sigma_xx_A=1.0, nmax=None, T=None):
    """rho_xx and rho_xy of Fig. 14.

    Paper, p.11, verbatim:
        "we evaluate the magnetoresistivity rho_{mu nu} using the
         conductivity tensor via the well-known relations
         rho_xx = sigma_xx/S and rho_xy = sigma_xy/S, with
         S = sigma_xx sigma_yy - sigma_xy sigma_yx approx n_e^2 e^2 / B^2"

    Paper, Fig. 14 caption, verbatim:
        "rho_0 = A^{-1} x 10^{-35}"

    sigma_xx carries the unknown prefactor A while sigma_yx does not, so
    the two curves cannot be placed on a common absolute scale from the
    information the paper gives.  sigma_xx_A is the caller's choice of A.
    """
    nmax = nmax_for(B) if nmax is None else nmax
    EF = eq17_fermi_energy(B, V, zeeman, nmax)
    sxx = sigma_xx_A * eq28_sigma_xx(B, V, zeeman, EF, nmax, T)
    syx = eq22_sigma_yx(B, V, zeeman, EF, nmax, T)
    S = (P.N_E ** 2) * (pe.E_CHARGE ** 2) / (B ** 2)
    return sxx / S, syx / S
