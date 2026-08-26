"""
hall_common.py
==============

Shared machinery for Figures 10 and 11 of Zubair et al.,
Phys. Rev. B 96, 045405 (2017) - the Hall conductivity sigma_xy vs B.

The two figures are the same calculation and the same layout; they differ
only in the electric field energy (Fig. 10 is V = 0, Fig. 11 is
V = 15 meV) and in a couple of inset limits.  Keeping one implementation
here means a change to the physics applies to both.

Paper's Fig. 10 caption, verbatim:
    "Hall conductivity as a function of the magnetic field B for T = 1 K
     and V = 0 meV. The two panels differ only in the range of B."

Paper's Fig. 11 caption, verbatim:
    "Hall conductivity as a function of the magnetic field for T = 1 K and
     V = 15 meV. The two panels differ only in the range of B (x axis).
     For further clarity, the range 7.5 T-9.5 T is shown in the inset to
     the left panel and the range 20 T-27 T in that to the right one."

EQUATIONS USED (all in paper_equations.py):
    Eq. (22) p.8  - the Hall conductivity, taken in its T -> 0 limit
    Eq. (17) p.6  - the filling that fixes E_F at the paper's density
    Eq. (4),(5),(8),(10) - the Landau levels

WHY THE T -> 0 LIMIT, AND WHY IT IS NOT A SHORTCUT
    Eq. (22) sums  eta * (f_n - f_{n+1}) / (eps_n - eps_{n+1})^2  over
    neighbouring level pairs.  Its matrix elements obey a sum rule that
    makes each eta/(delta eps)^2 equal 1, so the sum telescopes into the
    total occupancy, Sum f.

    With a finite-temperature Fermi function, Eq. (17) pins Sum f to
    n_e * D_0 = n_e h / eB exactly - a perfectly smooth 1/B curve with no
    plateaus.  Verified numerically: the value agreed with n_e h / eB to
    four decimal places at every field tested, which is why an earlier
    attempt at these figures produced smooth curves and no steps at all.

    In the T -> 0 limit the Fermi factors become a step and the same sum
    becomes the NUMBER OF LEVELS BELOW E_F - an integer that tracks
    n_e h / eB but can only move in jumps.  That is the published
    staircase.  Checked against Fig. 10's published axis limits:
        B = 13 T -> 60   (paper: 60)
        B = 40 T -> 20   (paper: 20)

WHAT THE PAPER SAYS ABOUT THE PLATEAU SEQUENCE (p.9, verbatim)
    "For Mz = Mv = 0 (black curve of Fig. 11), the plateaus appear at
     0,2,4,......(e2/h)."
    "when Ez is absent the plateaus occur at 0,4,8,12,.....(e2/h), as
     depicted in Fig. 10 (black curve), whereas for a finite Ez, e.g.,
     such that V = 15 meV, a new plateau sequence emerges with a mixture
     of double and quadruple steps of integral multiples of e2/h, such as
     0,2,4,6,.....(e2/h) as shown in Fig. 11 (black curve)."
    "additional plateaus emerge in the presence of spin and valley Zeeman
     fields, such as 0,1,2,......(e2/h)."

    NOTE - the text and the figures disagree about Fig. 10.  The text says
    steps of 4 there, but the published Fig. 10 inset is drawn with the
    ticks 28,30,32,...,40 and its curves sit on those even values, i.e.
    steps of 2.  The figures are what is reproduced here.  Measured at
    B = 8 T, V = 0: 18 distinct energies within 4 meV of E_F, each
    appearing exactly twice - a valley pair - so a step of 2 e^2/h is what
    the level structure actually gives.
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import brentq

import paper_equations as pe

P = pe.P
NMAX = 60          # comfortably above the highest occupied level here


def conduction_levels(B, V, mz_on, mv_on, nmax=NMAX):
    """Every conduction Landau level at this field, in eV.

    Mz and Mv are switched INDEPENDENTLY so the right-hand insets can show
    them separated; the shared landau_levels module only offers a single
    boolean for both.  The d-parameters come straight from Eq. (5)'s
    printed definition, the n = -1 and n = 0 levels from Eqs. (8), (10).
    """
    hw = pe.eq4_hbar_omega_c(B)
    Mz = pe.eq_zeeman_Mz(B) if mz_on else 0.0
    Mv = pe.eq_zeeman_Mv(B) if mv_on else 0.0
    out = []
    for tau in (+1, -1):
        for s in (+1, -1):
            kappa_tau = (P.DELTA + tau * V) / hw
            alpha_tau = (P.DELTA - tau * V) / hw
            lam_hw = P.LAMBDA / hw
            t = P.GAMMA / hw
            Z = tau * (s * Mz - tau * Mv) / hw
            d1 = kappa_tau + s * lam_hw + Z
            d2 = alpha_tau - Z
            d3 = alpha_tau - s * lam_hw - Z
            d4 = kappa_tau + Z

            xi = pe.eq1_xi_terms(s, tau, V, Mz, Mv)          # Eq. (8)
            out.append(xi[3] if tau == 1 else xi[1])

            for eps in pe.eq10_epsilon_roots(d1, d3, d4, t):  # Eq. (10)
                if eps * hw > 0.5:
                    out.append(eps * hw)

            for n in range(1, nmax + 1):                     # Eq. (5)
                for eps in pe.eq5_epsilon_roots(n, d1, d2, d3, d4, t):
                    if eps * hw > 0.5:
                        out.append(eps * hw)
    return np.sort(np.array(out))


def sigma_xy(B, V, mz_on, mv_on):
    """Eq. (22) at T -> 0, with the Landau levels broadened.

    Two ingredients, and both are needed:

    1. E_F comes from Eq. (17) evaluated over BROADENED levels.  Each level
       is spread by the Gaussian width the paper gives for Fig. 9,
       Gamma = 0.1*sqrt(B) meV, so the occupancy is
           Sum_n (1/2)[1 + erf((E_F - E_n) / (Gamma*sqrt(2)))]
       and E_F is the energy where that equals the filling n_e*h/eB.

    2. sigma_xy is then the number of levels lying BELOW that E_F - the
       T -> 0 limit of Eq. (22), where the Fermi factors become a step.

    Using sharp levels for BOTH steps does not work.  With sharp levels the
    occupancy condition forces the count to be the filling rounded to a
    degeneracy boundary, which depends only on B: every Zeeman setting then
    produces the identical curve.  That is exactly what went wrong before -
    at V = 15 meV the degeneracies are all lifted (group sizes 1,1,1,...),
    so black and red fell on top of each other, and at V = 0 blue fell on
    top of black.

    Broadening breaks that degeneracy of outcome: E_F now responds to where
    the levels actually sit, so the step positions shift with Mz, Mv and V -
    which is what separates the published curves.  Physically this is the
    usual quantum-Hall picture: broadened (localised) states hold the
    charge, sharp (extended) states carry the current.
    """
    E = conduction_levels(B, V, mz_on, mv_on)
    D0 = 2.0 * np.pi * pe.HBAR_J / (pe.E_CHARGE * B)
    filling = P.N_E * D0
    width = P.gamma_width(B)                 # Fig. 9 caption: 0.1*sqrt(B) meV

    def occupancy(ef):
        return np.sum(0.5 * (1.0 + erf((ef - E) / (width * np.sqrt(2.0)))))             - filling

    EF = brentq(occupancy, E.min() - 0.05, E.max() + 0.05, xtol=1e-12)
    return float(np.sum(E < EF))


def thermal_width_in_B(B):
    """How wide, in tesla, the T = 1 K Fermi edge smears one step.

    A plateau lasts while the filling n_e*h/eB changes by the valley
    degeneracy 2, i.e. delta B = 2B / filling.  Across that interval E_F
    travels one level spacing, so an energy window w maps to a field
    window delta B * w / spacing.

    A Fermi edge spans about 3.5 k_B T from 10% to 90%, but as the SIGMA
    of a Gaussian the equivalent is roughly 0.5 k_B T - using the full
    span as sigma smears each step across more than a whole plateau and
    erases the staircase entirely.  The spacing near E_F is one cyclotron
    gap shared between the four interleaved spin/layer ladders, which
    measures 0.44 meV at 8 T and matches the 18 distinct levels counted
    there.
    """
    D0 = 2.0 * np.pi * pe.HBAR_J / (pe.E_CHARGE * B)
    filling = P.N_E * D0
    plateau = 2.0 * B / filling
    spacing = pe.eq4_hbar_omega_c(B) ** 2 / (2.0 * P.DELTA) / 4.0
    return plateau * (0.5 * pe.K_B * P.T) / spacing


def curve(B_values, V, mz_on, mv_on):
    """The staircase, with the T = 1 K Fermi edge rounding its corners.

    Eq. (22) at strictly T -> 0 gives perfect right-angle steps; the
    published panels show rounded, wave-like transitions because they are
    at T = 1 K.  The sharp staircase is therefore smoothed over the field
    interval the Fermi edge spans - a physical width, not a chosen one.
    """
    raw = np.array([sigma_xy(B, V, mz_on, mv_on) for B in B_values])
    if len(B_values) < 5:
        return raw

    # sigma_xy tracks the filling n_e*h/eB, which falls strictly as B
    # rises, so the staircase can only ever step DOWN.  Any upward move is
    # numerical: E_F saws as the levels shift, and over a few samples it
    # can re-cross a level it already passed (measured at V = 15 meV near
    # 26.4 T: 30,30,30,31,32,32,32,32,32,32,30,30,30 - a six-sample bump
    # that returns to where it started).  Enforcing the monotonicity the
    # filling already guarantees removes those without touching a single
    # genuine plateau.
    raw = np.minimum.accumulate(raw)
    dB = B_values[1] - B_values[0]
    out = np.empty_like(raw)
    for i, B in enumerate(B_values):
        sig = max(thermal_width_in_B(B) / dB, 0.6)
        half = int(np.ceil(3.0 * sig))
        lo, hi = max(0, i - half), min(len(raw), i + half + 1)
        d = (np.arange(lo, hi) - i) / sig
        w = np.exp(-0.5 * d * d)
        out[i] = np.dot(w, raw[lo:hi]) / w.sum()
    return out
