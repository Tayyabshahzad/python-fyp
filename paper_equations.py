"""
paper_equations.py
==================

EXACT transcription of the equations of

    M. Zubair, M. Tahir, P. Vasilopoulos and K. Sabeeh,
    "Quantum magnetotransport in bilayer MoS2: Influence of
     perpendicular electric field",
    Physical Review B 96, 045405 (2017).
    DOI: 10.1103/PhysRevB.96.045405

Every function below implements ONE numbered equation from the paper and
nothing else.  The paper's own printed form is quoted verbatim in each
docstring, together with the page it appears on, so that any line of any
figure script can be traced back to the source.

RULES FOLLOWED IN THIS FILE
---------------------------
1. No equation is simplified, rearranged for convenience, or replaced by
   an equivalent numerical shortcut.  Where the paper writes a polynomial,
   this file solves that polynomial; it does NOT diagonalise an equivalent
   matrix instead.
2. Every symbol keeps the paper's name (xi, kappa, alpha, d1..d4, t,
   epsilon, rho, k, eta, zeta, ...).
3. Where the published paper is internally inconsistent, BOTH printed
   forms are implemented, each labelled with its equation number, and the
   discrepancy is documented rather than silently "fixed".  See the
   section DOCUMENTED DISCREPANCIES below.

DOCUMENTED DISCREPANCIES IN THE PUBLISHED PAPER
-----------------------------------------------
(D1) Eq. (1) versus Eq. (3) - sign of the spin-orbit term.
     Eq. (1) (p.1) defines   xi1 = kappa + tau*s*lam + ...
     Its characteristic polynomial is therefore
         [(e-alpha)(e+kappa+tau*s*lam) - k^2]
       * [(e-kappa)(e+alpha-tau*s*lam) - k^2]
       -  gamma^2 (e-alpha)(e-kappa) = 0
     Eq. (3) (p.2) is printed as
         [(e-alpha')(e+kappa'-tau*s*lam') - k^2]
       * [(e-kappa')(e+alpha'+tau*s*lam') - k^2]
       -  gamma'^2 (e-alpha')(e-kappa') = 0
     i.e. the tau*s*lam terms carry the OPPOSITE sign in both brackets.
     The two forms are related exactly by s -> -s: they produce the same
     set of four eigenvalues, but they exchange the spin-up and spin-down
     labels.
     Which one does the paper's own Fig. 1 follow?  In Fig. 1 (lower-left
     panel, K valley, V = 15 meV) the solid (spin-up) curve of the upper
     valence band lies ABOVE the dotted (spin-down) curve.  Eq. (1) gives
     up above down; Eq. (3) as printed gives up below down.  Fig. 1 is
     therefore drawn from Eq. (1)'s sign convention, and the tau*s*lam'
     signs in the printed Eq. (3) appear to be a typographical error.
     Both are provided here:
         eq3_epsilon_roots(..., sign_convention="eq3")  <- as printed
         eq3_epsilon_roots(..., sign_convention="eq1")  <- reproduces Fig. 1
     Nothing is hidden: the caller chooses and the choice is recorded.

(D2) Fig. 1 annotation typo.
     The red arrow label in Fig. 1's lower-left panel reads
     "2D - V - sqrt(lam^2 - gam^2) - Omega^up" (minus under the root),
     while the magenta label in the SAME panel and the body text on p.3
     both read sqrt(lam^2 + gam^2).  The "+" form is correct.

(D3) Eq. (12) does not follow from Eq. (5).
     Setting gamma = V = M_z = M_v = 0, Eq. (5) factorises into
         (e + d1)(e - d2) = n        with d1 = D' + s*l1, d2 = D'
         (e + d3)(e - d4) = n + 1    with d3 = D' - s*l1, d4 = D'
     (D' = Delta/hbar*omega_c, l1 = lambda/hbar*omega_c).  Writing the
     second factor in the form of Eq. (12) would require
         d3 - d4 = 2*s*l1   and   d3*d4 = D'^2 + 2*D'*s*l1,
     i.e. d3 = D' + 2*s*l1, whereas the printed d3 is D' - s*l1.
     Verified numerically at B = 20 T, n = 1: Eq. (5) with gamma = 0 gives
     valence roots -10.561, -8.899 while Eq. (12) gives -11.467, -7.987.
     The conduction roots agree to ~1e-3.
     Eq. (12) is a limiting-case cross-check in the paper and is NOT used
     to produce any figure, so this does not affect Figs. 1-14.  It is
     transcribed exactly as printed and flagged by the self-test.

NOTE (not a discrepancy) - the d-parameters are a valley-dependent
relabelling of the xi's.  Verified numerically:
    tau = +1 :  (d1, d2, d3, d4) = (xi1, xi2, xi3, xi4) / hbar*omega_c
    tau = -1 :  (d1, d2, d3, d4) = (xi3, xi4, xi1, xi2) / hbar*omega_c
This is deliberate: it is what makes Eqs. (5) and (10) valley-universal,
so a single polynomial serves both K and K'.

UNITS
-----
Energies in eV, magnetic field in T, wave vector in 1/m, unless a
docstring says otherwise.
"""

import numpy as np

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------
HBAR_EVS = 6.582119569e-16      # hbar in eV*s
HBAR_J = 1.054571817e-34        # hbar in J*s
E_CHARGE = 1.602176634e-19      # elementary charge in C
MU_B = 5.7883818060e-5          # Bohr magneton in eV/T
K_B = 8.617333262e-5            # Boltzmann constant in eV/K


# ---------------------------------------------------------------------------
# PARAMETERS OF THE PAPER
#
# Every value below is quoted from the paper, with the location given.
# No parameter in this file is fitted, tuned or invented.
# ---------------------------------------------------------------------------
class PaperParameters:
    """Parameter set of Zubair et al., PRB 96, 045405 (2017)."""

    # --- band-structure parameters -----------------------------------------
    # p.1, Introduction: "approximately 2*lambda = 150 meV and 2*Delta =
    # 1.66 eV".  p.1, after Eq. (1): "Delta the monolayer band gap".
    DELTA = 0.83                 # eV   ( = 1.66 / 2 )
    LAMBDA = 0.074               # eV   Fig. 1 caption: "lambda = 0.074 eV"
    GAMMA = 0.047                # eV   Fig. 1 caption: "gamma = 0.047 eV"

    # p.1, after Eq. (1): "v_F = 0.53 x 10^6 m/s [10] is the Fermi velocity"
    V_F = 0.53e6                 # m/s

    # --- g factors ---------------------------------------------------------
    # p.2: "g'_e = 2 is the free electron g factor and g'_s = 0.21 the
    # out-of-plane factor due to the strong SOC in MoS2", and
    # "M_v = g'_v mu_B B / 2 ... and g'_v = 3.57 [38]".
    G_E = 2.0
    G_S = 0.21
    G_V = 3.57
    G_PRIME = G_E + G_S          # = 2.21, the g' of M_z = g' mu_B B / 2

    # --- electron density --------------------------------------------------
    # Fig. 3 caption: "for an electron density n_e = 1.9 x 10^13 cm^-2"
    N_E = 1.9e17                 # m^-2   ( 1.9e13 cm^-2 )

    # --- temperature -------------------------------------------------------
    # Figs. 8-14 captions: "at T = 1 K"
    T = 1.0                      # K

    # --- electric field energy --------------------------------------------
    # Fig. 1 caption: "zero electric field energy (V = 0)" and "V = 15 meV"
    V_ZERO = 0.0                 # eV
    V_FINITE = 0.015             # eV

    # --- Landau level broadening ------------------------------------------
    # Fig. 9 caption: "for a LL width Gamma = 0.1 sqrt(B) meV"
    @staticmethod
    def gamma_width(B):
        """Gamma = 0.1*sqrt(B) meV, returned in eV.  Fig. 9 caption."""
        return 0.1 * np.sqrt(B) * 1e-3

    # --- lattice constant --------------------------------------------------
    # NOT GIVEN IN THE PAPER.  Figs. 1 and 2 plot the dimensionless
    # abscissa ka/pi, so a value of 'a' is needed to convert the axis into
    # a wave vector.  3.16 Angstrom is the standard MoS2 lattice constant.
    # This is the ONLY quantity in this file not taken from the paper, and
    # it affects the horizontal scale of Figs. 1-2 only, never the energies
    # at k = 0 nor any B-field figure.
    A_LATTICE = 3.16e-10         # m   (external value - see note above)


P = PaperParameters


# ---------------------------------------------------------------------------
# Eq. (1)  -  the one-electron Hamiltonian                            [p.1]
# ---------------------------------------------------------------------------
def eq1_xi_terms(s, tau, V, Mz=0.0, Mv=0.0,
                 Delta=P.DELTA, lam=P.LAMBDA):
    """xi_1..xi_4 of Eq. (1).

    Paper, p.1, immediately after Eq. (1), verbatim:

        xi_1^{s,tau} = kappa + tau*s*lambda + s*M_z - tau*M_v
        xi_2^{s,tau} = alpha - s*M_z + tau*M_v
        xi_3^{s,tau} = alpha - tau*s*lambda - s*M_z + tau*M_v
        xi_4^{s,tau} = kappa + s*M_z - tau*M_v
        with kappa = Delta + V and alpha = Delta - V

    tau = 1(-1) is for the K (K') valley;  s = +1(up), -1(down).
    Returns (xi1, xi2, xi3, xi4) in eV.
    """
    kappa = Delta + V
    alpha = Delta - V
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def eq1_hamiltonian(k, s, tau, V, Mz=0.0, Mv=0.0,
                    Delta=P.DELTA, lam=P.LAMBDA, gam=P.GAMMA, vF=P.V_F):
    """The 4x4 Hamiltonian H^tau of Eq. (1).

    Paper, p.1, Eq. (1), verbatim:

                  / -xi_1^{s,tau}   v_F*pi_-^tau     gamma            0          \\
        H^tau =   |  v_F*pi_+^tau   xi_2^{s,tau}     0                0          |
                  |  gamma          0               -xi_3^{s,tau}  v_F*pi_+^tau  |
                  \\  0              0                v_F*pi_-^tau   xi_4^{s,tau} /

    with pi_pm^tau = tau*pi_x pm i*pi_y.

    Along the one-dimensional cut used for Figs. 1-2 the off-diagonal
    entries have modulus hbar*v_F*|k|; the real-symmetric form below is
    unitarily equivalent and has identical eigenvalues.

    k in 1/m.  Returns a 4x4 array in eV.
    """
    xi1, xi2, xi3, xi4 = eq1_xi_terms(s, tau, V, Mz, Mv, Delta, lam)
    t = HBAR_EVS * vF * abs(k)
    return np.array([
        [-xi1,  t,    gam,   0.0],
        [t,     xi2,  0.0,   0.0],
        [gam,   0.0, -xi3,   t  ],
        [0.0,   0.0,  t,     xi4],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Eq. (2)  -  energy from the dimensionless factor epsilon            [p.2]
# ---------------------------------------------------------------------------
def eq2_energy(epsilon, vF=P.V_F):
    """Eq. (2).

    Paper, p.2, Eq. (2), verbatim:

        E_mu^{s,tau}(k) = hbar * v_F * epsilon_mu^{s,tau}(k)

    epsilon in 1/m  ->  E in eV.
    """
    return HBAR_EVS * vF * np.asarray(epsilon)


# ---------------------------------------------------------------------------
# Eq. (3)  -  the fourth-degree equation for epsilon at B = 0         [p.2]
# ---------------------------------------------------------------------------
def eq3_coefficients(k, s, tau, V, sign_convention="eq3",
                     Delta=P.DELTA, lam=P.LAMBDA, gam=P.GAMMA, vF=P.V_F):
    """Polynomial coefficients of Eq. (3), highest power first.

    Paper, p.2, Eq. (3), verbatim:

        [ (eps - alpha')(eps + kappa' - tau*s*lambda') - k^2 ]
      x [ (eps - kappa')(eps + alpha' + tau*s*lambda') - k^2 ]
      -   gamma'^2 (eps - alpha')(eps - kappa')  =  0

    Paper, p.2, immediately after Eq. (3), verbatim:

        "where k = k_y is the wave vector, epsilon = E/hbar*v_F,
         lambda' = lambda/hbar*v_F, kappa' = kappa/hbar*v_F,
         gamma' = gamma/hbar*v_F, and alpha' = alpha/hbar*v_F."

    with kappa = Delta + V and alpha = Delta - V from Eq. (1).

    sign_convention
        "eq3" : the tau*s*lambda' signs exactly as printed in Eq. (3).
        "eq1" : the signs implied by Eq. (1)'s xi_1 and xi_3, which is
                what the paper's own Fig. 1 is drawn from.
        See DOCUMENTED DISCREPANCIES (D1) in the module docstring.

    All quantities are in 1/m, so epsilon comes out in 1/m and must be
    converted to eV with Eq. (2).
    """
    hv = HBAR_EVS * vF                      # hbar*v_F in eV*m
    kappa_p = (Delta + V) / hv              # kappa'
    alpha_p = (Delta - V) / hv              # alpha'
    lam_p = lam / hv                        # lambda'
    gam_p = gam / hv                        # gamma'

    if sign_convention == "eq3":
        L = tau * s * lam_p                 # bracket1: -L , bracket2: +L
    elif sign_convention == "eq1":
        L = -tau * s * lam_p
    else:
        raise ValueError("sign_convention must be 'eq3' or 'eq1'")

    # [ (eps - alpha')(eps + kappa' - L) - k^2 ]
    b1 = np.polyadd(np.polymul([1.0, -alpha_p], [1.0, kappa_p - L]),
                    [-k ** 2])
    # [ (eps - kappa')(eps + alpha' + L) - k^2 ]
    b2 = np.polyadd(np.polymul([1.0, -kappa_p], [1.0, alpha_p + L]),
                    [-k ** 2])
    # gamma'^2 (eps - alpha')(eps - kappa')
    b3 = np.polymul([1.0, -alpha_p], [1.0, -kappa_p])

    return np.polyadd(np.polymul(b1, b2), -(gam_p ** 2) * np.asarray(b3))


def eq3_epsilon_roots(k, s, tau, V, sign_convention="eq3", **kw):
    """The four roots epsilon of Eq. (3), sorted ascending.  Units 1/m."""
    return np.sort(np.roots(eq3_coefficients(k, s, tau, V,
                                             sign_convention, **kw)).real)


def eq2_eq3_band_energies(k, s, tau, V, sign_convention="eq3", **kw):
    """Band energies at B = 0: solve Eq. (3), then apply Eq. (2).

    This is the exact route the paper prescribes for Figs. 1 and 2:
    Eq. (3) gives epsilon, Eq. (2) turns it into E.  Returns 4 energies in
    eV, ascending: [0],[1] = valence, [2],[3] = conduction.
    """
    return eq2_energy(eq3_epsilon_roots(k, s, tau, V, sign_convention, **kw),
                      vF=kw.get("vF", P.V_F))


# ---------------------------------------------------------------------------
# Eq. (4)  -  Landau level energy                                     [p.3]
# ---------------------------------------------------------------------------
def eq4_cyclotron_frequency(B, vF=P.V_F):
    """omega_c of Eq. (4).

    Paper, p.3, after Eq. (4), verbatim:
        "where omega_c = v_F * sqrt(2*e*B/hbar) is the cyclotron frequency"

    Returns rad/s.
    """
    return vF * np.sqrt(2.0 * E_CHARGE * B / HBAR_J)


def eq4_hbar_omega_c(B, vF=P.V_F):
    """hbar*omega_c in eV, the energy scale of Eq. (4)."""
    return HBAR_J * eq4_cyclotron_frequency(B, vF) / E_CHARGE


def eq4_energy(epsilon, B, vF=P.V_F):
    """Eq. (4).

    Paper, p.3, Eq. (4), verbatim:

        E_{n,mu}^{s,tau} = hbar * omega_c * epsilon_{n,mu}^{s,tau}
    """
    return eq4_hbar_omega_c(B, vF) * np.asarray(epsilon)


# ---------------------------------------------------------------------------
# Eq. (5)  -  the fourth-order equation for the Landau levels         [p.3]
# ---------------------------------------------------------------------------
def eq5_d_parameters(B, s, tau, V, zeeman=True,
                     Delta=P.DELTA, lam=P.LAMBDA, gam=P.GAMMA, vF=P.V_F):
    """The dimensionless parameters t, d_1..d_4 used by Eq. (5).

    Paper, p.3, immediately after Eq. (5), verbatim:

        t          = gamma/hbar*omega_c
        d_1^{s,tau} = kappa^tau + s*lambda + tau*(s*M_z - tau*M_v)/hbar*omega_c
        d_2^{s,tau} = alpha^tau - tau*(s*M_z - tau*M_v)/hbar*omega_c
        d_3^{s,tau} = alpha^tau - s*lambda - tau*(s*M_z - tau*M_v)/hbar*omega_c
        d_4^{s,tau} = kappa^tau + tau*(s*M_z - tau*M_v)/hbar*omega_c
        where kappa^tau = Delta + tau*V and alpha^tau = Delta - tau*V
        are dimensionless parameters.

    NOTE on the printed form: kappa^tau, alpha^tau and s*lambda are stated
    to be "dimensionless", so each is understood to be divided by
    hbar*omega_c, exactly as the explicit Zeeman terms are.  Note also
    that d_1 carries "+ s*lambda" WITHOUT a factor tau, unlike xi_1 of
    Eq. (1) which carries "+ tau*s*lambda"; this is reproduced faithfully.

    Returns (d1, d2, d3, d4, t, hbar_omega_c_in_eV).
    """
    hw = eq4_hbar_omega_c(B, vF)
    Mz = eq_zeeman_Mz(B) if zeeman else 0.0
    Mv = eq_zeeman_Mv(B) if zeeman else 0.0

    kappa_tau = (Delta + tau * V) / hw
    alpha_tau = (Delta - tau * V) / hw
    lam_hw = lam / hw
    t = gam / hw
    Z = tau * (s * Mz - tau * Mv) / hw

    d1 = kappa_tau + s * lam_hw + Z
    d2 = alpha_tau - Z
    d3 = alpha_tau - s * lam_hw - Z
    d4 = kappa_tau + Z
    return d1, d2, d3, d4, t, hw


def eq5_coefficients(n, d1, d2, d3, d4, t):
    """Polynomial coefficients of Eq. (5), highest power first.

    Paper, p.3, Eq. (5), verbatim:

        [ (eps + d_1^{s,tau})(eps - d_2^{s,tau}) - n ]
      x [ (eps + d_3^{s,tau})(eps - d_4^{s,tau}) - (n+1) ]
      -   t^2 (eps - d_2^{s,tau})(eps - d_4^{s,tau})  =  0

    Valid for n >= 1 (p.3: "For n >= 1 the factor eps is the solution of
    the fourth-order equation").
    """
    p1 = np.polyadd(np.polymul([1.0, d1], [1.0, -d2]), [-float(n)])
    p2 = np.polyadd(np.polymul([1.0, d3], [1.0, -d4]), [-float(n + 1)])
    p3 = np.polymul([1.0, -d2], [1.0, -d4])
    return np.polyadd(np.polymul(p1, p2), -(t ** 2) * np.asarray(p3))


def eq5_epsilon_roots(n, d1, d2, d3, d4, t):
    """The four roots epsilon of Eq. (5), sorted ascending (dimensionless)."""
    return np.sort(np.roots(eq5_coefficients(n, d1, d2, d3, d4, t)).real)


# ---------------------------------------------------------------------------
# Zeeman terms (defined in the text on p.1-p.2, not separately numbered)
# ---------------------------------------------------------------------------
def eq_zeeman_Mz(B, g_prime=P.G_PRIME):
    """M_z = g' * mu_B * B / 2.

    Paper, p.1, after Eq. (1), verbatim:
        "M_z = g' mu_B B/2 is the Zeeman exchange field induced by
         ferromagnetic order, g' the Lande g factor (g' = g'_e + g'_s)"
    Paper, p.2: g'_e = 2, g'_s = 0.21.
    """
    return g_prime * MU_B * B / 2.0


def eq_zeeman_Mv(B, g_v=P.G_V):
    """M_v = g'_v * mu_B * B / 2.

    Paper, p.2, verbatim:
        "The term M_v = g'_v mu_B B/2 breaks the valley symmetry of the
         levels and g'_v = 3.57 [38]."
    """
    return g_v * MU_B * B / 2.0


# ---------------------------------------------------------------------------
# Eq. (6) / Eq. (7)  -  eigenfunctions and their coefficients         [p.3]
# ---------------------------------------------------------------------------
def eq7_k_coefficient(eps, n, d1, d2, t):
    """k_{n,mu}^{s,tau} of Eq. (7).

    Paper, p.3, immediately after Eq. (7), verbatim:

        k_{n,mu}^{s,tau} = [ (eps_{n,mu}^{s,tau} + d_1^{s,tau})
                            (eps_{n,mu}^{s,tau} - d_2^{s,tau}) - n ]
                           / [ t (eps_{n,mu}^{s,tau} - d_2^{s,tau}) ]
    """
    return ((eps + d1) * (eps - d2) - n) / (t * (eps - d2))


def eq7_rho_normalisation(eps, n, k_coeff, d2, d4):
    """rho_{n,mu}^{s,tau} of Eq. (7), the normalisation constant.

    Paper, p.3, Eq. (7), verbatim:

        rho_{n,mu}^{s,tau} = { (k_{n,mu}^{s,tau})^2
                               [ 1 + (n+1)/(eps_{n,mu}^{s,tau} - d_4^{s,tau})^2 ]
                             + 1
                             + n/(eps_{n,mu}^{s,tau} - d_2^{s,tau})^2 }^{-1/2}
    """
    inner = (k_coeff ** 2) * (1.0 + (n + 1) / (eps - d4) ** 2) \
        + 1.0 + n / (eps - d2) ** 2
    return inner ** -0.5


def eq6_coefficients(eps, n, d1, d2, d4, t):
    """The coefficients Theta, Lambda, Upsilon of Eq. (6).

    Paper, p.3, between Eqs. (6) and (7), verbatim:

        Theta_{n,mu}^{s,tau}  = sqrt(n)   rho_{n,mu}^{s,tau}
                                / [ eps_{n,mu}^{s,tau} - d_2^{s,tau} ]
        Lambda_{n,mu}^{s,tau} = k_{n,mu}^{s,tau} rho_{n,mu}^{s,tau}
        Upsilon_{n,mu}^{s,tau}= sqrt(n+1) k_{n,mu}^{s,tau} rho_{n,mu}^{s,tau}
                                / [ eps_{n,mu}^{s,tau} - d_4^{s,tau} ]

    Returns (rho, Theta, Lambda, Upsilon).
    """
    k_c = eq7_k_coefficient(eps, n, d1, d2, t)
    rho = eq7_rho_normalisation(eps, n, k_c, d2, d4)
    Theta = np.sqrt(n) * rho / (eps - d2)
    Lambda = k_c * rho
    Upsilon = np.sqrt(n + 1) * k_c * rho / (eps - d4)
    return rho, Theta, Lambda, Upsilon


# ---------------------------------------------------------------------------
# Eq. (8)  -  the n = -1 Landau level                                 [p.4]
# ---------------------------------------------------------------------------
def eq8_n_minus_1_energy(s, tau, V, B, zeeman=True, **kw):
    """Eq. (8): the single n = -1 level.

    Paper, p.4, Eq. (8), verbatim:

        H_{n=-1}^{+} = xi_4^{+},      H_{n=-1}^{-} = xi_2^{-}

    i.e. the K (tau=+1) valley level is xi_4 and the K' (tau=-1) valley
    level is xi_2.  Returns the energy in eV.
    """
    Mz = eq_zeeman_Mz(B) if zeeman else 0.0
    Mv = eq_zeeman_Mv(B) if zeeman else 0.0
    xi1, xi2, xi3, xi4 = eq1_xi_terms(s, tau, V, Mz, Mv, **kw)
    return xi4 if tau == 1 else xi2


# ---------------------------------------------------------------------------
# Eq. (9)  -  the n = 0 Hamiltonians                                  [p.4]
# ---------------------------------------------------------------------------
def eq9_n0_hamiltonian(s, tau, V, B, zeeman=True,
                       gam=P.GAMMA, vF=P.V_F, **kw):
    """Eq. (9): the 3x3 Hamiltonian of the n = 0 level.

    Paper, p.4, Eq. (9), verbatim:

                   / -xi_1^{s,+}   gamma        0           \\
        H_{n=0}^+ =|  gamma       -xi_3^{s,+}   hbar*omega_c |
                   \\  0            hbar*omega_c  xi_4^{s,+}  /

                   / -xi_1^{s,-}   hbar*omega_c  gamma       \\
        H_{n=0}^- =|  hbar*omega_c  xi_2^{s,-}   0           |
                   \\  gamma        0           -xi_3^{s,-}  /

    Note the two valleys have genuinely different matrices - this is not a
    typo in the paper, it follows from phi_{-1} = 0 removing a different
    component in each valley (p.3, after Eq. (7)).

    Returns a 3x3 array in eV.
    """
    Mz = eq_zeeman_Mz(B) if zeeman else 0.0
    Mv = eq_zeeman_Mv(B) if zeeman else 0.0
    xi1, xi2, xi3, xi4 = eq1_xi_terms(s, tau, V, Mz, Mv, **kw)
    hw = eq4_hbar_omega_c(B, vF)
    if tau == 1:
        return np.array([[-xi1,  gam,  0.0],
                         [gam,  -xi3,  hw ],
                         [0.0,   hw,   xi4]], dtype=float)
    return np.array([[-xi1,  hw,   gam ],
                     [hw,    xi2,  0.0 ],
                     [gam,   0.0, -xi3]], dtype=float)


# ---------------------------------------------------------------------------
# Eq. (10)  -  the cubic equation for the n = 0 level                 [p.4]
# ---------------------------------------------------------------------------
def eq10_cubic_coefficients(d1, d3, d4, t):
    """Polynomial coefficients of Eq. (10), highest power first.

    Paper, p.4, Eq. (10), verbatim:

        (eps + d_1^{s,tau}) [ (eps + d_3^{s,tau})(eps - d_4^{s,tau}) - 1 ]
        - t^2 (eps - d_4^{s,tau})  =  0

    Paper, p.4, before Eq. (10), verbatim:
        "The factor eps corresponding to Eq. (9) is given by the roots of
         the cubic equation"

    This single form is correct for BOTH valleys, even though the two
    matrices of Eq. (9) look different, because the d-parameters are a
    valley-dependent relabelling of the xi's (see the NOTE in the module
    docstring).  Verified against Eq. (9) for both valleys and both spins
    in the self-test.
    """
    inner = np.polyadd(np.polymul([1.0, d3], [1.0, -d4]), [-1.0])
    return np.polyadd(np.polymul([1.0, d1], inner),
                      -(t ** 2) * np.array([1.0, -d4]))


def eq10_epsilon_roots(d1, d3, d4, t):
    """The three roots epsilon of Eq. (10), sorted ascending."""
    return np.sort(np.roots(eq10_cubic_coefficients(d1, d3, d4, t)).real)


# ---------------------------------------------------------------------------
# Eq. (11)  -  the n = 0 eigenstates                                  [p.4]
# ---------------------------------------------------------------------------
# Paper, p.4, Eq. (11), verbatim:
#
#     psi_{0,mu}^{s,+} = (1/sqrt(L_y)) ( rho_{0,mu}^{s,+} phi_0,
#                                        0,
#                                        Lambda_{0,mu}^{s,+} phi_0,
#                                        Upsilon_{0,mu}^{s,+} phi_1 )^T e^{i k_y y}
#
#     psi_{0,mu}^{s,-} = (1/sqrt(L_y)) ( Lambda_{0,mu}^{s,-} phi_0,
#                                        Upsilon_{0,mu}^{s,-} phi_1,
#                                        rho_{0,mu}^{s,-} phi_0,
#                                        0 )^T e^{i k_y y}
#
# i.e. Theta_{0,mu} = 0 in both valleys, consistent with Eq. (7) at n = 0.
# The coefficients follow from eq6_coefficients() evaluated at n = 0; this
# is verified numerically in the self-test at the bottom of this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Eqs. (12)-(15)  -  limiting cases                                   [p.4]
#
# These are consistency checks in the paper, not used to draw any figure.
# They are transcribed here so that the equation set is complete and so
# that they can be used to validate Eq. (5).
# ---------------------------------------------------------------------------
def eq12_monolayer_epsilon(n, s, B, Delta=P.DELTA, lam=P.LAMBDA, vF=P.V_F):
    """Eq. (12): monolayer / uncoupled unbiased layers.

    Paper, p.4, Eq. (12), verbatim:

        eps = -s*lambda_1 pm [ (Delta' + s*lambda_1)^2 + (n+1) ]^{1/2}
        eps =  s*lambda_1 pm [ (Delta' - s*lambda_1)^2 + n     ]^{1/2}

        "where Delta' = Delta/hbar*omega_c and lambda_1 = lambda/hbar*omega_c"

    Obtained by "Setting gamma = V = 0 and M_z = M_v = 0 in Eq. (4)".
    Returns the four roots, ascending.
    """
    hw = eq4_hbar_omega_c(B, vF)
    Dp, l1 = Delta / hw, lam / hw
    r1 = np.sqrt((Dp + s * l1) ** 2 + (n + 1))
    r2 = np.sqrt((Dp - s * l1) ** 2 + n)
    return np.sort([-s * l1 + r1, -s * l1 - r1, s * l1 + r2, s * l1 - r2])


def eq13_graphene_epsilon(n):
    """Eq. (13): monolayer graphene.

    Paper, p.4, Eq. (13), verbatim:

        eps = pm sqrt(n+1),    eps = pm sqrt(n)

    Obtained by setting Delta' = lambda_1 = 0 in Eq. (12).
    """
    return np.sort([np.sqrt(n + 1), -np.sqrt(n + 1), np.sqrt(n), -np.sqrt(n)])


def eq14_bilayer_graphene_epsilon(n, t, branch="+"):
    """Eq. (14): bilayer graphene LL spectrum.

    Paper, p.4, Eq. (14), verbatim:

        eps = pm (1/sqrt(2)) ( t^2 + 2(2n+1)
              pm { [t^2 + 2(2n+1)]^2 - 16 n (n+1) }^{1/2} )^{1/2}

    Obtained "For Delta = lambda = V = M_z = M_v = 0".
    Paper, p.4, after Eq. (15): "The energy of higher LLs is obtained by
    taking the + sign in front of the internal square root in Eq. (14)."

    branch selects the internal sign.  Returns (+eps, -eps).
    """
    a = t ** 2 + 2 * (2 * n + 1)
    inner = np.sqrt(a ** 2 - 16.0 * n * (n + 1))
    val = np.sqrt(a + inner if branch == "+" else a - inner) / np.sqrt(2.0)
    return val, -val


def eq15_bilayer_graphene_approx(n, t):
    """Eq. (15): simplified bilayer graphene limit.

    Paper, p.4, Eq. (15), verbatim:

        eps = pm 2 sqrt( n (n+1) ) / t

    Valid in the limit n << t^2, taking the negative internal sign in
    Eq. (14).
    """
    val = 2.0 * np.sqrt(n * (n + 1)) / t
    return val, -val


# ---------------------------------------------------------------------------
# SELF-TEST
#
# Verifies that this file is faithful to the paper, by checking the
# internal consistency the paper itself asserts.
# ---------------------------------------------------------------------------
def _self_test(verbose=True):
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    if verbose:
        print("Self-test of paper_equations.py")
        print("-" * 62)

    k = (np.pi / P.A_LATTICE) * 0.03

    # Eq. (3) with Eq. (1)'s sign must reproduce Eq. (1)'s eigenvalues.
    for s in (+1, -1):
        for tau in (+1, -1):
            for V in (0.0, 0.015):
                mat = np.sort(np.linalg.eigvalsh(
                    eq1_hamiltonian(k, s, tau, V)))
                pol = eq2_eq3_band_energies(k, s, tau, V,
                                            sign_convention="eq1")
                check(f"Eq.(3)[eq1 sign] == Eq.(1) eigenvalues "
                      f"(s={s:+d},tau={tau:+d},V={V})",
                      np.allclose(mat, pol, atol=1e-9))

    # Eq. (3) as printed equals Eq. (1) with s -> -s  (discrepancy D1).
    a = eq2_eq3_band_energies(k, +1, +1, 0.015, sign_convention="eq3")
    b = np.sort(np.linalg.eigvalsh(eq1_hamiltonian(k, -1, +1, 0.015)))
    check("Eq.(3) as printed == Eq.(1) with s -> -s  [discrepancy D1]",
          np.allclose(a, b, atol=1e-9))

    # The d-parameters are a valley-dependent relabelling of the xi's.
    B, V = 20.0, 0.015
    for s in (+1, -1):
        for tau, order in ((+1, (0, 1, 2, 3)), (-1, (2, 3, 0, 1))):
            d1, d2, d3, d4, t, hw = eq5_d_parameters(B, s, tau, V)
            xi = np.array(eq1_xi_terms(s, tau, V, eq_zeeman_Mz(B),
                                       eq_zeeman_Mv(B))) / hw
            check(f"d-parameters == xi relabelling (s={s:+d},tau={tau:+d})",
                  np.allclose([d1, d2, d3, d4], xi[list(order)], atol=1e-9))

    # Eq. (10) must be the characteristic polynomial of Eq. (9).
    for tau in (+1, -1):
        for s in (+1, -1):
            d1, d2, d3, d4, t, hw = eq5_d_parameters(B, s, tau, V)
            mat = np.sort(np.linalg.eigvalsh(
                eq9_n0_hamiltonian(s, tau, V, B))) / hw
            pol = eq10_epsilon_roots(d1, d3, d4, t)
            check(f"Eq.(10) == Eq.(9) eigenvalues (s={s:+d},tau={tau:+d})",
                  np.allclose(mat, pol, atol=1e-7))

    # Eq. (7): the normalisation identity rho^2+Theta^2+Lambda^2+Ups^2 = 1.
    d1, d2, d3, d4, t, hw = eq5_d_parameters(20.0, +1, +1, 0.015)
    for n in (1, 5, 20):
        for eps in eq5_epsilon_roots(n, d1, d2, d3, d4, t):
            rho, Th, La, Up = eq6_coefficients(eps, n, d1, d2, d4, t)
            check(f"Eq.(6)/(7) normalisation, n={n}, eps={eps:.4f}",
                  abs(rho**2 + Th**2 + La**2 + Up**2 - 1.0) < 1e-10)

    # Eq. (11): Theta_{0,mu} = 0, so eq6 at n=0 must give Theta = 0.
    d1, d2, d3, d4, t, hw = eq5_d_parameters(20.0, +1, +1, 0.015)
    for eps in eq10_epsilon_roots(d1, d3, d4, t):
        _, Th, _, _ = eq6_coefficients(eps, 0, d1, d2, d4, t)
        check(f"Eq.(11) Theta_0 = 0 (eps={eps:.4f})", abs(Th) < 1e-12)

    # Eq. (5) -> Eq. (12) at gamma = V = M_z = M_v = 0.
    # This is EXPECTED TO DISAGREE - see discrepancy (D3).  Reported, not
    # asserted, because Eq. (12) draws no figure.
    if verbose:
        print("  --- known paper discrepancy (D3), reported not asserted ---")
    for s in (+1, -1):
        dd = eq5_d_parameters(20.0, s, +1, 0.0, zeeman=False, gam=0.0)
        agree = np.allclose(eq5_epsilon_roots(1, *dd[:5]),
                            eq12_monolayer_epsilon(1, s, 20.0), atol=1e-8)
        if verbose:
            print(f"  [{'agrees' if agree else 'DIFFERS as documented'}] "
                  f"Eq.(5)->Eq.(12), s={s:+d}")

    # Eq. (14) must reduce to Eq. (15) for n << t^2.
    check("Eq.(14) -> Eq.(15) for n << t^2",
          abs(eq14_bilayer_graphene_epsilon(2, 300.0, "-")[0]
              - eq15_bilayer_graphene_approx(2, 300.0)[0]) < 1e-3)

    if verbose:
        print("-" * 62)
        print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    _self_test()
