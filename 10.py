# Bilayer MoS2 Fig. 10 -- Hall conductivity sigma_yx vs B, T=1K, V=0meV.
# STATUS: IN PROGRESS / NOT LOCKED. Do not treat this file's output as a
# finished figure -- see the status note below before touching it again.
#
# What's VERIFIED correct here (checked numerically, not just visually):
#   1. Eigenvector coefficients k_{n,mu}^{s,tau}, rho_{n,mu}^{s,tau} (Eq. 6/7)
#      for n>=1 (from the quartic roots) satisfy the normalization identity
#      rho^2+Theta^2+Lambda^2+Upsilon^2 = 1 EXACTLY (to 1e-10), for every
#      level tested.
#   2. The n=0 special case (3x3 matrix, Codes/3.py's n0_levels_eV) gives
#      eigenvector-extracted k (=Lambda/rho) that matches the SAME general
#      closed-form k formula evaluated with n=0, to 6 decimal places --
#      confirms the general Eq.7 formula extends cleanly to n=0, matching
#      Appendix A's k_{0,mu}^{s,tau} usage.
#   3. sigma_yx magnitude/envelope matches the reference PDF almost exactly:
#      B=6.5T -> 120.9 (e^2/h) vs reference's top-left value ~120;
#      B=13T -> 60.4 (e^2/h) vs reference's bottom value ~60.
#      This was only reached after fixing a units bug: the energy
#      differences in Eq. 22's denominator must use the DIMENSIONLESS
#      epsilon (same units as d1-d4, i.e. E/(hbar*omega_c)), NOT the actual
#      eV energy -- using eV there inflated results by ~1/hbar_omega_c^2
#      (a ~200-400x error).
#
# What's WRONG / UNRESOLVED:
#   The reference shows a quantized STAIRCASE (flat plateaus at integer
#   multiples of e^2/h, e.g. steps of height 2 and 4). This implementation
#   instead gives a perfectly SMOOTH, monotonic curve -- verified at B
#   spacings down to 0.005 T and by checking the derivative d(sigma)/dB
#   across the whole range (6.5-13T): it varies smoothly from -18.3 to
#   -4.7 with NO flat (near-zero-derivative) region anywhere. This is not
#   a plotting-resolution artifact.
#   RE-CHECKED (fresh diagnostic, B=8T, V=0): printed every (n,s,tau) level
#   within 10 meV of E_F. Found ~45 distinct levels that close, spanning
#   n=6 through n=18, with adjacent-n spacing near E_F of only ~0.3-0.5 meV
#   (vs kT=0.086 meV -- only ~4-6x, confirming the original hypothesis) even
#   though hbar*omega_c(8T) itself is 54.4 meV -- i.e. the LL ladder is
#   ~100x more tightly packed near E_F than the bare cyclotron scale would
#   suggest. This is a resonance/crowding effect from n being comparable to
#   d1..d4 ~ Delta/(hbar*omega_c) ~ 15 in these dimensionless quartic units,
#   not a spacing computed wrong by a constant factor (the SAME build_levels
#   code was independently verified against normalization identities and
#   against the n=0 closed-form, so the level energies themselves are not
#   obviously buggy).
#
#   SECOND, MORE THOROUGH INVESTIGATION ROUND (same day, fresh attempt at a
#   real fix, not just re-confirming the symptom):
#   1. Restricted the eta/zeta sum to same-index (sorted-energy-order) mu
#      pairing only, instead of the full 4x4 mu x mu' cross product, to test
#      whether over-summing unphysical cross-branch matrix elements was
#      smearing the steps. Result: virtually IDENTICAL output (120.85 vs
#      120.9 at B=6.5T) -- ruled out, this is not the cause.
#   2. Tested NMAX convergence (60 vs 100 vs 150 vs 250): sigma_yx changes by
#      <0.001% -- ruled out truncation/missing-tail as the cause.
#   3. Ultra-fine B resolution scan (0.001 T steps, 200x finer than the
#      earlier 0.005 T check): confirmed the curve is smooth to machine
#      precision at every scale tested -- not a plotting/sampling artifact.
#   4. Re-read Appendix A (Eq. A1-A6) and Eq. 22/23/24 directly from the PDF,
#      character by character, and manually re-derived the paper's own
#      algebraic reindexing note ("replacing n-1 with n... the sum starts at
#      n=1 for both terms") -- proved by hand that eta_{n,mu,mu'} =
#      zeta_{n+1,mu',mu} exactly, confirming this file's two-loop
#      implementation is mathematically equivalent to the paper's own
#      simplified form. No transcription error found.
#   5. Re-read Eq. 6/7 (the k_{n,mu} and rho_{n,mu} eigenvector formulas)
#      directly from the PDF and compared term-by-term against k_rho() below
#      -- matches exactly, including the exact placement of every d2/d4
#      term inside rho's normalization sum.
#   CONCLUSION: every formula from Eq. 4 through Eq. 24 plus Appendix A has
#   now been individually re-verified against the primary source text, not
#   just against this file's own internal consistency checks, and none of
#   the four most plausible numerical-implementation bugs (cross-term
#   over-summing, truncation, resolution artifact, formula transcription)
#   reproduce or explain the missing plateaus. This is now believed to be
#   either a genuine emergent feature of this specific 4-band model that
#   requires an additional regularization/broadening scheme the paper does
#   not state explicitly, or a bug too subtle to find by manual inspection
#   without the original authors' source code. NOT resolved -- do not spend
#   more time re-deriving the same formulas again without new information;
#   the next real lead would have to come from the authors directly.
import numpy as np
from scipy.optimize import brentq

hbar_J = 1.054571817e-34
e_ch = 1.602176634e-19
muB = 5.7883818060e-5
vF = 0.53e6
Delta = 0.83
lam = 0.074
gamma = 0.047
g_e, g_s, g_v = 2.0, 0.21, 3.57
gprime = g_e + g_s


def hw_eV(B):
    return hbar_J * vF * np.sqrt(2 * e_ch * B / hbar_J) / e_ch


def Mz_eV(B, z):
    return gprime * muB * B / 2.0 if z else 0.0


def Mv_eV(B, z):
    return g_v * muB * B / 2.0 if z else 0.0


def d_params(B, s, tau, z, V):
    hw = hw_eV(B)
    kappa_tau = (Delta + tau * V) / hw
    alpha_tau = (Delta - tau * V) / hw
    lam_hw = lam / hw
    t = gamma / hw
    Z = tau * (s * Mz_eV(B, z) - tau * Mv_eV(B, z)) / hw
    d1 = kappa_tau + s * lam_hw + Z
    d2 = alpha_tau - Z
    d3 = alpha_tau - s * lam_hw - Z
    d4 = kappa_tau + Z
    return d1, d2, d3, d4, t, hw


def quartic_roots(n, d1, d2, d3, d4, t):
    p1 = np.poly1d([1.0, d1 - d2, -d1 * d2 - n])
    p2 = np.poly1d([1.0, d3 - d4, -d3 * d4 - (n + 1)])
    p3 = np.poly1d([1.0, -(d2 + d4), d2 * d4])
    Q = p1 * p2 - (t ** 2) * p3
    return np.sort(np.roots(Q.coeffs).real)


def xi_terms(s, tau, B, z, V):
    kappa = Delta + V
    alpha = Delta - V
    Mz, Mv = Mz_eV(B, z), Mv_eV(B, z)
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def n_minus1_level_ev(s, tau, B, z, V):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, z, V)
    return xi4 if tau == 1 else xi2


def n0_levels_ev(s, tau, B, z, V):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, z, V)
    hw = hw_eV(B)
    if tau == 1:
        H = np.array([[-xi1, gamma, 0.0], [gamma, -xi3, hw], [0.0, hw, xi4]])
    else:
        H = np.array([[-xi1, hw, gamma], [hw, xi2, 0.0], [gamma, 0.0, -xi3]])
    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def k_rho(eps, n, d1, d2, d4, t):
    k = ((eps + d1) * (eps - d2) - n) / (t * (eps - d2))
    rho2 = 1.0 / (1 + n / (eps - d2) ** 2 + k ** 2 * (1 + (n + 1) / (eps - d4) ** 2))
    return k, np.sqrt(rho2)


def build_levels(B, s, tau, z, V, NMAX):
    """dict n -> list of (eps_dimless, k, rho); n=-1 has k=rho=None (trivial)."""
    d1, d2, d3, d4, t, hw = d_params(B, s, tau, z, V)
    levels = {}
    eps_m1 = n_minus1_level_ev(s, tau, B, z, V) / hw
    levels[-1] = [(eps_m1, None, None)]
    e0_ev, evecs0 = n0_levels_ev(s, tau, B, z, V)
    lv0 = []
    for i in range(3):
        eps0 = e0_ev[i] / hw
        rho0, Lam0, Ups0 = evecs0[:, i]
        k0 = Lam0 / rho0
        lv0.append((eps0, k0, abs(rho0)))
    levels[0] = lv0
    for n in range(1, NMAX + 1):
        roots = quartic_roots(n, d1, d2, d3, d4, t)
        lv = []
        for eps in roots:
            k, rho = k_rho(eps, n, d1, d2, d4, t)
            lv.append((eps, k, rho))
        levels[n] = lv
    return levels, (d1, d2, d3, d4, t, hw)


def fermi(eps_ev, EF, kBT):
    x = np.clip((eps_ev - EF) / kBT, -500, 500)
    return 1.0 / (1.0 + np.exp(x))


def fermi_energy(B, z, V, NMAX, ne_target_m2=1.9e17, T=1.0):
    kBT = 8.617333262e-5 * T
    Econd = []
    for tau in (1, -1):
        for s in (1, -1):
            levels, (d1, d2, d3, d4, t, hw) = build_levels(B, s, tau, z, V, NMAX)
            for n, lv in levels.items():
                for eps, k, rho in lv:
                    e_ev = eps * hw
                    if e_ev > 0.5:
                        Econd.append(e_ev)
    Econd = np.array(Econd)
    l_B2 = hbar_J / (e_ch * B)
    D0 = 2 * np.pi * l_B2

    def ne_of_EF(ef):
        x = np.clip((Econd - ef) / kBT, -500, 500)
        return np.sum(1.0 / (1.0 + np.exp(x))) / D0

    lo, hi = Econd.min() - 0.05, Econd.max() + 0.05
    return brentq(lambda ef: ne_of_EF(ef) - ne_target_m2, lo, hi, xtol=1e-10)


def sigma_yx(B, z, V, NMAX, EF, T=1.0):
    """Hall conductivity in units of e^2/h, per Eq. (22) + Appendix A."""
    kBT = 8.617333262e-5 * T
    total = 0.0
    for tau in (1, -1):
        for s in (1, -1):
            levels, (d1, d2, d3, d4, t, hw) = build_levels(B, s, tau, z, V, NMAX)
            for n in range(1, NMAX):
                for (eps_n, k_n, rho_n) in levels[n]:
                    for (eps_np1, k_np1, rho_np1) in levels[n + 1]:
                        eps_n_d4 = eps_n - d4
                        eps_np1_d2 = eps_np1 - d2
                        eta = (n + 1) * (rho_n * rho_np1) ** 2 * (k_n * k_np1 / eps_n_d4 + 1.0 / eps_np1_d2) ** 2
                        f_n = fermi(eps_n * hw, EF, kBT)
                        f_np1 = fermi(eps_np1 * hw, EF, kBT)
                        denom = (eps_n - eps_np1) ** 2
                        if denom < 1e-24:
                            continue
                        total += 0.5 * eta * (f_n - f_np1) / denom
            for n in range(2, NMAX + 1):
                for (eps_n, k_n, rho_n) in levels[n]:
                    for (eps_nm1, k_nm1, rho_nm1) in levels[n - 1]:
                        eps_nm1_d4 = eps_nm1 - d4
                        eps_n_d2 = eps_n - d2
                        zeta = n * (rho_n * rho_nm1) ** 2 * (k_n * k_nm1 / eps_nm1_d4 + 1.0 / eps_n_d2) ** 2
                        f_n = fermi(eps_n * hw, EF, kBT)
                        f_nm1 = fermi(eps_nm1 * hw, EF, kBT)
                        denom = (eps_n - eps_nm1) ** 2
                        if denom < 1e-24:
                            continue
                        total += -0.5 * zeta * (f_n - f_nm1) / denom
            for (eps_0, k_0, rho_0) in levels[0]:
                for (eps_1, k_1, rho_1) in levels[1]:
                    eps_1_d2 = eps_1 - d2
                    eps_0_d4 = eps_0 - d4
                    eta01 = (rho_0 * rho_1) ** 2 * (k_0 * k_1 / eps_0_d4 + 1.0 / eps_1_d2) ** 2
                    f_0 = fermi(eps_0 * hw, EF, kBT)
                    f_1 = fermi(eps_1 * hw, EF, kBT)
                    denom = (eps_0 - eps_1) ** 2
                    if denom < 1e-24:
                        continue
                    total += 1.0 * eta01 * (f_0 - f_1) / denom
            eps_m1, _, _ = levels[-1][0]
            f_m1 = fermi(eps_m1 * hw, EF, kBT)
            for (eps_0, k_0, rho_0) in levels[0]:
                term = (rho_0 * k_0) ** 2
                f_0 = fermi(eps_0 * hw, EF, kBT)
                denom = (eps_m1 - eps_0) ** 2
                if denom < 1e-24:
                    continue
                total += 1.0 * term * (f_m1 - f_0) / denom
    return total


def curve(B_range, z, V, NMAX):
    out = np.empty(len(B_range))
    for i, B in enumerate(B_range):
        if i % 50 == 0:
            print(f"    B-point {i}/{len(B_range)} (B={B:.2f} T)...")
        EF = fermi_energy(B, z, V, NMAX)
        out[i] = sigma_yx(B, z, V, NMAX, EF)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    V = 0.0

    B_low = np.linspace(6.5, 13.0, 90)
    B_high = np.linspace(13.0, 40.0, 120)
    B_inset_low = np.linspace(7.5, 9.5, 40)
    B_inset_high = np.linspace(20.0, 27.0, 55)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    panels = [
        (axes[0], B_low, B_inset_low, [0.42, 0.42, 0.5, 0.5]),
        (axes[1], B_high, B_inset_high, [0.35, 0.42, 0.55, 0.5]),
    ]
    for ax, B_range, B_inset, inset_pos in panels:
        NMAX = 150 if B_range[0] < 4.0 else 60
        black = curve(B_range, False, V, NMAX)
        red = curve(B_range, True, V, NMAX)
        ax.plot(B_range, black, color='black', lw=0.9, label=r"$M_z,M_v=0$")
        ax.plot(B_range, red, color='red', lw=0.9, label=r"$M_z,M_v\neq 0$")
        ax.set_xlabel(r"$B$ (T)")
        ax.set_ylabel(r"$\sigma_{xy}$ ($e^2/h$)")
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        ax.text(0.05, 0.08, "V = 0 meV", transform=ax.transAxes, fontsize=9)

        iax = ax.inset_axes(inset_pos)
        NMAX_i = 150 if B_inset[0] < 4.0 else 60
        iax.plot(B_inset, curve(B_inset, False, V, NMAX_i), color='black', lw=0.8)
        iax.plot(B_inset, curve(B_inset, True, V, NMAX_i), color='red', lw=0.8)
        iax.tick_params(labelsize=6)
        iax.set_xlabel(r"$B$ (T)", fontsize=6, labelpad=1)
        iax.set_ylabel(r"$\sigma_{xy}(e^2/h)$", fontsize=6, labelpad=1)
        print("  inset done")

    axes[0].set_xlim(6.5, 13.0)
    axes[1].set_xlim(13.0, 40.0)

    fig.suptitle(
        r"Hall conductivity $\sigma_{xy}$ vs $B$ at $T=1$ K, $V=0$ meV. "
        r"NOTE: envelope/magnitude verified, but quantized plateau structure "
        r"is NOT reproduced here -- see file header caveat.",
        fontsize=9
    )
    fig.tight_layout()
    plt.savefig("bilayer_MoS2_fig10_draft.png", dpi=200)
    print("Saved: bilayer_MoS2_fig10_draft.png")
