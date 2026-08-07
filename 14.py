# Bilayer MoS2 Fig. 14 -- longitudinal (rho_xx) and Hall (rho_xy)
# resistivities vs B, T=1K, Mz,Mv != 0, via rho_xx=sigma_xx/S,
# rho_xy=sigma_xy/S with S = sigma_xx*sigma_yy - sigma_xy*sigma_yx
# ~= n_e^2*e^2/B^2 (the paper's own stated approximation). 2x2 panels:
# rows V=0/15meV, columns low-B/high-B.
#
# STATUS / CAVEATS (read before trusting this figure):
#   - rho_xx reuses Fig. 12's sigma_xx (Codes/12.py, Eq. 28) -- same
#     free-prefactor-A caveat applies (shape is physical, absolute scale
#     is an empirical fit to the reference's y-axis range since A is not
#     given numerically in the paper).
#   - rho_xy reuses Fig. 10's sigma_yx (Codes/10.py, Eq. 22 + Appendix A)
#     -- INHERITS its unresolved issue: that calculation's magnitude is
#     verified correct but it comes out as a smooth curve, not the
#     quantized staircase the reference shows for rho_xy. Do not treat
#     the rho_xy curve here as matching the reference's step structure.
#     See Codes/10.py's header for the full two-round investigation record
#     (level-crowding diagnostic, mu-pairing test, NMAX convergence test,
#     ultra-fine resolution scan, full manual re-derivation of Eq. 6/7/22-24
#     + Appendix A against the primary PDF text) -- genuinely unresolved,
#     not a skipped/neglected check.
#   - PAPER (p.10): rho_0 = A^{-1}*1e-35, using the SAME undetermined A
#     as Eq. 28's sigma_xx prefactor A=(e^2/h)(beta*N_I*|U_0|^2/(pi*l_B^2*
#     Gamma*k_s^2)). sigma_yx (Eq. 22/Appendix A), by contrast, has NO A
#     dependence at all -- it is a fixed, topological quantity. This means
#     rho_xx/rho_0 ~ A^2*(raw sigma_xx sum) while rho_xy/rho_0 ~ A^1*
#     (raw sigma_yx, A-free) -- two DIFFERENT powers of the same unknown
#     A. VERIFIED NUMERICALLY that a single A fit from rho_xy's height
#     predicts rho_xx ~38 orders of magnitude too small -- i.e. one shared
#     A cannot make both curves land in the reference's range at once with
#     the information given in the paper (no numeric N_I, U_0, k_s, Gamma
#     anywhere in the text). Given that, rho_xx and rho_xy below use TWO
#     INDEPENDENT empirical display scales (RHO_XX_SCALE, RHO_XY_SCALE),
#     each anchored to match the reference's B=2 T starting value -- same
#     category of approximation as Fig. 12's A_SCALE, just applied
#     separately to each curve instead of tying them together. Absolute
#     curve heights are therefore approximate; the B-dependence SHAPE
#     (from the real S=n_e^2e^2/B^2 division) is physical.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams['font.family'] = 'serif'

hbar_J = 1.054571817e-34
e_ch = 1.602176634e-19
muB = 5.7883818060e-5
vF = 0.53e6
Delta = 0.83
lam = 0.074
gamma = 0.047
g_e, g_s, g_v = 2.0, 0.21, 3.57
gprime = g_e + g_s
NE_TARGET = 1.9e17  # m^-2

SIGMA_XX_SCALE = 5200.0  # same empirical A_SCALE as Codes/11.py (Fig. 12)
# Independently anchored so rho_xx(B=2T), rho_xy(B=2T) ~ reference's
# starting values (~0.0015, ~0.0017 rho_0 respectively) -- see caveat above.
RHO_XX_SCALE = 5.1e-13
RHO_XY_SCALE = 2.6e-5


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


def n_ef_for(B):
    return 150 if B < 4.0 else 60


def build_levels(B, s, tau, z, V):
    NMAX = n_ef_for(B)
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
    return levels, (d1, d2, d3, d4, t, hw), NMAX


def fermi(eps_ev, EF, kBT):
    x = np.clip((eps_ev - EF) / kBT, -500, 500)
    return 1.0 / (1.0 + np.exp(x))


def fermi_energy_and_levels(B, z, V, T=1.0):
    kBT = 8.617333262e-5 * T
    per_st = {}
    Econd = []
    for tau in (1, -1):
        for s in (1, -1):
            levels, dparams, NMAX = build_levels(B, s, tau, z, V)
            per_st[(s, tau)] = (levels, dparams, NMAX)
            for n, lv in levels.items():
                for eps, k, rho in lv:
                    e_ev = eps * dparams[5]
                    if e_ev > 0.5:
                        Econd.append(e_ev)
    Econd = np.array(Econd)
    l_B2 = hbar_J / (e_ch * B)
    D0 = 2 * np.pi * l_B2

    def ne_of_EF(ef):
        x = np.clip((Econd - ef) / kBT, -500, 500)
        return np.sum(1.0 / (1.0 + np.exp(x))) / D0

    lo, hi = Econd.min() - 0.05, Econd.max() + 0.05
    EF = brentq(lambda ef: ne_of_EF(ef) - NE_TARGET, lo, hi, xtol=1e-10)
    return EF, per_st


def sigma_xx_total(per_st, EF, T=1.0):
    kBT = 8.617333262e-5 * T
    total = 0.0
    for (s, tau), (levels, (d1, d2, d3, d4, t, hw), NMAX) in per_st.items():
        eps_m1, _, _ = levels[-1][0]
        f_m1 = fermi(eps_m1 * hw, EF, kBT)
        total += f_m1 * (1 - f_m1)
        for n in range(0, NMAX + 1):
            for (eps_n, k_n, rho_n) in levels[n]:
                eps_n_d2 = eps_n - d2
                eps_n_d4 = eps_n - d4
                bracket = ((2 * n + 1) * (1 + k_n ** 2) ** 2
                           + (2 * n - 1) * n ** 2 / eps_n_d2 ** 4
                           + (2 * n + 3) * (n + 1) ** 2 * k_n ** 4 / eps_n_d4 ** 4)
                f_n = fermi(eps_n * hw, EF, kBT)
                total += rho_n ** 4 * bracket * f_n * (1 - f_n)
    return SIGMA_XX_SCALE * total


def sigma_yx_total(per_st, EF, T=1.0):
    kBT = 8.617333262e-5 * T
    total = 0.0
    for (s, tau), (levels, (d1, d2, d3, d4, t, hw), NMAX) in per_st.items():
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
    e2_h = e_ch ** 2 / (2 * np.pi * hbar_J)  # e^2/h in Siemens
    return total * e2_h


def resistivities(B, z, V):
    EF, per_st = fermi_energy_and_levels(B, z, V)
    sxx = sigma_xx_total(per_st, EF)
    syx = sigma_yx_total(per_st, EF)
    S = (NE_TARGET ** 2) * (e_ch ** 2) / (B ** 2)
    rho_xx = (sxx / S) * RHO_XX_SCALE
    rho_xy = (syx / S) * RHO_XY_SCALE
    return rho_xx, rho_xy


B_low = np.linspace(2.0, 15.0, 110)
B_high = np.linspace(15.0, 40.0, 90)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

for row, V in enumerate([0.0, 0.015]):
    for col, B_range in enumerate([B_low, B_high]):
        ax = axes[row][col]
        rxx = np.empty(len(B_range))
        rxy = np.empty(len(B_range))
        for i, B in enumerate(B_range):
            if i % 50 == 0:
                print(f"    B-point {i}/{len(B_range)} (B={B:.2f} T)...")
            rxx[i], rxy[i] = resistivities(B, True, V)
        ax.plot(B_range, rxy, color='red', lw=0.9, label=r"$\rho_{xy}$")
        ax.plot(B_range, rxx, color='black', lw=0.7, label=r"$\rho_{xx}$")
        ax.set_xlabel(r"$B$ (T)")
        ax.set_ylabel(r"$\rho_{xy},\rho_{xx}$ ($\rho_0$)")
        ax.legend(loc='upper left', fontsize=8, frameon=False)
        vmev = f"{V*1000:.0f}"
        ax.text(0.05, 0.75, f"V = {vmev} meV", transform=ax.transAxes,
                fontsize=9, ha='left', va='bottom')
        print(f"row(V={vmev}meV) col={col} done")

for ax in (axes[0][0], axes[1][0]):
    ax.set_xlim(2, 15)
    ax.set_ylim(0.0, 0.020)
for ax in (axes[0][1], axes[1][1]):
    ax.set_xlim(15, 40)
    ax.set_ylim(0.0, 0.05)

fig.suptitle(
    r"Longitudinal ($\rho_{xx}$) and Hall ($\rho_{xy}$) resistivities vs $B$, "
    r"$T=1$K, $M_z,M_v\neq 0$. NOTE: $\rho_{xy}$ inherits Fig. 10/11's "
    r"unresolved missing-plateau issue (see file header).",
    fontsize=9
)
fig.tight_layout()

plt.savefig("bilayer_MoS2_fig14_draft.png", dpi=200)
print("Saved: bilayer_MoS2_fig14_draft.png")
plt.show()
