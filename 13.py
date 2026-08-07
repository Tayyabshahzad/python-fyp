# Bilayer MoS2 Fig. 13 -- spin (Ps) and valley (Pv) polarization vs B,
# T=1K, V=15meV, Mz,Mv != 0 (paper's caption: "parameters same as Fig. 11
# for Mz != Mv != 0" -- Fig. 11 is V=15meV). Per Eq. (29)/(30):
#   Ps = [(sig^{K,up}+sig^{K',dn}) - (sig^{K,dn}+sig^{K',up})]
#        / [(sig^{K,up}+sig^{K',dn}) + (sig^{K,dn}+sig^{K',up})]
#   Pv = [(sig^{K,up}+sig^{K,dn}) - (sig^{K',up}+sig^{K',dn})]
#        / [(sig^{K,up}+sig^{K,dn}) + (sig^{K',up}+sig^{K',dn})]
# where K=tau=+1, K'=tau=-1, up=s=+1, dn=s=-1, and sig^{tau,s} is the
# per-(s,tau) contribution to Eq.(28)'s sigma_xx sum (Codes/11.py).
# Since Ps,Pv are RATIOS, Eq.(28)'s free prefactor A cancels exactly --
# no scaling ambiguity here, unlike Fig. 12.
# Also plots the Mz=Mv=0 case (both P's are exactly 0, per the paper's
# text -- shown as the flat blue curve in the reference).
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
V = 0.015  # Fig. 11/13 parameters: V = 15 meV


def hw_eV(B):
    return hbar_J * vF * np.sqrt(2 * e_ch * B / hbar_J) / e_ch


def Mz_eV(B, z):
    return gprime * muB * B / 2.0 if z else 0.0


def Mv_eV(B, z):
    return g_v * muB * B / 2.0 if z else 0.0


def d_params(B, s, tau, z):
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


def xi_terms(s, tau, B, z):
    kappa = Delta + V
    alpha = Delta - V
    Mz, Mv = Mz_eV(B, z), Mv_eV(B, z)
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def n_minus1_level_ev(s, tau, B, z):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, z)
    return xi4 if tau == 1 else xi2


def n0_levels_ev(s, tau, B, z):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, z)
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


def build_levels(B, s, tau, z):
    NMAX = n_ef_for(B)
    d1, d2, d3, d4, t, hw = d_params(B, s, tau, z)
    levels = {}
    eps_m1 = n_minus1_level_ev(s, tau, B, z) / hw
    levels[-1] = [(eps_m1, None, None)]
    e0_ev, evecs0 = n0_levels_ev(s, tau, B, z)
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


def fermi_energy(B, z, ne_target_m2=1.9e17, T=1.0):
    kBT = 8.617333262e-5 * T
    Econd = []
    for tau in (1, -1):
        for s in (1, -1):
            levels, (d1, d2, d3, d4, t, hw), NMAX = build_levels(B, s, tau, z)
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


def sigma_xx_per_branch(B, s, tau, z, EF, T=1.0):
    """Eq.(28)+Appendix B contribution from a single (s,tau) branch,
    WITHOUT the free prefactor A (cancels in the Ps/Pv ratios)."""
    kBT = 8.617333262e-5 * T
    levels, (d1, d2, d3, d4, t, hw), NMAX = build_levels(B, s, tau, z)
    total = 0.0
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
    return total


def polarizations(B, z):
    EF = fermi_energy(B, z)
    sK_up = sigma_xx_per_branch(B, +1, +1, z, EF)
    sK_dn = sigma_xx_per_branch(B, -1, +1, z, EF)
    sKp_up = sigma_xx_per_branch(B, +1, -1, z, EF)
    sKp_dn = sigma_xx_per_branch(B, -1, -1, z, EF)
    Ps_num = (sK_up + sKp_dn) - (sK_dn + sKp_up)
    Ps_den = (sK_up + sKp_dn) + (sK_dn + sKp_up)
    Pv_num = (sK_up + sK_dn) - (sKp_up + sKp_dn)
    Pv_den = (sK_up + sK_dn) + (sKp_up + sKp_dn)
    Ps = Ps_num / Ps_den if abs(Ps_den) > 1e-30 else 0.0
    Pv = Pv_num / Pv_den if abs(Pv_den) > 1e-30 else 0.0
    return Ps, Pv


B_grid = np.linspace(1.0, 30.0, 900)

Ps_vals = np.empty(len(B_grid))
Pv_vals = np.empty(len(B_grid))
for i, B in enumerate(B_grid):
    if i % 100 == 0:
        print(f"B-point {i}/{len(B_grid)} (B={B:.2f} T)...")
    Ps_vals[i], Pv_vals[i] = polarizations(B, True)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(B_grid, Ps_vals, color='black', lw=0.9, label=r"$P_s$")
ax.plot(B_grid, Pv_vals, color='red', lw=0.9, linestyle='--', label=r"$P_v$")
ax.axhline(0, color='blue', lw=1.2)
ax.set_xlim(0, 30)
ax.set_ylim(-1.05, 1.05)
ax.set_xlabel(r"$B$ (T)")
ax.set_ylabel(r"$P_s, P_v$")
ax.legend(loc='upper right', fontsize=9, frameon=False)

fig.suptitle(
    r"Spin $P_s$ and valley $P_v$ polarization vs $B$, $T=1$K, $V=15$meV, "
    r"$M_z,M_v\neq 0$. Blue: $M_z=M_v=0$ (both vanish).",
    fontsize=10
)
fig.tight_layout()

plt.savefig("bilayer_MoS2_fig13_draft.png", dpi=200)
print("Saved: bilayer_MoS2_fig13_draft.png")
plt.show()
