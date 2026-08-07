# Bilayer MoS2 Fig. 8 -- Fermi energy EF vs B at T = 1 K, zoomed in to
# resolve the fine intra-LL oscillations. 2x2 panels: rows are V=0/15 meV,
# columns are the low-B (2-13 T) / high-B (13-40 T) ranges (paper's own
# panel split). Each panel overlays Mz=Mv=0 (black) vs Mz,Mv != 0 (red).
#
# Reuses the exact conduction-band LL solver already verified for Figs.
# 6-7 (Codes/5.py, Codes/6.py); only new things here are (i) V threaded
# as a parameter instead of a fixed module constant, since both V=0 and
# V=15meV are needed side by side, and (ii) a much finer B grid, since
# resolving these oscillations needs far more points than the coarse
# fan plots did.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams['font.family'] = 'serif'

hbar_J = 1.054571817e-34     # J*s
e_ch = 1.602176634e-19       # C
muB = 5.7883818060e-5        # eV/T

vF = 0.53e6                  # m/s
Delta = 0.83                 # eV
lam = 0.074                  # eV
gamma = 0.047                # eV

g_e, g_s, g_v = 2.0, 0.21, 3.57
gprime = g_e + g_s

# N_EF=60 (vs Figs. 6/7's 150) verified <1e-9 eV agreement for B>=4T and
# V=0 at all B -- big speedup, needed since Fig. 8's fine B grid takes
# ~40x more brentq solves than Figs. 6/7 did. BUT for V=15meV at B<4T,
# N_EF=60 is NOT enough (verified: gives EF off by up to 1e-2 eV, a fake
# discontinuity/cliff around B~3T) -- kappa/alpha split by +-V there means
# more levels are needed to reach the same target density. Use 150 in that
# regime, 60 elsewhere.
def n_ef_for(B):
    return 150 if B < 4.0 else 60


def hw_eV(B):
    omega_c = vF * np.sqrt(2 * e_ch * B / hbar_J)
    return hbar_J * omega_c / e_ch


def Mz_eV(B, zeeman):
    return gprime * muB * B / 2.0 if zeeman else 0.0


def Mv_eV(B, zeeman):
    return g_v * muB * B / 2.0 if zeeman else 0.0


def d_params(B, s, tau, zeeman, V):
    hw = hw_eV(B)
    kappa_tau = (Delta + tau * V) / hw
    alpha_tau = (Delta - tau * V) / hw
    lam_hw = lam / hw
    t = gamma / hw
    Z = tau * (s * Mz_eV(B, zeeman) - tau * Mv_eV(B, zeeman)) / hw
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


def xi_terms(s, tau, B, zeeman, V):
    kappa = Delta + V
    alpha = Delta - V
    Mz, Mv = Mz_eV(B, zeeman), Mv_eV(B, zeeman)
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def n_minus1_level(s, tau, B, zeeman, V):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, zeeman, V)
    return xi4 if tau == 1 else xi2


def n0_levels_eV(s, tau, B, zeeman, V):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, zeeman, V)
    hw = hw_eV(B)
    if tau == 1:
        H = np.array([[-xi1, gamma, 0.0], [gamma, -xi3, hw], [0.0, hw, xi4]])
    else:
        H = np.array([[-xi1, hw, gamma], [hw, xi2, 0.0], [gamma, 0.0, -xi3]])
    return np.sort(np.linalg.eigvalsh(H))


def all_conduction_energies(B, zeeman, V):
    n_ef = n_ef_for(B)
    energies = []
    for tau in (1, -1):
        for s in (1, -1):
            energies.append(n_minus1_level(s, tau, B, zeeman, V))
            lvl0 = n0_levels_eV(s, tau, B, zeeman, V)
            labels0 = ['--', '-+', '++'] if tau == -1 else ['--', '-+', '+-']
            cond_label = '++' if tau == -1 else '+-'
            energies.append(lvl0[labels0.index(cond_label)])
            d1, d2, d3, d4, t, hw = d_params(B, s, tau, zeeman, V)
            for n in range(1, n_ef + 1):
                roots = quartic_roots(n, d1, d2, d3, d4, t) * hw
                energies.extend(roots[2:])
    return np.array(energies)


def fermi_energy_curve(B_values, zeeman, V, ne_target_m2=1.9e17, T=1.0):
    kBT = 8.617333262e-5 * T
    EF = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 50 == 0:
            print(f"    B-point {i}/{len(B_values)} (B={B:.2f} T)...")
        E = all_conduction_energies(B, zeeman, V)
        l_B2 = hbar_J / (e_ch * B)
        D0 = 2 * np.pi * l_B2

        def ne_of_EF(ef):
            x = np.clip((E - ef) / kBT, -500, 500)
            return np.sum(1.0 / (1.0 + np.exp(x))) / D0

        lo, hi = E.min() - 0.05, E.max() + 0.05
        EF[i] = brentq(lambda ef: ne_of_EF(ef) - ne_target_m2, lo, hi, xtol=1e-10)
    return EF


B_low = np.linspace(2, 13, 800)
B_high = np.linspace(13, 40, 500)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

for row, V in enumerate([0.0, 0.015]):
    for col, B_range in enumerate([B_low, B_high]):
        ax = axes[row][col]
        EF_off = fermi_energy_curve(B_range, False, V)
        EF_on = fermi_energy_curve(B_range, True, V)
        ax.plot(B_range, EF_off * 10, color='black', lw=0.8, label=r"$M_z,M_v=0$")
        ax.plot(B_range, EF_on * 10, color='red', lw=0.8, label=r"$M_z,M_v\neq 0$")
        ax.set_xlabel(r"$B$ (T)")
        ax.set_ylabel(r"$E_F$ ($10^{-1}$ eV)")
        ax.legend(loc='upper left', fontsize=8, frameon=False)
        vmev = f"{V*1000:.0f}"
        ax.text(0.97, 0.05, f"V = {vmev} meV", transform=ax.transAxes,
                fontsize=9, ha='right', va='bottom')
        print(f"V={vmev}meV col={col} done")

axes[0][0].set_xlim(2, 13)
axes[0][0].set_ylim(8.512, 8.521)
axes[0][1].set_xlim(13, 40)
axes[0][1].set_ylim(8.485, 8.545)
axes[1][0].set_xlim(2, 13)
axes[1][0].set_ylim(8.507, 8.521)
axes[1][1].set_xlim(13, 40)
axes[1][1].set_ylim(8.495, 8.545)

fig.suptitle(
    r"Fermi energy $E_F$ vs $B$ at $T=1$ K. Upper: $V=0$ meV; lower: "
    r"$V=15$ meV. Panels differ only in the range of $B$.",
    fontsize=10
)
fig.tight_layout()

plt.savefig("bilayer_MoS2_fig8_draft.png", dpi=200)
print("Saved: bilayer_MoS2_fig8_draft.png")
plt.show()
