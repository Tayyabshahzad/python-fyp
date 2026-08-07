# Bilayer MoS2 Fig. 5 -- "As in Fig. 3 but for V = 15 meV" (paper's own
# caption). Identical physics/solver to Codes/2.py (Fig. 3); only V and
# the resulting axis ranges/legend positions change.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

hbar_J = 1.054571817e-34     # J*s
e_ch = 1.602176634e-19       # C
muB = 5.7883818060e-5        # eV/T

vF = 0.53e6                  # m/s
Delta = 0.83                 # eV
lam = 0.074                  # eV
gamma = 0.047                # eV
V = 0.015                    # eV -- Fig. 5 is V = 15 meV (the one change vs Fig. 3)

g_e, g_s, g_v = 2.0, 0.21, 3.57
gprime = g_e + g_s

N_MAX = 24
N_EF = 150
B_grid = np.linspace(0.4, 40, 240)


def omega_c(B):
    return vF * np.sqrt(2 * e_ch * B / hbar_J)


def hw_eV(B):
    return hbar_J * omega_c(B) / e_ch


def Mz_eV(B):
    return gprime * muB * B / 2.0


def Mv_eV(B):
    return g_v * muB * B / 2.0


def d_params(B, s, tau):
    """Dimensionless d1..d4 and t, all in units of hbar*omega_c."""
    hw = hw_eV(B)
    kappa_tau = (Delta + tau * V) / hw
    alpha_tau = (Delta - tau * V) / hw
    lam_hw = lam / hw
    t = gamma / hw
    Z = tau * (s * Mz_eV(B) - tau * Mv_eV(B)) / hw
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


def xi_terms(s, tau, B):
    """Original Eq. (1) xi's (eV) -- kappa=Delta+V, alpha=Delta-V (NO tau
    multiplying V here, unlike the LL d-parameters' kappa_tau/alpha_tau).
    Same convention as Fig. 3/Codes/2.py; matters now since V != 0."""
    kappa = Delta + V
    alpha = Delta - V
    Mz, Mv = Mz_eV(B), Mv_eV(B)
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def n_minus1_level(s, tau, B):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B)
    return xi4 if tau == 1 else xi2


def n0_levels_eV(s, tau, B):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B)
    hw = hw_eV(B)
    if tau == 1:
        H = np.array([[-xi1, gamma, 0.0], [gamma, -xi3, hw], [0.0, hw, xi4]])
    else:
        H = np.array([[-xi1, hw, gamma], [hw, xi2, 0.0], [gamma, 0.0, -xi3]])
    return np.sort(np.linalg.eigvalsh(H))


COND_COLOR = {1: {'+-': 'red', '++': 'black'}, -1: {'+-': 'blue', '++': 'green'}}
VAL_COLOR = {1: {'-+': 'red', '--': 'black'}, -1: {'-+': 'blue', '--': 'green'}}


def collect_branch(s, tau):
    out = {mu: np.full(len(B_grid), np.nan) for mu in ['--', '-+', '+-', '++']}
    for ib, B in enumerate(B_grid):
        n_m1 = n_minus1_level(s, tau, B)
        out['++' if tau == 1 else '+-'][ib] = n_m1

        lvl0 = n0_levels_eV(s, tau, B)
        labels0 = ['--', '-+', '++'] if tau == -1 else ['--', '-+', '+-']
        for lab, val in zip(labels0, lvl0):
            out[lab][ib] = val

    per_n = {mu: [] for mu in ['--', '-+', '+-', '++']}
    for n in range(1, N_MAX + 1):
        curve = {mu: np.full(len(B_grid), np.nan) for mu in ['--', '-+', '+-', '++']}
        for ib, B in enumerate(B_grid):
            d1, d2, d3, d4, t, hw = d_params(B, s, tau)
            roots = quartic_roots(n, d1, d2, d3, d4, t)
            for lab, val in zip(['--', '-+', '+-', '++'], roots):
                curve[lab][ib] = val * hw
        for mu in per_n:
            per_n[mu].append(curve[mu])

    return out, per_n


def all_conduction_energies(B):
    energies = []
    for tau in (1, -1):
        for s in (1, -1):
            energies.append(n_minus1_level(s, tau, B))
            lvl0 = n0_levels_eV(s, tau, B)
            labels0 = ['--', '-+', '++'] if tau == -1 else ['--', '-+', '+-']
            cond_label = '++' if tau == -1 else '+-'
            energies.append(lvl0[labels0.index(cond_label)])
            d1, d2, d3, d4, t, hw = d_params(B, s, tau)
            for n in range(1, N_EF + 1):
                roots = quartic_roots(n, d1, d2, d3, d4, t) * hw
                energies.extend(roots[2:])
    return np.array(energies)


def fermi_energy_curve(B_values, ne_target_m2=1.9e17, T=1.0):
    from scipy.optimize import brentq
    kBT = 8.617333262e-5 * T
    EF = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 40 == 0:
            print(f"  fermi_energy_curve: point {i}/{len(B_values)} (B={B:.2f} T)...")
        E = all_conduction_energies(B)
        l_B2 = hbar_J / (e_ch * B)
        D0 = 2 * np.pi * l_B2

        def ne_of_EF(ef):
            x = np.clip((E - ef) / kBT, -500, 500)
            return np.sum(1.0 / (1.0 + np.exp(x))) / D0

        lo, hi = E.min() - 0.05, E.max() + 0.05
        EF[i] = brentq(lambda ef: ne_of_EF(ef) - ne_target_m2, lo, hi, xtol=1e-10)
    return EF


def legend_entry(ax, x, y, ls, color, text, fontsize=8):
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.add_patch(plt.Rectangle((x - 0.01 * xr, y - 0.018 * yr), 0.16 * xr, 0.036 * yr,
                                facecolor='white', edgecolor='none', zorder=5))
    ax.plot([x, x + 0.05 * xr], [y, y], color=color, linestyle=ls, lw=1.5,
            zorder=6, clip_on=False)
    ax.text(x + 0.065 * xr, y, text, color='black', fontsize=fontsize,
            va='center', ha='left', zorder=6)


def plot_panel(ax, mu_keys, color_map, ylabel):
    for tau in (1, -1):
        for s in (+1, -1):
            print(f"  collecting LL branch tau={tau:+d} s={s:+d}...")
            n_m1_and_0, per_n = collect_branch(s, tau)
            ls = '-' if s == 1 else ':'
            for mu in mu_keys:
                color = color_map[tau][mu]
                ax.plot(B_grid, n_m1_and_0[mu] * 10, color=color, lw=0.8, linestyle=ls)
                for curve in per_n[mu]:
                    ax.plot(B_grid, curve * 10, color=color, lw=0.6, linestyle=ls)
    ax.set_xlabel(r"$B$ (T)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 40)


print("Fig 5: starting computation...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

print("Conduction-band panel:")
plot_panel(axes[0], ['+-', '++'], COND_COLOR, r"$E$ ($10^{-1}$ eV)")
axes[0].set_ylim(8.0, 8.6)
axes[0].set_title("Conduction band")

print("Fermi energy curve:")
B_ef = B_grid[B_grid >= 1.0]
EF_curve = fermi_energy_curve(B_ef)
axes[0].plot(B_ef, EF_curve * 10, color='magenta', lw=1.3, label=r"$E_F$")
axes[0].legend(loc='upper left', fontsize=8, frameon=False)
axes[0].text(1, 8.44, "V = 15 meV", fontsize=8, va='center', ha='left', zorder=6,
             bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

# Conduction-panel legend: one bottom block, 4 colour-columns x 2 rows
# (solid/dotted), matching Fig. 5's layout in the reference PDF image.
legend_entry(axes[0], 1, 8.115, '-', 'red', r"$E^{\uparrow,+}_{n,+-}$")
legend_entry(axes[0], 1, 8.055, ':', 'red', r"$E^{\downarrow,+}_{n,+-}$")
legend_entry(axes[0], 12, 8.115, '-', 'black', r"$E^{\uparrow,+}_{n,++}$")
legend_entry(axes[0], 12, 8.055, ':', 'black', r"$E^{\downarrow,+}_{n,++}$")
legend_entry(axes[0], 23, 8.115, '-', 'green', r"$E^{\uparrow,-}_{n,++}$")
legend_entry(axes[0], 23, 8.055, ':', 'green', r"$E^{\downarrow,-}_{n,++}$")
legend_entry(axes[0], 32, 8.115, '-', 'blue', r"$E^{\uparrow,-}_{n,+-}$")
legend_entry(axes[0], 32, 8.055, ':', 'blue', r"$E^{\downarrow,-}_{n,+-}$")

print("Valence-band panel:")
plot_panel(axes[1], ['-+', '--'], VAL_COLOR, r"$E$ ($10^{-1}$ eV)")
axes[1].set_ylim(-9.6, -7.4)
axes[1].set_title("Valence band")
axes[1].text(25, -8.60, "V = 15 meV", fontsize=8, va='center', ha='left', zorder=6,
             bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

# Valence-panel legend: pixel-sampling the reference PDF confirms the
# blue/green swatches DO sit on an opaque white backing box that masks the
# dense curve fan underneath (verified: pure white pixels sampled right
# through where fan lines should cross at that height) -- so box=True
# (matplotlib's normal opaque legend look) is correct here, same as the
# conduction panel.
legend_entry(axes[1], 1, -7.95, '-', 'red', r"$E^{\uparrow,+}_{n,-+}$")
legend_entry(axes[1], 1, -8.15, ':', 'red', r"$E^{\downarrow,+}_{n,-+}$")
legend_entry(axes[1], 1, -8.40, '-', 'black', r"$E^{\uparrow,+}_{n,--}$")
legend_entry(axes[1], 1, -8.60, ':', 'black', r"$E^{\downarrow,+}_{n,--}$")
legend_entry(axes[1], 13, -7.95, '-', 'blue', r"$E^{\uparrow,-}_{n,-+}$")
legend_entry(axes[1], 13, -8.15, ':', 'blue', r"$E^{\downarrow,-}_{n,-+}$")
legend_entry(axes[1], 13, -8.40, '-', 'green', r"$E^{\uparrow,-}_{n,--}$")
legend_entry(axes[1], 13, -8.60, ':', 'green', r"$E^{\downarrow,-}_{n,--}$")

fig.suptitle(
    r"Energy spectrum of bilayer MoS$_2$ vs $B$ for $M_z,M_v \neq 0$, $V=15$ meV.",
    fontsize=10
)
fig.tight_layout()

plt.savefig("bilayer_MoS2_fig5_draft.png", dpi=200)
print("Saved: bilayer_MoS2_fig5_draft.png")
plt.show()
