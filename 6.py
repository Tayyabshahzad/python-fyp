# Bilayer MoS2 Fig. 6 -- LLs in the conduction band vs B for V = 0 meV.
# Left panel: Mz = Mv = 0 (valley-degenerate). Right panel: Mz,Mv != 0
# (valley degeneracy lifted). Magenta curve: EF vs B from Eq. (17).
# Identical solver to Codes/2.py (Fig. 3) / Codes/4.py (Fig. 5); only V is
# fixed at 0 here and the Zeeman terms are toggled on/off per panel.
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
V = 0.0                      # eV -- Fig. 6 is V = 0 meV

g_e, g_s, g_v = 2.0, 0.21, 3.57
gprime = g_e + g_s

N_MAX = 24
N_EF = 150
B_grid = np.linspace(0.4, 40, 240)


def omega_c(B):
    return vF * np.sqrt(2 * e_ch * B / hbar_J)


def hw_eV(B):
    return hbar_J * omega_c(B) / e_ch


def Mz_eV(B, zeeman):
    return gprime * muB * B / 2.0 if zeeman else 0.0


def Mv_eV(B, zeeman):
    return g_v * muB * B / 2.0 if zeeman else 0.0


def d_params(B, s, tau, zeeman):
    """Dimensionless d1..d4 and t, all in units of hbar*omega_c."""
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


def xi_terms(s, tau, B, zeeman):
    """Original Eq. (1) xi's (eV)."""
    kappa = Delta + V
    alpha = Delta - V
    Mz, Mv = Mz_eV(B, zeeman), Mv_eV(B, zeeman)
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def n_minus1_level(s, tau, B, zeeman):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, zeeman)
    return xi4 if tau == 1 else xi2


def n0_levels_eV(s, tau, B, zeeman):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, zeeman)
    hw = hw_eV(B)
    if tau == 1:
        H = np.array([[-xi1, gamma, 0.0], [gamma, -xi3, hw], [0.0, hw, xi4]])
    else:
        H = np.array([[-xi1, hw, gamma], [hw, xi2, 0.0], [gamma, 0.0, -xi3]])
    return np.sort(np.linalg.eigvalsh(H))


COND_COLOR = {1: {'+-': 'red', '++': 'black'}, -1: {'+-': 'blue', '++': 'green'}}


def collect_branch(s, tau, zeeman):
    out = {mu: np.full(len(B_grid), np.nan) for mu in ['--', '-+', '+-', '++']}
    for ib, B in enumerate(B_grid):
        n_m1 = n_minus1_level(s, tau, B, zeeman)
        out['++' if tau == 1 else '+-'][ib] = n_m1

        lvl0 = n0_levels_eV(s, tau, B, zeeman)
        labels0 = ['--', '-+', '++'] if tau == -1 else ['--', '-+', '+-']
        for lab, val in zip(labels0, lvl0):
            out[lab][ib] = val

    per_n = {mu: [] for mu in ['--', '-+', '+-', '++']}
    for n in range(1, N_MAX + 1):
        curve = {mu: np.full(len(B_grid), np.nan) for mu in ['--', '-+', '+-', '++']}
        for ib, B in enumerate(B_grid):
            d1, d2, d3, d4, t, hw = d_params(B, s, tau, zeeman)
            roots = quartic_roots(n, d1, d2, d3, d4, t)
            for lab, val in zip(['--', '-+', '+-', '++'], roots):
                curve[lab][ib] = val * hw
        for mu in per_n:
            per_n[mu].append(curve[mu])

    return out, per_n


def all_conduction_energies(B, zeeman):
    energies = []
    for tau in (1, -1):
        for s in (1, -1):
            energies.append(n_minus1_level(s, tau, B, zeeman))
            lvl0 = n0_levels_eV(s, tau, B, zeeman)
            labels0 = ['--', '-+', '++'] if tau == -1 else ['--', '-+', '+-']
            cond_label = '++' if tau == -1 else '+-'
            energies.append(lvl0[labels0.index(cond_label)])
            d1, d2, d3, d4, t, hw = d_params(B, s, tau, zeeman)
            for n in range(1, N_EF + 1):
                roots = quartic_roots(n, d1, d2, d3, d4, t) * hw
                energies.extend(roots[2:])
    return np.array(energies)


def fermi_energy_curve(B_values, zeeman, ne_target_m2=1.9e17, T=1.0):
    from scipy.optimize import brentq
    kBT = 8.617333262e-5 * T
    EF = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 40 == 0:
            print(f"  fermi_energy_curve: point {i}/{len(B_values)} (B={B:.2f} T)...")
        E = all_conduction_energies(B, zeeman)
        l_B2 = hbar_J / (e_ch * B)
        D0 = 2 * np.pi * l_B2

        def ne_of_EF(ef):
            x = np.clip((E - ef) / kBT, -500, 500)
            return np.sum(1.0 / (1.0 + np.exp(x))) / D0

        lo, hi = E.min() - 0.05, E.max() + 0.05
        EF[i] = brentq(lambda ef: ne_of_EF(ef) - ne_target_m2, lo, hi, xtol=1e-10)
    return EF


def legend_entry(ax, x, y, ls, color, text, fontsize=8, width_scale=0.20):
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.add_patch(plt.Rectangle((x - 0.01 * xr, y - 0.018 * yr), width_scale * xr, 0.036 * yr,
                                facecolor='white', edgecolor='none', zorder=5))
    ax.plot([x, x + 0.05 * xr], [y, y], color=color, linestyle=ls, lw=1.5,
            zorder=6, clip_on=False)
    ax.text(x + 0.065 * xr, y, text, color='black', fontsize=fontsize,
            va='center', ha='left', zorder=6)


def plot_branch(ax, s, tau, zeeman, mu_keys, color_map):
    print(f"  collecting LL branch tau={tau:+d} s={s:+d} zeeman={zeeman}...")
    n_m1_and_0, per_n = collect_branch(s, tau, zeeman)
    ls = '-' if s == 1 else ':'
    for mu in mu_keys:
        color = color_map[tau][mu]
        ax.plot(B_grid, n_m1_and_0[mu] * 10, color=color, lw=0.8, linestyle=ls)
        for curve in per_n[mu]:
            ax.plot(B_grid, curve * 10, color=color, lw=0.6, linestyle=ls)


print("Fig 6: starting computation...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ---------------- Left panel: Mz = Mv = 0 (valley-degenerate) ----------------
ax = axes[0]
for s in (+1, -1):
    plot_branch(ax, s, 1, False, ['+-', '++'], COND_COLOR)
ax.set_xlim(0, 40)
ax.set_ylim(8.295, 8.555)
ax.set_xlabel(r"$B$ (T)")
ax.set_ylabel(r"$E$ ($10^{-1}$ eV)")
ax.set_title("Conduction band")

print("Fermi energy curve (left panel):")
B_ef = B_grid[B_grid >= 1.0]
EF0 = fermi_energy_curve(B_ef, False)
ax.plot(B_ef, EF0 * 10, color='magenta', lw=1.3)
ax.text(1, 8.545, r"$E_F$", fontsize=9, va='center', ha='left', color='black')
ax.plot([1, 4], [8.548, 8.548], color='magenta', lw=1.3, clip_on=False)
ax.text(29, 8.505, "V = 0 meV", fontsize=8, va='center', ha='left')

legend_entry(ax, 22, 8.395, '-', 'red', r"$E^{\uparrow,+}_{n,+-}=E^{\uparrow,-}_{n,+-}$")
legend_entry(ax, 22, 8.352, ':', 'red', r"$E^{\downarrow,+}_{n,+-}=E^{\downarrow,-}_{n,+-}$")
legend_entry(ax, 9, 8.313, ':', 'black', r"$E^{\downarrow,+}_{n,++}=E^{\downarrow,-}_{n,++}$")
legend_entry(ax, 27, 8.313, '-', 'black', r"$E^{\uparrow,+}_{n,++}=E^{\uparrow,-}_{n,++}$")

# ---------------- Right panel: Mz,Mv != 0 (valley degeneracy lifted) --------
ax = axes[1]
for tau in (1, -1):
    for s in (+1, -1):
        plot_branch(ax, s, tau, True, ['+-', '++'], COND_COLOR)
ax.set_xlim(0, 40)
ax.set_ylim(8.205, 8.555)
ax.set_xlabel(r"$B$ (T)")
ax.set_ylabel(r"$E$ ($10^{-1}$ eV)")
ax.set_title("Conduction band")

print("Fermi energy curve (right panel):")
EF1 = fermi_energy_curve(B_ef, True)
ax.plot(B_ef, EF1 * 10, color='magenta', lw=1.3)
ax.text(1, 8.545, r"$E_F$", fontsize=9, va='center', ha='left', color='black')
ax.plot([1, 4], [8.548, 8.548], color='magenta', lw=1.3, clip_on=False)
ax.text(34, 8.228, "V = 0 meV", fontsize=8, va='center', ha='left',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

legend_entry(ax, 30, 8.352, '-', 'blue', r"$E^{\uparrow,-}_{n,+-}$")
legend_entry(ax, 30, 8.318, ':', 'blue', r"$E^{\downarrow,-}_{n,+-}$")
# Row 1/row 2 are NOT a symmetric solid/dotted grid in the reference --
# pixel-verified against the PDF: row 1 = red-solid, green-dotted,
# green-solid; row 2 = red-dotted, black-solid, black-dotted, "V=0meV".
legend_entry(ax, 1, 8.258, '-', 'red', r"$E^{\uparrow,+}_{n,+-}$", width_scale=0.11)
legend_entry(ax, 17, 8.258, ':', 'green', r"$E^{\downarrow,-}_{n,++}$", width_scale=0.11)
legend_entry(ax, 27, 8.258, '-', 'green', r"$E^{\uparrow,-}_{n,++}$", width_scale=0.11)
legend_entry(ax, 1, 8.228, ':', 'red', r"$E^{\downarrow,+}_{n,+-}$", width_scale=0.11)
legend_entry(ax, 12, 8.228, '-', 'black', r"$E^{\uparrow,+}_{n,++}$", width_scale=0.11)
legend_entry(ax, 23, 8.228, ':', 'black', r"$E^{\downarrow,+}_{n,++}$", width_scale=0.11)

fig.suptitle(
    r"LLs in bilayer MoS$_2$ (conduction band) vs $B$ for $V=0$ meV. "
    r"Left: $M_z=M_v=0$; right: $M_z\neq M_v\neq 0$. Magenta: $E_F$ vs $B$.",
    fontsize=10
)
fig.tight_layout()

plt.savefig("bilayer_MoS2_fig6_draft.png", dpi=200)
print("Saved: bilayer_MoS2_fig6_draft.png")
plt.show()
