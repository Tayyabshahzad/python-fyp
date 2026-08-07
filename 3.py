# Bilayer MoS2 Fig. 3 -- Landau-level spectrum vs magnetic field B, V=0,
# with Mz, Mv (spin/valley Zeeman) turned ON -- per the paper's caption
# "for Mz, Mv != 0, and V = 0" (verified against the actual PDF page image;
# an earlier plain-text extraction had mis-OCR'd this as "Mz,Mv = 0").
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

# ---------- Parameters (same physical constants as Fig. 1/2) ----------
hbar_eVs = 6.582119569e-16   # eV*s
hbar_J = 1.054571817e-34     # J*s
e_ch = 1.602176634e-19       # C
muB = 5.7883818060e-5        # eV/T  (Bohr magneton)

vF = 0.53e6                  # m/s
Delta = 0.83                 # eV
lam = 0.074                  # eV
gamma = 0.047                # eV
V = 0.0                      # eV (Fig. 3 is V=0)

g_e, g_s, g_v = 2.0, 0.21, 3.57
gprime = g_e + g_s            # 2.21

N_MAX = 24                    # highest LL index plotted (n=1..N_MAX)
N_EF = 150                    # higher LL cutoff used only for the E_F self-
                               # consistency sum -- needs many more filled
                               # states than we bother drawing, especially
                               # at low B where D0=2*pi*l_B^2 is large.
B_grid = np.linspace(0.4, 40, 240)   # T (start slightly above 0 -- omega_c=0 there)


def omega_c(B):
    return vF * np.sqrt(2 * e_ch * B / hbar_J)          # rad/s


def hw_eV(B):
    return hbar_J * omega_c(B) / e_ch                    # hbar*omega_c, in eV


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
    """Eq. (5): [(e+d1)(e-d2)-n][(e+d3)(e-d4)-(n+1)] - t^2(e-d2)(e-d4) = 0."""
    p1 = np.poly1d([1.0, d1 - d2, -d1 * d2 - n])
    p2 = np.poly1d([1.0, d3 - d4, -d3 * d4 - (n + 1)])
    p3 = np.poly1d([1.0, -(d2 + d4), d2 * d4])
    Q = p1 * p2 - (t ** 2) * p3
    r = np.roots(Q.coeffs)
    return np.sort(r.real)


def xi_terms(s, tau, B):
    """Original Eq. (1) xi's (eV), NOT the LL-grouped d-parameters -- the
    n=-1 and n=0 special cases (Eq. 8/9) are built directly from these, per
    the paper's own explicit n=-1 formula E=Delta+tau*s*Mz-Mv (verified
    numerically: using the grouped d2/d4 for tau=-1 gives the WRONG sign on
    the Mz/Mv terms and made the wrong branch decline with B -- this bit us
    once, don't revert to d-parameters here)."""
    kappa = Delta + V
    alpha = Delta - V
    Mz, Mv = Mz_eV(B), Mv_eV(B)
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def n_minus1_level(s, tau, B):
    """Eq. (8): a single level, built from the ORIGINAL xi2/xi4 (not d2/d4)."""
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B)
    return xi4 if tau == 1 else xi2


def n0_levels_eV(s, tau, B):
    """Eq. (9): the two valley-specific 3x3 blocks, in actual eV (real
    gamma and real hbar*omega_c(B) as couplings), diagonalized directly."""
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B)
    hw = hw_eV(B)
    if tau == 1:
        H = np.array([[-xi1, gamma, 0.0], [gamma, -xi3, hw], [0.0, hw, xi4]])
    else:
        H = np.array([[-xi1, hw, gamma], [hw, xi2, 0.0], [gamma, 0.0, -xi3]])
    return np.sort(np.linalg.eigvalsh(H))


# Colour convention (matches Fig. 1's already-approved K/K' + mu scheme):
# tau=+1 -> red("+-")/black("++") for conduction, red("-+")/black("--") valence
# tau=-1 -> blue("+-")/green("++") for conduction, blue("-+")/green("--") valence
COND_COLOR = {1: {'+-': 'red', '++': 'black'}, -1: {'+-': 'blue', '++': 'green'}}
VAL_COLOR = {1: {'-+': 'red', '--': 'black'}, -1: {'-+': 'blue', '--': 'green'}}


def collect_branch(s, tau):
    """Return dict: mu-label -> array of energies (eV) over B_grid, for
    n=-1, n=0, n=1..N_MAX, all at fixed (s, tau)."""
    out = {mu: np.full(len(B_grid), np.nan) for mu in ['--', '-+', '+-', '++']}
    for ib, B in enumerate(B_grid):
        # n = -1 (single level; label per the paper's "reserved" convention)
        n_m1 = n_minus1_level(s, tau, B)
        out['++' if tau == 1 else '+-'][ib] = n_m1

        # n = 0 (3 levels; remaining 3 of the 4 mu labels, energy-sorted)
        lvl0 = n0_levels_eV(s, tau, B)
        labels0 = ['--', '-+', '++'] if tau == -1 else ['--', '-+', '+-']
        for lab, val in zip(labels0, lvl0):
            out[lab][ib] = val

    # n >= 1 (4 levels each, per B) -- store as separate per-n curves
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
    """Every conduction-band eigenvalue (eV) at this B, across both spins,
    both valleys, and n=-1,0,1..N_MAX -- used for the self-consistent
    Fermi energy, Eq. (17)."""
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
                energies.extend(roots[2:])  # top 2 of the sorted 4 = conduction
    return np.array(energies)


def fermi_energy_curve(B_values, ne_target_m2=1.9e17, T=1.0):
    """Eq. (17): solve for E_F(B) at fixed electron density ne, via the
    Fermi-Dirac occupation sum over every conduction-band LL state."""
    from scipy.optimize import brentq
    kBT = 8.617333262e-5 * T  # eV
    EF = np.empty(len(B_values))
    for i, B in enumerate(B_values):
        if i % 40 == 0:
            print(f"  fermi_energy_curve: point {i}/{len(B_values)} (B={B:.2f} T)...")
        E = all_conduction_energies(B)
        l_B2 = hbar_J / (e_ch * B)          # magnetic length^2, m^2
        D0 = 2 * np.pi * l_B2               # m^2

        def ne_of_EF(ef):
            x = np.clip((E - ef) / kBT, -500, 500)
            return np.sum(1.0 / (1.0 + np.exp(x))) / D0

        lo, hi = E.min() - 0.05, E.max() + 0.05
        EF[i] = brentq(lambda ef: ne_of_EF(ef) - ne_target_m2, lo, hi, xtol=1e-10)
    return EF


def legend_entry(ax, x, y, ls, color, text, fontsize=8):
    """Short colored line-sample + text label, matching Fig. 1/2's
    hand-drawn legend style (data coordinates). A white backing box behind
    both the swatch and the text keeps them legible against the very
    dense LL fan (this figure has ~200 curves per panel, unlike Fig. 1) --
    without it, legend swatches/text placed anywhere inside the axes get
    visually lost among the criss-crossing lines."""
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


print("Fig 3: starting computation...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

print("Conduction-band panel:")
plot_panel(axes[0], ['+-', '++'], COND_COLOR, r"$E$ ($10^{-1}$ eV)")
axes[0].set_ylim(8.20, 8.60)
axes[0].set_title("Conduction band")

print("Fermi energy curve:")
B_ef = B_grid[B_grid >= 1.0]
EF_curve = fermi_energy_curve(B_ef)
axes[0].plot(B_ef, EF_curve * 10, color='magenta', lw=1.3, label=r"$E_F$")
axes[0].legend(loc='upper left', fontsize=8, frameon=False)

# Conduction-panel legend, positioned per the reference PDF page image.
legend_entry(axes[0], 1, 8.290, '-', 'red', r"$E^{\uparrow,+}_{n,+-}$")
legend_entry(axes[0], 1, 8.265, ':', 'red', r"$E^{\downarrow,+}_{n,+-}$")
legend_entry(axes[0], 12, 8.265, '-', 'black', r"$E^{\uparrow,+}_{n,++}$")
legend_entry(axes[0], 12, 8.235, ':', 'black', r"$E^{\downarrow,+}_{n,++}$")
legend_entry(axes[0], 23, 8.265, ':', 'green', r"$E^{\downarrow,-}_{n,++}$")
legend_entry(axes[0], 32, 8.265, '-', 'green', r"$E^{\uparrow,-}_{n,++}$")
axes[0].text(32, 8.235, "V = 0 meV", fontsize=8, va='center', ha='left', zorder=6,
             bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
legend_entry(axes[0], 29, 8.375, '-', 'blue', r"$E^{\uparrow,-}_{n,+-}$")
legend_entry(axes[0], 29, 8.320, ':', 'blue', r"$E^{\downarrow,-}_{n,+-}$")

print("Valence-band panel:")
plot_panel(axes[1], ['-+', '--'], VAL_COLOR, r"$E$ ($10^{-1}$ eV)")
axes[1].set_ylim(-9.6, -7.4)
axes[1].set_title("Valence band")

# Valence-panel legend, positioned per the reference PDF page image.
legend_entry(axes[1], 1, -7.95, '-', 'red', r"$E^{\uparrow,+}_{n,-+}$")
legend_entry(axes[1], 1, -8.15, ':', 'red', r"$E^{\downarrow,+}_{n,-+}$")
legend_entry(axes[1], 1, -8.40, '-', 'black', r"$E^{\uparrow,+}_{n,--}$")
legend_entry(axes[1], 1, -8.60, ':', 'black', r"$E^{\downarrow,+}_{n,--}$")
legend_entry(axes[1], 13, -7.95, '-', 'blue', r"$E^{\uparrow,-}_{n,-+}$")
legend_entry(axes[1], 13, -8.15, ':', 'blue', r"$E^{\downarrow,-}_{n,-+}$")
legend_entry(axes[1], 13, -8.40, '-', 'green', r"$E^{\uparrow,-}_{n,--}$")
legend_entry(axes[1], 13, -8.60, ':', 'green', r"$E^{\downarrow,-}_{n,--}$")
axes[1].text(25, -8.60, "V = 0 meV", fontsize=8, va='center', ha='left', zorder=6,
             bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

fig.suptitle(
    r"Energy spectrum of bilayer MoS$_2$ vs $B$ for $M_z,M_v \neq 0$, $V=0$.",
    fontsize=10
)
fig.tight_layout()

plt.savefig("bilayer_MoS2_fig3_draft.png", dpi=200)
print("Saved: bilayer_MoS2_fig3_draft.png")
plt.show()
