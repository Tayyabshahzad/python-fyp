# Bilayer MoS2 Fig. 4 -- LL "stick diagram" at fixed B=30 T, V=0, comparing
# Mz,Mv=0 (left panels) vs Mz,Mv!=0 (right... no: TOP row) -- see layout
# note below. Reuses the exact same quartic/cubic/n=-1 solver verified in
# Codes/2.py (Fig. 3), just evaluated at one B instead of swept over B.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

hbar_J = 1.054571817e-34
e_ch = 1.602176634e-19
muB = 5.7883818060e-5

vF = 0.53e6
Delta = 0.83
lam = 0.074
gamma = 0.047
V = 0.0

g_e, g_s, g_v = 2.0, 0.21, 3.57
gprime = g_e + g_s

B0 = 30.0     # T -- fixed for the whole figure
N_MAX = 6     # highest LL index shown (matches the reference's label range)
DEGEN_TOL = 7e-4   # eV -- levels closer than this get merged into one
                   # drawn line with a combined ";"-joined label (the paper
                   # explicitly documents several such "unobservable
                   # splitting" degeneracies between adjacent (n,mu) pairs).


def hw_eV(B):
    omega_c = vF * np.sqrt(2 * e_ch * B / hbar_J)
    return hbar_J * omega_c / e_ch


def Mz_eV(B, zeeman_on):
    return gprime * muB * B / 2.0 if zeeman_on else 0.0


def Mv_eV(B, zeeman_on):
    return g_v * muB * B / 2.0 if zeeman_on else 0.0


def xi_terms(s, tau, B, zeeman_on):
    kappa, alpha = Delta + V, Delta - V
    Mz, Mv = Mz_eV(B, zeeman_on), Mv_eV(B, zeeman_on)
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def d_params(B, s, tau, zeeman_on):
    hw = hw_eV(B)
    kappa_tau, alpha_tau = (Delta + tau * V) / hw, (Delta - tau * V) / hw
    lam_hw, t = lam / hw, gamma / hw
    Mz, Mv = Mz_eV(B, zeeman_on), Mv_eV(B, zeeman_on)
    Z = tau * (s * Mz - tau * Mv) / hw
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


def n_minus1_level(s, tau, B, zeeman_on):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, zeeman_on)
    return xi4 if tau == 1 else xi2


def n0_levels_eV(s, tau, B, zeeman_on):
    xi1, xi2, xi3, xi4 = xi_terms(s, tau, B, zeeman_on)
    hw = hw_eV(B)
    if tau == 1:
        H = np.array([[-xi1, gamma, 0.0], [gamma, -xi3, hw], [0.0, hw, xi4]])
    else:
        H = np.array([[-xi1, hw, gamma], [hw, xi2, 0.0], [gamma, 0.0, -xi3]])
    return np.sort(np.linalg.eigvalsh(H))


def collect_levels(B, zeeman_on):
    """All (E, n, mu, s, tau) tuples for n=-1..N_MAX, both spins, both
    valleys, at fixed B."""
    levels = []
    for tau in (1, -1):
        for s in (1, -1):
            levels.append((n_minus1_level(s, tau, B, zeeman_on), -1,
                            '++' if tau == 1 else '+-', s, tau))
            lvl0 = n0_levels_eV(s, tau, B, zeeman_on)
            labels0 = ['--', '-+', '++'] if tau == -1 else ['--', '-+', '+-']
            for lab, val in zip(labels0, lvl0):
                levels.append((val, 0, lab, s, tau))
            d1, d2, d3, d4, t, hw = d_params(B, s, tau, zeeman_on)
            for n in range(1, N_MAX + 1):
                roots = quartic_roots(n, d1, d2, d3, d4, t) * hw
                for lab, val in zip(['--', '-+', '+-', '++'], roots):
                    levels.append((val, n, lab, s, tau))
    return levels


SPIN_ARROW = {1: r"\uparrow", -1: r"\downarrow"}
# tau=+1 text colour is red for mu="+-" and black for mu="++" (conduction)
# or red for "-+" (valence, second-layer "--" branch omitted, per the
# paper's explicit note); tau=-1 is blue/green, matching Figs. 1-3.
TEXT_COLOR_4C = {
    (1, '+-'): 'red', (1, '++'): 'black', (1, '-+'): 'red',
    (-1, '+-'): 'blue', (-1, '++'): 'green', (-1, '-+'): 'blue',
}
# At Mz=Mv=0 the two valleys are numerically identical (the LL SOC term has
# no tau dependence), so the reference reuses just red/black (by mu) for
# BOTH columns instead of introducing blue/green -- verified against the
# actual PDF page image, not guessed.
TEXT_COLOR_FLAT = {
    (1, '+-'): 'red', (1, '++'): 'black', (1, '-+'): 'red',
    (-1, '+-'): 'red', (-1, '++'): 'black', (-1, '-+'): 'red',
}


def cluster(levels_subset):
    """Sort by energy and merge entries within DEGEN_TOL into one group."""
    items = sorted(levels_subset, key=lambda r: r[0])
    groups = []
    for it in items:
        if groups and abs(it[0] - groups[-1][0][0]) < DEGEN_TOL:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def label_of(n, mu, s):
    return fr"$({n},{mu},{SPIN_ARROW[s]})$"


MIN_GAP = 0.0012  # eV -- minimum visually-distinct spacing between lines
                   # inside one cluster. Even when two (or four) levels are
                   # EXACTLY degenerate (e.g. n=-1 at Mz=Mv=0, where all 4
                   # combinations are mathematically identical), the
                   # reference always draws them as separate close-but-
                   # distinct lines -- never collapsed into one. This
                   # nudges apart anything closer than MIN_GAP while
                   # preserving true relative order and the true mean.


def spread_positions(items_sorted):
    """items_sorted: list of (E,...) tuples, ascending by E. Returns a
    list of display-E's, same order, with every consecutive gap >=
    MIN_GAP, re-centered on the true mean so the visual nudge doesn't
    drift the whole cluster."""
    n = len(items_sorted)
    if n <= 1:
        return [it[0] for it in items_sorted]
    true_E = [it[0] for it in items_sorted]
    disp = list(true_E)
    for i in range(1, n):
        if disp[i] - disp[i - 1] < MIN_GAP:
            disp[i] = disp[i - 1] + MIN_GAP
    shift = np.mean(true_E) - np.mean(disp)
    return [d + shift for d in disp]


def draw_panel(ax, levels, mu_keep, annotations, flat_color=False):
    """levels: list of (E,n,mu,s,tau). mu_keep: set of mu labels to draw
    (conduction panels keep both branches; valence panels keep only '-+',
    since 'the second-layer valence levels are not shown', per caption).

    Clustering is done PER VALLEY (the paper's documented "adjacent LL"
    degeneracies, e.g. E^{s,+}_{n+1,+-}=E^{s,+}_{n,++}, are within-valley
    -- the LL SOC term has no explicit tau dependence, so at Mz=Mv=0 the
    two valleys become numerically identical, but they are still drawn/
    labeled as two separate columns, matching the reference. Only n=-1
    (which has no independent within-valley partner) gets cross-valley
    merged when it is genuinely coincident.)"""
    TEXT_COLOR = TEXT_COLOR_FLAT if flat_color else TEXT_COLOR_4C
    left = cluster([r for r in levels if r[2] in mu_keep and r[4] == 1])
    right = cluster([r for r in levels if r[2] in mu_keep and r[4] == -1])

    # merge the n=-1 group across valleys if they coincide (only happens
    # for Mz=Mv=0, V=0, where the LL Hamiltonian is exactly valley-symmetric)
    left_m1 = next((g for g in left if g[0][1] == -1), None)
    right_m1 = next((g for g in right if g[0][1] == -1), None)
    merged_m1 = None
    if left_m1 and right_m1 and abs(left_m1[0][0] - right_m1[0][0]) < DEGEN_TOL:
        merged_m1 = left_m1 + right_m1
        left, right = [g for g in left if g is not left_m1], [g for g in right if g is not right_m1]

    def draw_side(groups, tau_g, ha, x_line, x_text):
        for grp in groups:
            grp_sorted = sorted(grp, key=lambda r: r[0])
            disp = spread_positions(grp_sorted)
            for r, E in zip(grp_sorted, disp):
                style = '-' if r[3] == 1 else ':'
                ax.plot(x_line, [E, E], transform=ax.get_yaxis_transform(),
                        color=TEXT_COLOR[(tau_g, r[2])], linestyle=style,
                        lw=1.0, clip_on=False)
            E_center = np.mean(disp)
            label_txt = "; ".join(label_of(r[1], r[2], r[3]) for r in grp_sorted)
            ax.text(x_text, E_center, label_txt, transform=ax.get_yaxis_transform(),
                    color=TEXT_COLOR[(tau_g, grp_sorted[0][2])], fontsize=4.0,
                    ha=ha, va='center')

    draw_side(left, 1, 'right', (0.20, 0.40), 0.185)
    draw_side(right, -1, 'left', (0.60, 0.80), 0.815)

    if merged_m1:
        grp_sorted = sorted(merged_m1, key=lambda r: r[0])
        disp = spread_positions(grp_sorted)
        for r, E in zip(grp_sorted, disp):
            style = '-' if r[3] == 1 else ':'
            ax.plot([0.20, 0.80], [E, E], transform=ax.get_yaxis_transform(),
                    color='black', linestyle=style, lw=1.0, clip_on=False)
        E_center = np.mean(disp)
        label_txt = "; ".join(label_of(r[1], r[2], r[3]) for r in grp_sorted)
        ax.text(0.5, E_center, label_txt, transform=ax.get_yaxis_transform(),
                color='black', fontsize=4.0, ha='center', va='center')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():          # bold border on all 4 sides
        spine.set_visible(True)
        spine.set_linewidth(1.6)
    ax.annotate('', xy=(0.045, 0.82), xytext=(0.045, 0.10), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2))
    ax.text(0.045, 0.94, "E", transform=ax.transAxes, fontsize=11, ha='center', va='bottom')
    for x, y, txt, fs, ha, va in annotations:
        ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs, ha=ha, va=va)


print("Fig 4: starting computation...")
fig, axes = plt.subplots(2, 2, figsize=(15, 18))

print("  collecting levels (Mz,Mv=0)...")
lev_z0 = collect_levels(B0, zeeman_on=False)
print("  collecting levels (Mz,Mv!=0)...")
lev_zN = collect_levels(B0, zeeman_on=True)
print("  levels collected, building panels...")

# Header/footer text, positioned to match each of the 4 reference panels
# individually (their layouts are NOT identical to each other).
ann_panel1 = [  # top-left: conduction, Mz,Mv=0
    (0.30, 0.94, r"$M_z,M_v=0$", 10, 'center', 'bottom'),
    (0.20, 0.87, r"$\tau=+1$", 10, 'center', 'bottom'),
    (0.5, 0.94, "V = 0 meV", 9, 'center', 'bottom'),
    (0.80, 0.87, r"$\tau=-1$", 10, 'center', 'bottom'),
    (0.5, 0.015, "B = 30 T", 9, 'center', 'bottom'),
]
ann_panel2 = [  # top-right: conduction, Mz,Mv!=0
    (0.5, 0.94, r"$M_z,M_v\neq 0$", 10, 'center', 'bottom'),
    (0.29, 0.87, r"$\tau=+1$", 10, 'center', 'bottom'),
    (0.71, 0.87, r"$\tau=-1$", 10, 'center', 'bottom'),
    (0.92, 0.94, "B = 30 T", 9, 'right', 'bottom'),
    (0.5, 0.015, "V = 0 meV", 9, 'center', 'bottom'),
]
ann_panel3 = [  # bottom-left: valence, Mz,Mv=0
    (0.15, 0.94, r"$\tau=+1$", 10, 'center', 'bottom'),
    (0.85, 0.94, r"$\tau=-1$", 10, 'center', 'bottom'),
    (0.5, 0.94, "V = 0 meV", 9, 'center', 'bottom'),
    (0.5, 0.87, "B = 30 T", 9, 'center', 'bottom'),
]
ann_panel4 = [  # bottom-right: valence, Mz,Mv!=0
    (0.15, 0.94, r"$\tau=+1$", 10, 'center', 'bottom'),
    (0.60, 0.94, r"$\tau=-1$", 10, 'center', 'bottom'),
    (0.94, 0.94, "B = 30 T", 9, 'right', 'bottom'),
    (0.5, 0.90, "V = 0 meV", 9, 'center', 'bottom'),
]

draw_panel(axes[0, 0], lev_z0, {'+-', '++'}, ann_panel1, flat_color=True)
draw_panel(axes[0, 1], lev_zN, {'+-', '++'}, ann_panel2, flat_color=False)
draw_panel(axes[1, 0], lev_z0, {'-+'}, ann_panel3, flat_color=True)
draw_panel(axes[1, 1], lev_zN, {'-+'}, ann_panel4, flat_color=False)

for ax, ylo, yhi in [(axes[0, 0], 0.815, 0.905), (axes[0, 1], 0.815, 0.905),
                      (axes[1, 0], -0.805, -0.720), (axes[1, 1], -0.805, -0.720)]:
    ax.set_ylim(ylo, yhi)

fig.suptitle(
    r"LL spectrum of bilayer MoS$_2$ at $B=30$ T and $V=0$, labeled by $(n,\mu,s)$.",
    fontsize=11
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("bilayer_MoS2_fig4_draft.png", dpi=200)
print("Saved: bilayer_MoS2_fig4_draft.png")
plt.show()
