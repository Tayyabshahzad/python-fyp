# Bilayer MoS2 Fig. 2 -- band structure vs electric field V (3D waterfall plot)
# Requirements: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

# ---------- Parameters (identical to Codes/0.py, Fig. 1) ----------
hbar = 6.582119569e-16  # eV*s
a = 3.16e-10            # m (lattice constant, same assumption as Fig. 1)
vF = 0.53e6             # m/s
Delta = 0.83            # eV (since 2*Delta = 1.66 eV)
lam = 0.074             # eV (lambda)
gamma = 0.047           # eV (gamma)
Mz = 0.0
Mv = 0.0

# Dimensionless momentum grid q = ka/pi (same range/resolution as Fig. 1)
q = np.linspace(-0.1, 0.1, 201)
k = (np.pi / a) * q
i0 = np.argmin(np.abs(q))

plt.rcParams['font.family'] = 'serif'

# The four V "slices" drawn in Fig. 2 (exact tick values on the paper's V axis)
V_SLICES = [0.0, 0.005, 0.01, 0.015]

tau = 1  # Fig. 2 shows a single valley's curves (K), same convention as Fig. 1's K panel


def xi_terms(tau, s, V):
    """xi_i^{s,tau} as in the paper (eV). Identical to Codes/0.py."""
    kappa = Delta + V
    alpha = Delta - V
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def H_matrix(kx, tau, s, V):
    xi1, xi2, xi3, xi4 = xi_terms(tau, s, V)
    t = vF * hbar * abs(kx)
    return np.array([
        [-xi1, t, gamma, 0.0],
        [t, xi2, 0.0, 0.0],
        [gamma, 0.0, -xi3, t],
        [0.0, 0.0, t, xi4]
    ], dtype=float)


def bands(tau, V):
    """Sorted eigenvalues for spin up/down at each k. Returns (Eup, Edn),
    each shape (len(k), 4), ascending: [0,1]=valence, [2,3]=conduction."""
    Eup = np.zeros((len(k), 4))
    Edn = np.zeros((len(k), 4))
    for i, kx in enumerate(k):
        Eup[i] = np.sort(np.linalg.eigvalsh(H_matrix(kx, tau, +1, V)))
        Edn[i] = np.sort(np.linalg.eigvalsh(H_matrix(kx, tau, -1, V)))
    return Eup, Edn


# Fixed band <-> colour map, matching Fig. 1's K-panel convention (verified
# against the PDF in that figure): conduction outer(b=3)=black inner(b=2)=red;
# valence lower(b=0)=black upper(b=1)=red. No eigenvector heuristic is needed
# here since within -0.1<=q<=0.1 the bands never cross.
COND_COLOR = {2: 'red', 3: 'black'}
VAL_COLOR = {0: 'black', 1: 'red'}

K_LEFT, K_RIGHT = q[0], q[-1]


def draw_axes(ax, e_bottom, e_top, e_ticks, e_tick_fmt, k_ticks):
    """Manually build the paper's 3-edge axis frame, anchored exactly at
    the data's own corner point (k=-0.1, V=0), instead of relying on
    mplot3d's default bounding-box axes (which float away from the data
    and don't match the paper's hand-drawn look)."""
    lw = 1.8
    span = e_top - e_bottom

    # ka/pi axis: horizontal, at the front (V=0), baseline e_bottom, with a
    # dense comb of minor ticks plus labeled major ticks (matches the
    # paper's ruler-style axis, not just 3 bare ticks).
    ax.plot([K_LEFT, K_RIGHT], [0, 0], [e_bottom, e_bottom], color='black', lw=lw)
    for kt in np.linspace(K_LEFT, K_RIGHT, 21):
        ax.plot([kt, kt], [0, 0], [e_bottom, e_bottom - 0.006 * span], color='black', lw=1.0)
    for kt in k_ticks:
        ax.plot([kt, kt], [0, 0], [e_bottom, e_bottom - 0.012 * span], color='black', lw=lw)
        ax.text(kt, -0.006, e_bottom - 0.035 * span, f"{kt:.1f}",
                ha='center', va='top', fontsize=9)
    ax.text(0.0, 0.0, e_bottom - 0.10 * span, r"$ka/\pi$",
            ha='center', va='top', fontsize=10)

    # E axis: vertical, at the front-left corner (k=-0.1, V=0), same
    # dense-minor-tick ruler style.
    ax.plot([K_LEFT, K_LEFT], [0, 0], [e_bottom, e_top], color='black', lw=lw)
    for et in np.linspace(e_bottom, e_top, 21):
        ax.plot([K_LEFT, K_LEFT - 0.0025], [0, 0], [et, et], color='black', lw=1.0)
    for et in e_ticks:
        ax.plot([K_LEFT, K_LEFT - 0.005], [0, 0], [et, et], color='black', lw=lw)
        ax.text(K_LEFT - 0.014, 0, et, e_tick_fmt(et), ha='right', va='center', fontsize=9)
    ax.text(K_LEFT - 0.075, 0, e_bottom + 0.35 * span, "E (eV)",
            ha='center', va='center', fontsize=10, rotation=90)

    # V axis: diagonal, from the same left corner but at the ceiling e_top
    ax.plot([K_LEFT, K_LEFT], [0, V_SLICES[-1]], [e_top, e_top], color='black', lw=lw)
    for vt in V_SLICES:
        label = f"{vt:g}" if vt else "0"
        gap = 0.028 * span if vt else 0.012 * span
        ax.text(K_LEFT - 0.018, vt, e_top + gap, label,
                ha='right', va='bottom', fontsize=9)
    ax.text(K_LEFT - 0.11, V_SLICES[-1] * 0.60, e_top + 0.09 * span, "V (eV)",
            ha='center', va='bottom', fontsize=10)


def draw_waterfall(ax, band_indices, color_map, e_top):
    """Draw one V-slice per entry in V_SLICES (black curves first, then
    red, matching the paper's draw order), plus an elbow-shaped dashed
    blue guide line per non-zero V: a vertical drop from the V axis down
    to that slice's own band-vertex level, then a horizontal floor line
    at that level running along k (matching the paper's guide lines)."""
    ordered = sorted(band_indices, key=lambda b: color_map[b] != 'black')
    for V in V_SLICES:
        Eup, Edn = bands(tau, V)
        ys = np.full_like(q, V)
        for b in ordered:
            color = color_map[b]
            ax.plot(q, ys, Eup[:, b], color=color, lw=1.1)
            if V != 0.0:
                ax.plot(q, ys, Edn[:, b], color=color, lw=0.8, linestyle=':')
        if V != 0.0:
            e_floor = Eup[i0, ordered[0]]
            ax.plot([K_LEFT, K_LEFT], [V, V], [e_top, e_floor],
                    color='blue', lw=1.2, linestyle='--')
            ax.plot([K_LEFT, K_RIGHT], [V, V], [e_floor, e_floor],
                    color='blue', lw=1.2, linestyle='--')


def build_panel(ax, band_indices, color_map, e_bottom, e_top, e_ticks, e_tick_fmt):
    ax.set_axis_off()
    draw_waterfall(ax, band_indices, color_map, e_top)
    draw_axes(ax, e_bottom, e_top, e_ticks, e_tick_fmt, k_ticks=[-0.1, 0.0, 0.1])
    ax.set_xlim(K_LEFT, K_RIGHT)
    ax.set_ylim(0, V_SLICES[-1])
    ax.set_zlim(e_bottom, e_top)
    ax.set_proj_type('ortho')
    ax.view_init(elev=20, azim=-110)
    ax.set_box_aspect((1.2, 2.2, 1.3))


print("Fig 2: starting computation...")
fig = plt.figure(figsize=(14, 8))

print("  building conduction-band waterfall panel...")
ax_cond = fig.add_subplot(1, 2, 1, projection='3d')
build_panel(ax_cond, [2, 3], COND_COLOR, e_bottom=0.81, e_top=0.955,
            e_ticks=[0.85, 0.90, 0.95], e_tick_fmt=lambda v: f"{v:.2f}")

print("  building valence-band waterfall panel...")
ax_val = fig.add_subplot(1, 2, 2, projection='3d')
build_panel(ax_val, [0, 1], VAL_COLOR, e_bottom=-1.01, e_top=-0.70,
            e_ticks=[-1.0, -0.9, -0.8], e_tick_fmt=lambda v: f"{v:.1f}")
print("  panels done, finalizing figure...")

fig.suptitle(
    r"Band structure of bilayer MoS$_2$ for different electric fields $E_z$. "
    r"Left: conduction band; Right: valence band.",
    fontsize=10
)

fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.22, wspace=0.15)

plt.savefig("bilayer_MoS2_fig2_matched.png", dpi=200)
print("Saved: bilayer_MoS2_fig2_matched.png")

plt.show()
