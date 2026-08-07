# Bilayer MoS2 Fig.1 style bands with annotations
# Requirements: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# ---------- Parameters from caption / text ----------
hbar = 6.582119569e-16  # eV·s
a = 3.16e-10            # m (lattice constant, editable)
vF = 0.53e6             # m/s
Delta = 0.83            # eV (since 2Δ = 1.66 eV)
lam = 0.074             # eV (λ)
gamma = 0.047           # eV (γ)
Mz = 0.0                # eV (set 0 for this figure)
Mv = 0.0                # eV (set 0 for this figure)

# Dimensionless momentum grid q = ka/π
q = np.linspace(-0.1, 0.1, 401)
k = (np.pi / a) * q
i0 = np.argmin(np.abs(q))


def xi_terms(tau, s, V):
    """xi_i^{s,tau} as in the paper (eV). tau=+1 (K), -1 (K'); s=+1 (up), -1 (down)."""
    kappa = Delta + V
    alpha = Delta - V
    xi1 = kappa + tau * s * lam + s * Mz - tau * Mv
    xi2 = alpha - s * Mz + tau * Mv
    xi3 = alpha - tau * s * lam - s * Mz + tau * Mv
    xi4 = kappa + s * Mz - tau * Mv
    return xi1, xi2, xi3, xi4


def H_matrix(kx, tau, s, V):
    """4x4 low-energy Hamiltonian along the ky cut (pi_+/pi_- -> real vF*hbar*|k|,
    which is unitarily equivalent to the paper's purely-imaginary off-diagonal
    entries and gives identical eigenvalues)."""
    xi1, xi2, xi3, xi4 = xi_terms(tau, s, V)
    t = vF * hbar * abs(kx)
    return np.array([
        [-xi1, t, gamma, 0.0],
        [t, xi2, 0.0, 0.0],
        [gamma, 0.0, -xi3, t],
        [0.0, 0.0, t, xi4]
    ], dtype=float)


def bands_with_layer(tau, s, V):
    """
    Eigen-energies at each kx, plus a 'layer colour' per band: the conduction
    pair mixes basis rows (1,3)=(xi2,xi4) and the valence pair mixes basis
    rows (0,2)=(xi1,xi3). Tagging each eigenstate 'black'/'red' by whichever
    of those two basis rows dominates its eigenvector reproduces the paper's
    two-colour scheme (black=E_{++}/E_{--}, red=E_{+-}/E_{-+}) in Fig. 1
    without needing to hand-track the mu quantum numbers.
    """
    E = np.zeros((len(k), 4))
    color = np.empty((len(k), 4), dtype=object)
    for i, kx in enumerate(k):
        w, v = np.linalg.eigh(H_matrix(kx, tau, s, V))
        order = np.argsort(w)
        E[i] = w[order]
        for j, idx in enumerate(order):
            vec = v[:, idx]
            if j < 2:  # valence pair -> mixing of rows 0 (xi1) and 2 (xi3)
                color[i, j] = 'black' if abs(vec[0]) >= abs(vec[2]) else 'red'
            else:      # conduction pair -> mixing of rows 1 (xi2) and 3 (xi4)
                color[i, j] = 'red' if abs(vec[1]) >= abs(vec[3]) else 'black'
    return E, color


def bands(tau, V):
    """Eigen-energies for s=up,down at each kx; returns (Eup, Edn, cUp, cDn)."""
    Eup, cUp = bands_with_layer(tau, +1, V)
    Edn, cDn = bands_with_layer(tau, -1, V)
    return Eup, Edn, cUp, cDn


def band_color(color_col):
    """Majority-vote colour for one band across all k (robust to the
    ambiguous colour flip exactly at an accidental degeneracy)."""
    black_votes = np.sum(color_col == 'black')
    return 'black' if black_votes >= len(color_col) / 2 else 'red'


def cross_arrow(ax_top, ax_bot, y_top, y_bot, label, color, x_pos=0.0, xm=0.94,
                 y_offset=0.0, ha='right', label_ax=None, label_y=None):
    """Double-headed arrow that visually crosses the broken axis, connecting
    a point in the conduction sub-axis to a point in the valence sub-axis.
    x_pos (data coords) offsets the arrow itself sideways -- pass y_top/
    y_bot already evaluated AT that same x_pos (not at k=0), otherwise the
    arrow tip won't actually touch the curve it's meant to connect to.
    xm (axes fraction) + ha position the label horizontally; label_ax picks
    which of ax_top/ax_bot the label is drawn on (default ax_top); label_y
    (data coords, in label_ax's own y-scale) sets its height directly --
    if omitted, defaults to y_top + y_offset on ax_top."""
    con = ConnectionPatch(xyA=(x_pos, y_top), coordsA=ax_top.transData,
                           xyB=(x_pos, y_bot), coordsB=ax_bot.transData,
                           arrowstyle='<->', color=color, lw=1.3,
                           mutation_scale=12)
    ax_top.figure.add_artist(con)
    if label_ax is None:
        label_ax = ax_top
    if label_y is None:
        label_y = y_top + y_offset
    label_ax.annotate(label, xy=(xm, label_y), xycoords=label_ax.get_yaxis_transform(),
                       xytext=(xm, label_y), textcoords=label_ax.get_yaxis_transform(),
                       color=color, fontsize=7.5, ha=ha, va='bottom', clip_on=False)


def draw_panel(fig, outer_gs, block_row, col, tau, V):
    """Draw one (valley, V) panel as a broken-axis pair: ax_top = conduction,
    ax_bot = valence, touching with zero gap (hspace=0) so their shared
    spine reads as a single bold divider line, matching the paper's Fig. 1."""
    inner = outer_gs[block_row, col].subgridspec(2, 1, height_ratios=[1, 1.3], hspace=0.0)
    ax_top = fig.add_subplot(inner[0])
    ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

    Eup, Edn, cUp, cDn = bands(tau, V)

    for b in range(4):
        colU = band_color(cUp[:, b])
        colD = band_color(cDn[:, b])
        if V != 0.0 and tau == -1 and b == 0:
            # K' valence, lowest band (mu=--) at V!=0: force fully black
            # (per explicit visual request), rather than the eigenvector-
            # heuristic colors (which flip red/black between spin channels
            # here and would otherwise mismatch solid vs dotted).
            colU = colD = 'black'
        if V != 0.0 and tau == -1 and b == 1:
            # K' valence, upper band (mu=-+) at V!=0: force fully red for
            # the same reason (heuristic gave black solid / red dotted --
            # a mismatch within the same mu-group).
            colU = colD = 'red'
        if V == 0.0 and tau == -1 and b == 0:
            # K' valence at V=0: heuristic swaps red/black relative to the
            # K panel (b=0 comes out red, b=1 black) -- force it to match
            # K's convention (b=0 = mu=-- = black, b=1 = mu=-+ = red).
            colU = colD = 'black'
        if V == 0.0 and tau == -1 and b == 1:
            colU = colD = 'red'
        if V == 0.0 and tau == -1 and b == 2:
            # K' conduction at V=0: heuristic also swaps red/black here
            # (b=2 comes out black, b=3 red) relative to the fixed "++"/"+-"
            # legend colours (black=outer/"++", red=inner/"+-") -- force it
            # to match: b=2 (inner, mu=+-) = red, b=3 (outer, mu=++) = black.
            colU = colD = 'red'
        if V == 0.0 and tau == -1 and b == 3:
            colU = colD = 'black'
        ax = ax_top if b >= 2 else ax_bot
        ax.plot(q, Eup[:, b], color=colU, lw=1.6)              # solid = spin up
        # At V=0 the two spins are numerically identical (exact degeneracy),
        # so overlaying a dotted spin-down line on top of the solid spin-up
        # line only adds a speckled rendering artefact -- skip it and let
        # the single solid curve represent both, matching the paper's plot
        # (the legend still lists both up/down entries).
        if V != 0.0:
            ax.plot(q, Edn[:, b], color=colD, lw=1.2, ls=':')   # dotted = spin down

    # y-limits based on the actual data, with a little margin (paper-like ranges)
    cond_lo, cond_hi = min(Eup[:, 2].min(), Edn[:, 2].min()), max(Eup[:, 3].max(), Edn[:, 3].max())
    val_lo, val_hi = min(Eup[:, 0].min(), Edn[:, 0].min()), max(Eup[:, 1].max(), Edn[:, 1].max())
    ax_top.set_ylim(cond_lo - 0.01, cond_hi + 0.015)
    ax_bot.set_ylim(val_lo - 0.02, val_hi + 0.02)

    ax_top.set_xlim(-0.1, 0.1)
    ax_bot.set_xlabel(r"$ka/\pi$")
    ax_top.set_ylabel("Energy (eV)")
    ax_bot.set_ylabel("Energy (eV)")

    # Single clean box: ax_top and ax_bot are drawn touching (zero gap) so
    # the bottom spine of ax_top and the top spine of ax_bot coincide into
    # one bold divider line. No axis-break tick marks are drawn -- just a
    # plain rectangular border on all 4 sides, as requested.
    ax_top.tick_params(which='both', direction='in', top=True, right=True,
                        bottom=True, labelbottom=False)
    ax_bot.tick_params(which='both', direction='in', top=True, right=True,
                        bottom=True)
    for spine in ax_top.spines.values():
        spine.set_linewidth(1.4)
    for spine in ax_bot.spines.values():
        spine.set_linewidth(1.4)

    if V == 0.0:
        valley_lbl = "K" if tau == 1 else "K'"
        ax_top.set_title(fr"${valley_lbl}$ valley")

    # exact paper legend labels: conduction = mu(++,+-), valence = mu(-+,--)
    tsym = "+" if tau == 1 else "-"
    lbl = lambda s, mu: fr"$E^{{{s},{tsym}}}_{{{mu}}}$"
    up_a, dn_a = r"\uparrow", r"\downarrow"

    def legend_entry(ax, xc, y, ls, swatch_color, text, va='top', y_is_data=False):
        """A little [line-sample]+[black text] legend entry, swatch centred
        just left of xc, text starting right after -- text is always BLACK,
        only the sample line is colour-coded to the curve it represents.
        va='top' (default) anchors the TOP of the text/swatch at y, so
        y=0.995 sits right against the panel's top border with ~0 gap.
        y_is_data=True switches y to an actual data value (e.g. -0.85 on
        the energy axis) while xc stays axes-fraction (0=left..1=right),
        using the same mixed transform trick as the cross-panel labels."""
        transform = ax.get_yaxis_transform() if y_is_data else ax.transAxes
        ax.plot([xc - 0.05, xc - 0.01], [y, y], transform=transform,
                 color=swatch_color, linestyle=ls, lw=1.5, clip_on=False)
        ax.text(xc, y, text, transform=transform, color='black',
                fontsize=8, va=va, ha='left')

    # conduction: black-curve legend CENTERED over ka/pi=0.000 (the vertical
    # centre line where the red gap-arrow sits), right under the top border.
    # red-curve legend sits in the bottom-left corner, where the curve is
    # nowhere near (the curve only dips down close to the panel bottom
    # right at k=0, in the centre -- the corners stay empty at all k).
    legend_entry(ax_top, 0.525, 0.90, '-', 'black', lbl(up_a, "++"))
    legend_entry(ax_top, 0.525, 0.81, ':', 'black', lbl(dn_a, "++"))
    legend_entry(ax_top, 0.07, 0.24, '-', 'red', lbl(up_a, "+-"))
    legend_entry(ax_top, 0.07, 0.15, ':', 'red', lbl(dn_a, "+-"))
    # valence: red-curve legend top-left, black-curve legend top-right.
    # b=1 / mu=-+ is red in every panel; only the K' solid-entry height
    # differs (-0.75 vs -0.85) per explicit visual request.
    if V != 0.0 and tau == -1:
        legend_entry(ax_bot, 0.16, -0.73, '-', 'red', lbl(up_a, "-+"), y_is_data=True)
        legend_entry(ax_bot, 0.16, -0.88, ':', 'red', lbl(dn_a, "-+"), y_is_data=True)
    elif V == 0.0 and tau == 1:
        legend_entry(ax_bot, 0.16, -0.73, '-', 'red', lbl(up_a, "-+"), y_is_data=True)
        legend_entry(ax_bot, 0.16, -0.76, ':', 'red', lbl(dn_a, "-+"), y_is_data=True)
    elif V == 0.0 and tau == -1:
        legend_entry(ax_bot, 0.16, -0.75, '-', 'red', lbl(up_a, "-+"), y_is_data=True)
        legend_entry(ax_bot, 0.16, -0.78, ':', 'red', lbl(dn_a, "-+"), y_is_data=True)
    else:
        legend_entry(ax_bot, 0.16, -0.85, '-', 'red', lbl(up_a, "-+"), y_is_data=True)
        legend_entry(ax_bot, 0.16, -0.88, ':', 'red', lbl(dn_a, "-+"), y_is_data=True)
    legend_entry(ax_bot, 0.86, 0.96, '-', 'black', lbl(up_a, "--"))
    legend_entry(ax_bot, 0.86, 0.87, ':', 'black', lbl(dn_a, "--"))

    root = np.sqrt(lam ** 2 + gamma ** 2)

    if V == 0.0:
        # The paper only annotates these gap/split values on the K panel
        # (left) -- since K and K' are numerically identical at V=0, the
        # K' (right) panel just shows the bare curves, no red/blue labels.
        if tau == 1:
            # --- red cross-panel arrow: conduction bottom <-> valence top ---
            cross_arrow(ax_top, ax_bot, Eup[i0, 2], Eup[i0, 1],
                        r"$2\Delta-\sqrt{\lambda^2+\gamma^2}$", 'red')
            # --- blue arrow inside valence sub-axis: the 2*sqrt(l^2+g^2) split ---
            ax_bot.annotate('', xy=(0, Eup[i0, 1]), xytext=(0, Eup[i0, 0]),
                             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.3))
            ax_bot.text(0.012, (Eup[i0, 0] + Eup[i0, 1]) / 2,
                        r"$2\sqrt{\lambda^2+\gamma^2}$", color='blue', fontsize=9, va='center')
        ax_bot.text(0.0, val_lo - 0.012, r"$V=0~\mathrm{meV}$", fontsize=9, ha='center')
    else:
        Omega_up = +lam * V / root
        Omega_dn = -lam * V / root

        if tau == 1:
            # blue "2V" split, inside the conduction sub-axis
            ax_top.annotate('', xy=(0, Eup[i0, 3]), xytext=(0, Eup[i0, 2]),
                             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.3))
            ax_top.text(0.012, (Eup[i0, 2] + Eup[i0, 3]) / 2, r"$2V$",
                        color='blue', fontsize=9, va='center')

            # red / magenta cross-panel gap arrows (spin up / spin down channel).
            # Offset sideways so they don't overlap -- magenta on the LEFT
            # (x=-0.02), red on the RIGHT (x=+0.02). Each arrow must use the
            # curve's actual value AT that same x, not at k=0 (i0), otherwise
            # its tip doesn't actually touch the (dotted/solid) curve it's
            # supposed to connect to and a visible gap appears.
            ix_r = np.argmin(np.abs(q - 0.02))     # red     -> x = +0.02
            ix_m = np.argmin(np.abs(q - (-0.02)))  # magenta -> x = -0.02
            cross_arrow(ax_top, ax_bot, Eup[ix_r, 2], Eup[ix_r, 1],
                        r"$2\Delta-V-\sqrt{\lambda^2+\gamma^2}-\Omega^{\uparrow}$", 'red',
                        x_pos=0.02, xm=0.98, y_offset=0.004)
            cross_arrow(ax_top, ax_bot, Edn[ix_m, 2], Edn[ix_m, 1],
                        r"$2\Delta-V-\sqrt{\lambda^2+\gamma^2}+\Omega^{\downarrow}$", 'magenta',
                        x_pos=-0.02, xm=0.06, ha='left',
                        label_ax=ax_bot, label_y=-0.73)

            # blue fraction: spin splitting of the valence-top branch = 2V*lam/root
            ax_bot.annotate('', xy=(0, Eup[i0, 1]), xytext=(0, Edn[i0, 1]),
                             arrowprops=dict(arrowstyle='<->', color='blue', lw=1.1))
            ax_bot.text(0.0, -0.85,
                        r"$\dfrac{2V\lambda}{\sqrt{\lambda^2+\gamma^2}}$",
                        color='blue', fontsize=6.5, va='center', ha='center')
        ax_bot.text(0.0, val_lo - 0.012, r"$V=15~\mathrm{meV}$", fontsize=9, ha='center')


# ----------------- Compose the figure: 3 outer rows (V=0 block, spacer, V=15 block) x 2 cols -----------------
# Each (valley, V) block is kept a clear RECTANGLE, not a square: the
# conduction sub-panel (smaller height_ratio=1) sits on top, the taller
# valence sub-panel (height_ratio=1.3) sits below.
#
# TO CHANGE THE WIDTH (only), edit FIG_WIDTH below -- bigger number = wider
# panels, SAME height. This is the one knob to turn; nothing else needs to
# change and it will not disturb the touching-divider / legend layout.
FIG_WIDTH = 15.6    # inches; try 7.5, 8.5, etc. to make panels wider
FIG_HEIGHT = 15.0  # inches; controls height only

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = fig.add_gridspec(3, 2, height_ratios=[2.3, 0.15, 2.3], hspace=0.06, wspace=0.45)

print("Fig 1: starting computation...")
print("  panel: K valley, V=0 meV...")
draw_panel(fig, gs, 0, 0, tau=+1, V=0.0)     # K,  V=0
print("  panel: K' valley, V=0 meV...")
draw_panel(fig, gs, 0, 1, tau=-1, V=0.0)     # K', V=0
print("  panel: K valley, V=15 meV...")
draw_panel(fig, gs, 2, 0, tau=+1, V=0.015)   # K,  V=15meV
print("  panel: K' valley, V=15 meV...")
draw_panel(fig, gs, 2, 1, tau=-1, V=0.015)   # K', V=15meV
print("  panels done, finalizing figure...")

fig.suptitle(
    r"Band structure of bilayer MoS$_2$ for $\lambda=0.074$ eV and $\gamma=0.047$ eV.  "
    r"Upper: $V=0$; Lower: $V=15$ meV.  $\Omega^s = s\lambda V/\sqrt{\lambda^2+\gamma^2}$.",
    fontsize=10
)

fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.06)

# Save BEFORE show(): once plt.show() returns (window closed), the figure is
# gone, so a savefig() placed after it would write a blank PNG.
plt.savefig("bilayer_MoS2_fig1_matched.png", dpi=300, bbox_inches="tight")
print("Saved: bilayer_MoS2_fig1_matched.png")

plt.show()
