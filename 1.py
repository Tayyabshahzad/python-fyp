# =============================================================================
# FIGURE 1  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# "Quantum magnetotransport in bilayer MoS2: Influence of perpendicular
#  electric field", Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 1 caption, verbatim:
#   "Band structure of bilayer MoS2 for lambda = 0.074 eV and gamma = 0.047
#    eV. The upper panels are for zero electric field energy (V = 0) and the
#    lower ones for V = 15 meV. The left (right) panels are for the K (K')
#    valley and Omega^s = s*lambda*V/[lambda^2 + gamma^2]^{1/2}."
#
# EQUATIONS USED  (all implemented in paper_equations.py, none invented here):
#   Eq. (3)  p.2 - the fourth-degree equation whose roots are epsilon
#   Eq. (2)  p.2 - E = hbar*v_F*epsilon, converts those roots into eV
#   Eq. (1)  p.1 - the Hamiltonian Eq. (3) is derived from (used as a check)
#
# HOW THE FIGURE IS BUILT
#   for each of 401 points k on the horizontal axis:
#       solve Eq. (3)          -> 4 roots epsilon
#       apply Eq. (2)          -> 4 energies in eV
#   repeat for s = +1 and s = -1 (spin up / down)   -> 8 curves per panel
#   repeat for (tau, V) = (+1,0), (-1,0), (+1,15meV), (-1,15meV) -> 4 panels
#
# LAYOUT
#   Axis limits, tick values and panel proportions below were measured
#   directly off the published figure (300 dpi render of PDF page 2):
#     frame 581 x 722 px, split into two EQUAL 361 px sub-panels
#     V=0     : conduction 0.7997..0.9080 , valence -1.0219..-0.5947
#     V=15meV : conduction 0.7877..0.9080 , valence -1.0005..-0.6483
#     x-axis  : -0.1 .. 0.1 in both
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import paper_equations as pe

P = pe.P

# --- typography: the paper is set in Times; STIX is its metric-compatible
# --- open counterpart, and matches for both text and mathematics.
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.2

# Dimensionless abscissa of the paper, q = ka/pi, and the wave vector it
# corresponds to.  Eq. (3) is solved at each of these k.
Q = np.linspace(-0.1, 0.1, 401)
K = (np.pi / P.A_LATTICE) * Q
I0 = np.argmin(np.abs(Q))

# Sign convention for Eq. (3).  "eq1" reproduces the published Fig. 1;
# "eq3" is Eq. (3) exactly as printed.  See discrepancy (D1) in
# paper_equations.py - the two differ only by exchanging the spin labels.
SIGN = "eq1"

# Band index -> colour, following the paper's mu labelling.
#   mu = (--) lowest valence  = black      mu = (-+) upper valence = red
#   mu = (+-) lower conduction = red       mu = (++) upper conduction = black
# Both spins of a given band share a colour; spin is shown by line style.
BAND_COLOR = {0: "black", 1: "red", 2: "red", 3: "black"}
BAND_MU = {0: "--", 1: "-+", 2: "+-", 3: "++"}

# Measured axis limits, see header.
YLIM = {
    0.0:   {"cond": (0.7997, 0.9080), "val": (-1.0219, -0.5947)},
    0.015: {"cond": (0.7877, 0.9080), "val": (-1.0005, -0.6483)},
}


def bands(tau, V):
    """Energies of all four bands over the whole k range, both spins.

    Eq. (3) -> Eq. (2), evaluated at every k.  Returns two arrays of shape
    (len(K), 4), ascending in energy: columns 0,1 = valence, 2,3 = conduction.
    """
    up = np.empty((len(K), 4))
    dn = np.empty((len(K), 4))
    for i, k in enumerate(K):
        up[i] = pe.eq2_eq3_band_energies(k, +1, tau, V, sign_convention=SIGN)
        dn[i] = pe.eq2_eq3_band_energies(k, -1, tau, V, sign_convention=SIGN)
    return up, dn


def legend_entry(ax, x, y, style, color, text, fontsize=8.5):
    """One legend row: a short colour-coded line sample, then black text.

    x, y are axes fractions.  The paper draws its legends this way - the
    sample carries the colour and the line style, the label itself is
    always black.
    """
    kw = {"dashes": (1, 1.4)} if style == ":" else {}
    ax.plot([x, x + 0.105], [y, y], transform=ax.transAxes, color=color,
            linestyle=style, lw=1.3, clip_on=False, **kw)
    ax.text(x + 0.135, y, text, transform=ax.transAxes, color="black",
            fontsize=fontsize, va="center", ha="left")


def mu_label(mu, spin, tau):
    """The paper's curve labels, e.g. E^{up,+}_{++}."""
    arrow = r"\uparrow" if spin > 0 else r"\downarrow"
    valley = "+" if tau > 0 else "-"
    return fr"$\mathrm{{E}}^{{{arrow},{valley}}}_{{{mu}}}$"


def draw_panel(fig, gs_cell, tau, V):
    """One (valley, V) panel: conduction above, valence below, sharing a
    single bold divider (hspace = 0), exactly as in the published figure."""
    inner = gs_cell.subgridspec(2, 1, height_ratios=[1, 1], hspace=0.0)
    ax_c = fig.add_subplot(inner[0])
    ax_v = fig.add_subplot(inner[1], sharex=ax_c)

    up, dn = bands(tau, V)

    for b in range(4):
        ax = ax_c if b >= 2 else ax_v
        color = BAND_COLOR[b]
        ax.plot(Q, up[:, b], color=color, lw=1.25)
        # At V = 0 the two spins are exactly degenerate, so the dotted
        # spin-down curve would sit precisely on top of the solid spin-up
        # one and only produce a speckled artefact.  The paper shows a
        # single curve there too, while still listing both in the legend.
        if V != 0.0:
            ax.plot(Q, dn[:, b], color=color, lw=1.15, linestyle=":",
                    dashes=(1, 1.4))

    lim = YLIM[V]
    ax_c.set_ylim(*lim["cond"])
    ax_v.set_ylim(*lim["val"])
    ax_c.set_xlim(-0.1, 0.1)

    ax_c.set_yticks([0.83, 0.86, 0.89])
    ax_c.set_yticklabels(["0.83", "0.86", "0.89"])
    ax_v.set_yticks([-1.0, -0.9, -0.8])
    ax_v.set_yticklabels([r"$-1$", r"$-0.9$", r"$-0.8$"])
    ax_v.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax_v.set_xticklabels([r"$-0.1$", r"$-0.05$", "0", "0.05", "0.1"])

    for ax in (ax_c, ax_v):
        ax.tick_params(which="both", direction="in", top=True, right=True,
                       bottom=True, left=True, labelsize=10.5,
                       length=5, width=1.0)
        ax.tick_params(which="minor", length=2.6)
        ax.minorticks_on()
        ax.tick_params(which="minor", top=True, right=True, bottom=True,
                       left=True)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2)
    ax_c.tick_params(labelbottom=False)

    ax_v.set_xlabel(r"ka/$\pi$", fontsize=12.5, labelpad=2)
    ax_c.set_ylabel("Energy (eV)", fontsize=11.5)
    ax_v.set_ylabel("Energy (eV)", fontsize=11.5)

    # ---- legends, laid out as in the published panel --------------------
    legend_entry(ax_c, 0.37, 0.87, "-", BAND_COLOR[3], mu_label("++", +1, tau))
    legend_entry(ax_c, 0.37, 0.72, ":", BAND_COLOR[3], mu_label("++", -1, tau))
    legend_entry(ax_c, 0.02, 0.33, "-", BAND_COLOR[2], mu_label("+-", +1, tau))
    legend_entry(ax_c, 0.02, 0.18, ":", BAND_COLOR[2], mu_label("+-", -1, tau))
    # On the K panel at V != 0 the paper moves the red valence legend down,
    # to leave the top-left corner free for the magenta gap label.
    y_red = (0.30, 0.17) if (tau == 1 and V != 0.0) else (0.93, 0.78)
    legend_entry(ax_v, 0.02, y_red[0], "-", BAND_COLOR[1], mu_label("-+", +1, tau))
    legend_entry(ax_v, 0.02, y_red[1], ":", BAND_COLOR[1], mu_label("-+", -1, tau))
    legend_entry(ax_v, 0.60, 0.93, "-", BAND_COLOR[0], mu_label("--", +1, tau))
    legend_entry(ax_v, 0.60, 0.78, ":", BAND_COLOR[0], mu_label("--", -1, tau))

    # ---- the V label the paper prints inside every panel -----------------
    ax_v.text(0.5, 0.06, f"V = {V*1000:.0f} meV", transform=ax_v.transAxes,
              fontsize=11, ha="center")

    annotate(ax_c, ax_v, up, dn, tau, V)
    return ax_c, ax_v


def cross_arrow(ax_top, ax_bot, x, y_top, y_bot, color):
    """Double-headed arrow spanning the broken axis, from a point in the
    conduction sub-panel to a point in the valence sub-panel.  y_top and
    y_bot must both be evaluated AT x, or the tips will not touch the
    curves they are meant to connect."""
    con = ConnectionPatch(xyA=(x, y_top), coordsA=ax_top.transData,
                          xyB=(x, y_bot), coordsB=ax_bot.transData,
                          arrowstyle="<->", color=color, lw=1.1,
                          mutation_scale=9)
    ax_top.figure.add_artist(con)


def annotate(ax_c, ax_v, up, dn, tau, V):
    """The gap / splitting arrows.  The paper draws these on the K panels
    only; the K' panels show the bare curves."""
    if tau != 1:
        return
    lam, gam, D = P.LAMBDA, P.GAMMA, P.DELTA
    root = np.sqrt(lam ** 2 + gam ** 2)

    if V == 0.0:
        # gap:  2*Delta - sqrt(lambda^2 + gamma^2)     [p.2, item (iii)]
        cross_arrow(ax_c, ax_v, 0.0, up[I0, 2], up[I0, 1], "red")
        ax_c.text(0.545, 0.11, r"$2\Delta-\sqrt{\lambda^2+\gamma^2}$",
                  transform=ax_c.transAxes, color="red", fontsize=10,
                  va="center")
        # valence interlayer splitting: 2*sqrt(lambda^2 + gamma^2)  [p.2 (ii)]
        ax_v.annotate("", xy=(0, up[I0, 1]), xytext=(0, up[I0, 0]),
                      arrowprops=dict(arrowstyle="<->", color="blue", lw=1.1,
                                      mutation_scale=9))
        ax_v.text(0.53, 0.46, r"$2\sqrt{\lambda^2+\gamma^2}$",
                  transform=ax_v.transAxes, color="blue", fontsize=10.5,
                  va="center")
    else:
        # conduction layer splitting: 2V                          [p.2 (ii)]
        ax_c.annotate("", xy=(0, up[I0, 3]), xytext=(0, up[I0, 2]),
                      arrowprops=dict(arrowstyle="<->", color="blue", lw=1.1,
                                      mutation_scale=9))
        ax_c.text(0.545, 0.30, r"$2V$", transform=ax_c.transAxes,
                  color="blue", fontsize=11, va="center")

        # gap for each spin channel, offset sideways so the two arrows do
        # not overlap.  Each is evaluated at its own x.
        ir = np.argmin(np.abs(Q - 0.012))
        im = np.argmin(np.abs(Q + 0.012))
        cross_arrow(ax_c, ax_v, +0.012, up[ir, 2], up[ir, 1], "red")
        cross_arrow(ax_c, ax_v, -0.012, dn[im, 2], dn[im, 1], "magenta")
        ax_c.text(0.55, 0.075,
                  r"$2\Delta-V-\sqrt{\lambda^2+\gamma^2}-\Omega^{\uparrow}$",
                  transform=ax_c.transAxes, color="red", fontsize=8,
                  va="center")
        ax_v.text(0.015, 0.94,
                  r"$2\Delta-V-\sqrt{\lambda^2+\gamma^2}+\Omega^{\downarrow}$",
                  transform=ax_v.transAxes, color="magenta", fontsize=8,
                  va="center")

        # valence spin splitting: 2*V*lambda/sqrt(lambda^2+gamma^2) [p.2 (ii)]
        # The splitting is only ~25 meV, so the two arrowheads would collide
        # at the default size; mutation_scale is reduced to keep it legible.
        ax_v.annotate("", xy=(0, up[I0, 1]), xytext=(0, dn[I0, 1]),
                      arrowprops=dict(arrowstyle="<->", color="blue", lw=0.9,
                                      mutation_scale=5, shrinkA=0, shrinkB=0))
        ax_v.text(0.545, 0.70,
                  r"$\dfrac{2V\lambda}{\sqrt{\lambda^2+\gamma^2}}$",
                  transform=ax_v.transAxes, color="blue", fontsize=8.5,
                  va="top", ha="center")


# =============================================================================
if __name__ == "__main__":
    print("Fig. 1 - solving Eq. (3) and applying Eq. (2)...")

    fig = plt.figure(figsize=(7.3, 8.15))
    gs = fig.add_gridspec(2, 2, hspace=0.245, wspace=0.335,
                          left=0.105, right=0.985, top=0.995, bottom=0.062)

    for row, V in enumerate([0.0, 0.015]):
        for col, tau in enumerate([+1, -1]):
            print(f"  panel: {'K' if tau > 0 else chr(75)+chr(39)} valley, "
                  f"V = {V*1000:.0f} meV")
            draw_panel(fig, gs[row, col], tau, V)

    plt.savefig("bilayer_MoS2_fig1.png", dpi=300)
    print("Saved: bilayer_MoS2_fig1.png")
    plt.show()
