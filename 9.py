# =============================================================================
# FIGURE 9  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 9 caption, verbatim:
#   "Dimensionless density of states (DOS) with D_c = g_{s/v}/D_0
#    Gamma sqrt(2 pi) vs B for a LL width Gamma = 0.1 sqrt(B) meV. The
#    upper panels are for V = 0 meV and the lower ones for V = 15 meV."
#
# This file assembles the published 2x2 figure.  The physics, the
# equations used, and the full record of what has and has not worked all
# live in fig9_common.py - so this file and the four single-panel files
# 9_1.py .. 9_4.py always agree.  Run those instead when iterating on one
# panel; "python 9_1.py fast" gives a draft in about a minute.
#
# STATUS: this is the one figure of the fourteen that does not reproduce.
# The baseline is flat and the dominant packet near 5 T is right, but
# below about 4 T the curves swing more than the published ones do.  See
# fig9_common.py for the five hypotheses tested and ruled out.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

import fig9_common as f9
import paper_style as ps

PANELS = [
    # (V, B grid, ylim, yticks, xticks, V-label position)
    (0.000, np.linspace(1.2, 13.0, 800), (0.0, 2.00),
     [0.0, 0.5, 1.0, 1.5, 2.0], [2, 4, 6, 8, 10, 12], "top"),
    (0.000, np.linspace(13.0, 40.0, 500), (0.0, 2.42),
     [0.0, 0.5, 1.0, 1.5, 2.0], [15, 20, 25, 30, 35, 40], "mid"),
    (0.015, np.linspace(1.8, 13.0, 800), (0.0, 2.00),
     [0.0, 0.5, 1.0, 1.5, 2.0], [2, 4, 6, 8, 10, 12], "top"),
    (0.015, np.linspace(13.0, 40.0, 500), (0.0, 2.95),
     [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], [15, 20, 25, 30, 35, 40], "mid"),
]

if __name__ == "__main__":
    import sys
    draft = 150 if "fast" in sys.argv else None
    print("Fig. 9 - DOS at E_F, all four panels")

    fig = plt.figure(figsize=(10.4, 6.9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                          left=0.080, right=0.985, top=0.985, bottom=0.095)

    for i, (V, B, ylim, yticks, xticks, v_pos) in enumerate(PANELS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        f9.draw(V, B, ylim, yticks, xticks, v_pos, ax=ax, points=draft)

    ps.save(fig, "bilayer_MoS2_fig9.png")
    plt.show()
