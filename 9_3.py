# =============================================================================
# FIGURE 9, PANEL 3 of 4  -  lower left:  V = 15 meV, B = 1.8 .. 13 T
#
# of Zubair, Tahir, Vasilopoulos & Sabeeh, Phys. Rev. B 96, 045405 (2017).
#
# Figure 9's four panels are matched against the published figure one at a
# time, so each lives in its own file and renders on its own - a quarter of
# the work of the full figure.  All the physics, the equations used and the
# normalisation constants are in fig9_common.py; only this panel's axis
# limits are here.  Run this file to produce just this panel.
# =============================================================================
import sys

import numpy as np

import fig9_common as f9

if __name__ == "__main__":
    # "python 9_3.py fast" renders a quick draft instead of the
    # full-resolution panel - useful while adjusting the physics.
    DRAFT = 150 if "fast" in sys.argv else None

    f9.draw(
        V=0.015,
        B_values=np.linspace(1.8, 13.0, 800),
        ylim=(0.0, 2.00),
        yticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        xticks=[2, 4, 6, 8, 10, 12],
        v_pos="top",
        points=DRAFT,
        filename="bilayer_MoS2_fig9_3.png",
    )
