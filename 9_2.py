# =============================================================================
# FIGURE 9, PANEL 2 of 4  -  upper right: V = 0 meV,  B = 13 .. 40 T
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
    # "python 9_2.py fast" renders a quick draft instead of the
    # full-resolution panel - useful while adjusting the physics.
    DRAFT = 150 if "fast" in sys.argv else None

    f9.draw(
        V=0.0,
        B_values=np.linspace(13.0, 40.0, 500),
        ylim=(0.0, 2.42),
        yticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        xticks=[15, 20, 25, 30, 35, 40],
        v_pos="mid",
        points=DRAFT,
        filename="bilayer_MoS2_fig9_2.png",
    )
