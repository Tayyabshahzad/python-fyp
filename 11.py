# =============================================================================
# FIGURE 11  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 11 caption, verbatim:
#   "Hall conductivity as a function of the magnetic field for T = 1 K
#    and V = 15 meV. The two panels differ only in the range of B
#    (x axis). For further clarity, the range 7.5 T-9.5 T is shown in
#    the inset to the left panel and the range 20 T-27 T in that to
#    the right one."
#
# Figures 10 and 11 are the same calculation and the same layout, differing
# only in the electric field energy and two inset limits, so the physics
# lives in hall_common.py and the layout in hall_panels.py.  Both files
# name every equation they use.  This file holds only what is specific to
# Figure 11.
# =============================================================================
import matplotlib.pyplot as plt

import hall_panels

if __name__ == "__main__":
    print("Fig. 11 - Hall conductivity, Eq. (22) in the T -> 0 limit")
    hall_panels.draw(
        V=0.015,
        filename="bilayer_MoS2_fig11.png",
        right_inset_y=(30, 40),
        right_inset_yticks=[30, 32, 34, 36, 38, 40],
    )
    plt.show()
