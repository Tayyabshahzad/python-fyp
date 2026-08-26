# =============================================================================
# FIGURE 10  of  Zubair, Tahir, Vasilopoulos & Sabeeh,
# Phys. Rev. B 96, 045405 (2017).
#
# Paper's Fig. 10 caption, verbatim:
#   "Hall conductivity as a function of the magnetic field B for T = 1 K
#    and V = 0 meV. The two panels differ only in the range of B."
#
# Figures 10 and 11 are the same calculation and the same layout, differing
# only in the electric field energy and two inset limits, so the physics
# lives in hall_common.py and the layout in hall_panels.py.  Both files
# name every equation they use.  This file holds only what is specific to
# Figure 10.
# =============================================================================
import matplotlib.pyplot as plt

import hall_panels

if __name__ == "__main__":
    print("Fig. 10 - Hall conductivity, Eq. (22) in the T -> 0 limit")
    hall_panels.draw(
        V=0.0,
        filename="bilayer_MoS2_fig10.png",
        right_inset_y=(28, 40),
        right_inset_yticks=[28, 30, 32, 34, 36, 38, 40],
    )
    plt.show()
