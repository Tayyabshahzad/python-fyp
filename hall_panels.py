"""
hall_panels.py
==============

Drawing code shared by Figures 10 and 11 (Hall conductivity vs B).

The physics lives in hall_common.py; this file only lays the panels out.
Both figures use the same arrangement, measured off the published pages:

    left  panel : B 6.5..13 , sigma 60..120 , y ticks every 10 ; x 7..13
    right panel : B 13..40  , sigma 20..60  , y ticks every 10 ;
                  x 15..40 step 5
    left  inset : B 7.5..9.5 , two curves  (black Mz=Mv=0, red both != 0)
    right inset : B 20..27   , THREE curves separating the Zeeman terms -
                  blue Mz=0 Mv!=0, red Mz!=0 Mv=0, black Mz=0 Mv=0.
                  The main panels never show that split, which is why the
                  level builder takes Mz and Mv as independent switches.

Only the inset y-ranges differ between the two figures, so they are passed
in rather than hard-coded here.
"""

import numpy as np
import matplotlib.pyplot as plt

import hall_common as hc
import paper_style as ps

ps.apply()


def _panel(ax, V, B_values, ylim, yticks, xticks, inset_rect, spec, v_label):
    ax.plot(B_values, hc.curve(B_values, V, False, False),
            color="black", lw=1.0)
    ax.plot(B_values, hc.curve(B_values, V, True, True),
            color="red", lw=1.0)

    ax.set_xlim(B_values[0], B_values[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(r"B (T)", fontsize=15, labelpad=2)
    ax.set_ylabel(r"$\sigma_{xy}$ ($e^2$/h)", fontsize=15)
    ps.frame(ax, labelsize=13)

    ax.text(0.055, 0.335, v_label, transform=ax.transAxes, fontsize=13)
    ps.legend_entry(ax, 0.055, 0.235, "-", "black",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ = 0", fontsize=13,
                    sample=0.100, gap=0.030)
    ps.legend_entry(ax, 0.055, 0.130, "-", "red",
                    r"$\mathrm{M_z}$ , $\mathrm{M_v}$ $\neq$ 0", fontsize=13,
                    sample=0.100, gap=0.030)

    iax = ax.inset_axes(inset_rect)
    Bi = np.linspace(spec["x"][0], spec["x"][1], 500)
    for mz, mv, colour in spec["curves"]:
        iax.plot(Bi, hc.curve(Bi, V, mz, mv), color=colour, lw=0.9)
    iax.set_xlim(*spec["x"])
    iax.set_ylim(*spec["y"])
    iax.set_xticks(spec["xticks"])
    iax.set_yticks(spec["yticks"])
    iax.set_xlabel(r"B (T)", fontsize=8, labelpad=1)
    iax.set_ylabel(r"$\sigma_{xy}(e^2$/h)", fontsize=8, labelpad=1)
    ps.frame(iax, labelsize=7.5, minor=False)
    iax.text(0.03, 0.07, v_label, transform=iax.transAxes, fontsize=7.5)
    for i, (lbl, colour) in enumerate(spec["labels"]):
        ps.legend_entry(iax, 0.44, 0.93 - i * 0.115, "-", colour, lbl,
                        fontsize=6.5, sample=0.11, gap=0.035)


def draw(V, filename, right_inset_y, right_inset_yticks):
    """Build one whole Hall figure - both panels and both insets."""
    v_label = f"V = {V*1000:.0f} meV"
    print(f"  building {filename}  ({v_label})")

    fig = plt.figure(figsize=(11.6, 4.3))
    gs = fig.add_gridspec(1, 2, wspace=0.30,
                          left=0.085, right=0.99, top=0.98, bottom=0.155)

    print("    left panel  (B 6.5 .. 13 T)")
    _panel(fig.add_subplot(gs[0, 0]), V, np.linspace(6.5, 13.0, 900),
           (60, 120), [60, 70, 80, 90, 100, 110, 120],
           [7, 8, 9, 10, 11, 12, 13], [0.42, 0.50, 0.55, 0.47],
           {"x": (7.5, 9.5), "y": (84, 106),
            "xticks": [7.5, 8.0, 8.5, 9.0, 9.5],
            "yticks": [85, 90, 95, 100, 105],
            "curves": [(False, False, "black"), (True, True, "red")],
            "labels": [(r"$\mathrm{M_z}$ , $\mathrm{M_v}$ = 0", "black"),
                       (r"$\mathrm{M_z}$ , $\mathrm{M_v}$ $\neq$ 0", "red")]},
           v_label)

    print("    right panel (B 13 .. 40 T)")
    _panel(fig.add_subplot(gs[0, 1]), V, np.linspace(13.0, 40.0, 900),
           (20, 60), [20, 30, 40, 50, 60],
           [15, 20, 25, 30, 35, 40], [0.34, 0.50, 0.63, 0.47],
           {"x": (20, 27), "y": right_inset_y,
            "xticks": [20, 21, 22, 23, 24, 25, 26, 27],
            "yticks": right_inset_yticks,
            "curves": [(False, True, "blue"), (True, False, "red"),
                       (False, False, "black")],
            "labels": [(r"$\mathrm{M_Z}$ = 0, $\mathrm{M_V}$ $\neq$ 0", "blue"),
                       (r"$\mathrm{M_Z}$ $\neq$ 0, $\mathrm{M_V}$ = 0", "red"),
                       (r"$\mathrm{M_Z}$ = 0, $\mathrm{M_V}$ = 0", "black")]},
           v_label)

    ps.save(fig, filename)
    return fig
