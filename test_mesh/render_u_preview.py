#!/usr/bin/env python3
"""Render a 2D schematic of the inverted-U geometry with the new top port.

Outputs:
  test_mesh/u_preview.png  — annotated schematic (matches gen_u_2d.py)
"""
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Same parameters as gen_u_2d.py / gen_u_3d.py
W   = 0.025
Ws  = 0.010
Rc  = 0.060
L   = 0.150
Lb  = 0.030

hw = W / 2.0
Ri = Rc - hw
Ro = Rc + hw
xC, yC = Rc, L

theta_port_half  = math.asin((Ws / 2.0) / Ro)
port_theta_right = math.pi / 2.0 - theta_port_half
port_theta_left  = math.pi / 2.0 + theta_port_half

nr_right = (math.cos(port_theta_right), math.sin(port_theta_right))
nr_left  = (math.cos(port_theta_left),  math.sin(port_theta_left))

port_arch_right = (xC + Ro * nr_right[0], yC + Ro * nr_right[1])
port_arch_left  = (xC + Ro * nr_left[0],  yC + Ro * nr_left[1])
port_top_right  = (port_arch_right[0] + Lb * nr_right[0],
                   port_arch_right[1] + Lb * nr_right[1])
port_top_left   = (port_arch_left[0]  + Lb * nr_left[0],
                   port_arch_left[1]  + Lb * nr_left[1])


def arc_xy(theta_start, theta_end, R, n=80):
    th = np.linspace(theta_start, theta_end, n)
    return xC + R * np.cos(th), yC + R * np.sin(th)


fig, ax = plt.subplots(figsize=(8, 8))

# --- Inner arch (red) ---
xi, yi = arc_xy(0, math.pi, Ri)
ax.plot(xi, yi, color="#bbbbbb", lw=1.5)

# Inner straight walls (legs)
ax.plot([hw, hw], [0, L], color="#bbbbbb", lw=1.5)
ax.plot([2*Rc - hw, 2*Rc - hw], [0, L], color="#bbbbbb", lw=1.5)

# --- Outer arch — split by the port ---
xo_r, yo_r = arc_xy(0, port_theta_right, Ro)
xo_l, yo_l = arc_xy(port_theta_left, math.pi, Ro)
ax.plot(xo_r, yo_r, color="#444444", lw=2)
ax.plot(xo_l, yo_l, color="#444444", lw=2)

# Outer straight walls (legs)
ax.plot([-hw, -hw], [0, L], color="#444444", lw=2)
ax.plot([2*Rc + hw, 2*Rc + hw], [0, L], color="#444444", lw=2)

# --- Port walls (vertical, slightly fanned outward) ---
ax.plot([port_arch_right[0], port_top_right[0]],
        [port_arch_right[1], port_top_right[1]], color="#444444", lw=2)
ax.plot([port_arch_left[0], port_top_left[0]],
        [port_arch_left[1], port_top_left[1]], color="#444444", lw=2)

# --- Inlet faces and outlet (highlighted) ---
# inlet_main (bottom of left leg)
ax.plot([-hw, hw], [0, 0], color="tab:blue", lw=4)
# outlet
ax.plot([2*Rc - hw, 2*Rc + hw], [0, 0], color="tab:red", lw=4)
# inlet_small (top of port)
ax.plot([port_top_left[0], port_top_right[0]],
        [port_top_left[1], port_top_right[1]], color="tab:green", lw=4)

# --- Arrows showing flow direction ---
ax.annotate("", xy=(0, 0.04), xytext=(0, 0.01),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.5))
ax.annotate("", xy=(2*Rc, 0.01), xytext=(2*Rc, 0.04),
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.5))
midx_port = 0.5 * (port_top_left[0] + port_top_right[0])
ax.annotate("", xy=(midx_port, yC + Ro + 0.005),
            xytext=(midx_port, yC + Ro + Lb - 0.005),
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=2.5))

# Flow streamline hint through the arch
arch_mid_x, arch_mid_y = arc_xy(math.pi - 0.4, 0.4, Rc, n=50)
ax.plot(arch_mid_x, arch_mid_y, ":", color="tab:blue", lw=1.2, alpha=0.7)
ax.annotate("", xy=(arch_mid_x[-1], arch_mid_y[-1]),
            xytext=(arch_mid_x[-3], arch_mid_y[-3]),
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5, alpha=0.7))

# --- Labels ---
ax.text(0, -0.012, "inlet_main\n(4 m/s, 500 K)", ha="center", va="top",
        color="tab:blue", fontsize=10, fontweight="bold")
ax.text(2*Rc, -0.012, "outlet\n(101325 Pa)", ha="center", va="top",
        color="tab:red", fontsize=10, fontweight="bold")
ax.text(midx_port, yC + Ro + Lb + 0.005,
        "inlet_small\n(1 m/s ↓, 280 K)",
        ha="center", va="bottom", color="tab:green", fontsize=10, fontweight="bold")

# walls label
ax.text(-hw - 0.012, L/2, "walls\n(600 K)", ha="right", va="center",
        color="#444444", fontsize=10, rotation=90)

# Annotate apex injection physics
ax.annotate(
    "main flow passes\nhorizontally under\nthe injection port",
    xy=(xC + 0.005, yC + Ri - 0.005),
    xytext=(xC + 0.07, yC + Ri - 0.04),
    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    fontsize=9, color="gray", ha="left",
)

# --- Title and layout ---
ax.set_title(
    "Inverted-U duct with top-port secondary inlet\n"
    f"(W={W*1000:.0f} mm, Ws={Ws*1000:.0f} mm, "
    f"Rc={Rc*1000:.0f} mm, L={L*1000:.0f} mm, Lb={Lb*1000:.0f} mm)",
    fontsize=12,
)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Limits with a bit of padding
ax.set_xlim(-hw - 0.04, 2*Rc + hw + 0.04)
ax.set_ylim(-0.025, yC + Ro + Lb + 0.025)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "u_preview.png")
plt.tight_layout()
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"Wrote {out_path}")
