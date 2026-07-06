# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 03:29:07 2026

@author: 31690
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Initial parameter values; F will be scanned later
F = 0.13
kappa = 0.6
gainloss = 0.16

# Dimensionless parameters in the WSL formula
deltaf = 2 * kappa / F
gf = gainloss / F
n = deltaf
m = gf

# Coefficient matrix in the generating-function equation, u(t)/F
def A(t, m, n):
    return np.array([[1j * m, n * np.cos(t)],
                     [n * np.cos(t), -1j * m]], dtype=complex)

# Propagator equation: dY/dt = -i A(t) Y
def dYdt(t, Y, m, n):
    Y_matrix = Y.reshape(2, 2)
    A_t = A(t, m, n)
    dY_matrix = -1j * A_t.dot(Y_matrix)
    return dY_matrix.flatten()

#%% Scan F and compute U11(pi)-U22(pi)

kappa = 0.6
gainloss = 0.16

# x = F/(4*kappa), so reciprocal scan variable is 4*kappa/F
# Use fewer points than before to reduce lag
reciprocalofF_values = np.linspace(1, 41, 2001) / (4 * kappa)
farray = 1 / reciprocalofF_values

U11_U22_diff = np.zeros(len(farray), dtype=complex)

# Slightly fewer integration points to speed up
t_values = np.linspace(0, np.pi, 80)

for i, Ff in enumerate(farray):
    deltaf = 2 * kappa / Ff
    gf = gainloss / Ff
    n = deltaf
    m = gf

    Y0 = np.array([1, 0, 0, 1], dtype=complex)

    solution = integrate.solve_ivp(
        dYdt,
        (t_values[0], t_values[-1]),
        Y0,
        args=(m, n),
        t_eval=t_values,
        rtol=1e-7,
        atol=1e-9
    )

    Y_t_values = solution.y.T.reshape(-1, 2, 2)
    U11_U22_diff[i] = Y_t_values[-1, 0, 0] - Y_t_values[-1, 1, 1]

#%% Build the local WSL spectrum from U11(pi)-U22(pi)

q = np.arcsin(U11_U22_diff / (2j)) / np.pi

# Two adjacent reduced branches
Eresult = np.zeros((2, len(U11_U22_diff)), dtype=complex)
n_minus = 0
n_plus = 1

Eresult[0, :] = -q + 2 * n_minus
Eresult[1, :] =  q + (2 * n_plus - 1)

#%% Prepare plotting arrays

# x axis: F/(4*kappa)
xvals = farray / (4 * kappa)

# Sort by x for smooth plotting
sort_idx = np.argsort(xvals)
xplot = xvals[sort_idx]

# Im(E)/kappa, since Eresult is E/F
yplot_0 = (Eresult[0, :].imag * farray / kappa)[sort_idx]
yplot_1 = (Eresult[1, :].imag * farray / kappa)[sort_idx]

# Target cases
case_x = np.array([0.0538, 0.0545, 0.0533, 0.0550])
case_labels = ["1", "2", "3", "4"]

# Find nearest sampled points
case_idx = [np.argmin(np.abs(xplot - xx)) for xx in case_x]
case_x_near = xplot[case_idx]
case_y0 = yplot_0[case_idx]
case_y1 = yplot_1[case_idx]

#%% Plot Fig. S3(a): full spectrum + zoomed panel with four marked cases

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), dpi=150)
ax0, ax1 = axes

# ---------------- Left panel: full spectrum ----------------
ax0.plot(xplot, yplot_0, color="#1f4ea8", lw=1.4)
ax0.plot(xplot, yplot_1, color="#1f4ea8", lw=1.4)

ax0.set_xlabel(r"$F/4\kappa$", fontsize=16, labelpad=2)
ax0.set_ylabel(r"$Im(E)/\kappa$", fontsize=16, labelpad=3)

ax0.set_xlim(0.04, 0.12)
ax0.set_ylim(-0.10, 0.10)

ax0.set_xticks([0.04, 0.06, 0.08, 0.10, 0.12])
ax0.set_yticks([-0.10, -0.05, 0, 0.05, 0.10])
ax0.set_yticklabels([r"$-0.1$", r"$-0.05$", r"$0$", r"$0.05$", r"$0.1$"])
ax0.tick_params(axis='both', labelsize=12.5, top=True, right=True, length=2.8)

# Red zoom rectangle
x_rect0 = 0.0518
x_rect1 = 0.0556
y_rect0 = -0.052
y_rect1 = 0.052
rect = Rectangle(
    (x_rect0, y_rect0),
    x_rect1 - x_rect0,
    y_rect1 - y_rect0,
    fill=False,
    ec="red",
    lw=1.2
)
ax0.add_patch(rect)

# Red arrow
ax0.annotate(
    "",
    xy=((x_rect0 + x_rect1) / 2, y_rect1),
    xytext=(0.0557, 0.073),
    arrowprops=dict(arrowstyle="->", color="red", lw=1.0)
)

for spine in ax0.spines.values():
    spine.set_linewidth(0.8)

# ---------------- Right panel: zoomed local spectrum ----------------
ax1.plot(xplot, yplot_0, color="#1f4ea8", lw=1.4)
ax1.plot(xplot, yplot_1, color="#1f4ea8", lw=1.4)

# Mark the 4 selected cases
for i, (xx, y0, y1, lab) in enumerate(zip(case_x_near, case_y0, case_y1, case_labels)):
    # red points
    ax1.scatter([xx, xx], [y0, y1], s=20, color="red", zorder=5)
    # dashed vertical connector
    ax1.plot([xx, xx], [y0, y1], color="red", lw=0.9, ls=(0, (3, 3)), alpha=0.75)

# Number labels placed manually to mimic the figure
# case 1 = 0.0538 (upper-left)
# case 2 = 0.0545 (upper-right)
# case 3 = 0.0533 (lower-left)
# case 4 = 0.0550 (lower-right)

label_pos = {
    "1": (case_x_near[0], max(case_y0[0], case_y1[0]) + 0.010),
    "2": (case_x_near[1], max(case_y0[1], case_y1[1]) + 0.010),
    "3": (case_x_near[2], min(case_y0[2], case_y1[2]) - 0.010),
    "4": (case_x_near[3], min(case_y0[3], case_y1[3]) - 0.010),
}

for lab in case_labels:
    xx, yy = label_pos[lab]
    ax1.text(
        xx, yy, lab,
        ha="center", va="center",
        fontsize=10.5, color="#1f8a5b",
        bbox=dict(boxstyle="circle,pad=0.14", fc="#e8c77a", ec="none", alpha=0.95)
    )

ax1.set_xlabel(r"$F/4\kappa$", fontsize=16, labelpad=2)
ax1.set_xlim(0.052, 0.056)
ax1.set_ylim(-0.05, 0.05)

ax1.set_xticks([0.052, 0.053, 0.054, 0.055, 0.056])
ax1.set_yticks([-0.05, 0, 0.05])
ax1.set_yticklabels([r"$-0.05$", r"$0$", r"$0.05$"])
ax1.tick_params(axis='both', labelsize=12.5, top=True, right=True, length=2.8)

for spine in ax1.spines.values():
    spine.set_linewidth(0.8)

# Panel label
fig.text(0.012, 0.95, "(a)", fontsize=16, fontweight="bold")

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.92, wspace=0.22)
plt.show()