# -*- coding: utf-8 -*-
"""
Fig. S3(b): real-space dynamics for four F-detuned cases.

This script reproduces the four real-space dynamics panels in Fig. S3(b):
    case 1: F/(4*kappa) = 0.0538
    case 3: F/(4*kappa) = 0.0533
    case 2: F/(4*kappa) = 0.0545
    case 4: F/(4*kappa) = 0.0550

No disorder is included.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp


# ============================================================
# 1. Basic settings
# ============================================================
N = 202
CENTER_INDEX = 100

t_eval = np.linspace(0.0, 4.0, 4001)

# Initial Gaussian wave packet
k0 = 0.03
ksita = 0.0

psi0 = np.zeros(N, dtype=complex)
for i in range(N):
    psi0[i] = np.exp(-k0 * (CENTER_INDEX - i) ** 2) * np.exp(1.0j * i * ksita)


# ============================================================
# 2. Parameter convention
# ============================================================
# The Hamiltonian uses F_actual = pi, so that t is measured in units of
# the Bloch period. To realize a target q = F/(4*kappa), we set
#
#   kappa_actual = 0.6*pi/F_ratio
#   gamma_actual = 0.16*pi/F_ratio
#   F_actual     = pi
#
# Therefore:
#
#   q = F_actual/(4*kappa_actual) = F_ratio/2.4
#   F_ratio = 2.4*q
#
F_ACTUAL = np.pi
KAPPA_RATIO = 0.6
GAMMA_RATIO = 0.16

# Caption order:
# cases (1)-(4): q = F/(4*kappa) = 0.0538, 0.0545, 0.0533, 0.0550
case_info = {
    1: {"q": 0.0538},
    2: {"q": 0.0545},
    3: {"q": 0.0533},
    4: {"q": 0.0550},
}

# Panel layout:
# top-left: 1, top-right: 3, bottom-left: 2, bottom-right: 4
panel_order = [1, 3, 2, 4]


# ============================================================
# 3. Simulation function
# ============================================================
def run_detuned_case(q_F_over_4kappa, psi0, t_eval, max_step=0.05):
    """Run one clean F-detuned case without disorder."""
    F_ratio = 2.4 * q_F_over_4kappa

    kappa = KAPPA_RATIO * np.pi / F_ratio
    gamma = GAMMA_RATIO * np.pi / F_ratio

    psi0 = np.asarray(psi0, dtype=complex)
    n_sites = psi0.size

    # Build time-independent Hamiltonian
    H = np.zeros((n_sites, n_sites), dtype=complex)
    idx = np.arange(n_sites)

    # Linear potential and alternating gain/loss
    H[idx, idx] = (idx - CENTER_INDEX) * F_ACTUAL + 1j * ((-1) ** idx) * gamma

    # Nearest-neighbor hopping
    i = np.arange(n_sites - 1)
    H[i, i + 1] = kappa
    H[i + 1, i] = np.conjugate(kappa)

    def rhs(t, psi):
        return -1j * (H @ psi)

    sol = solve_ivp(
        rhs,
        [t_eval[0], t_eval[-1]],
        psi0,
        t_eval=t_eval,
        max_step=max_step,
    )

    ww = np.abs(sol.y)
    intensity = ww**2
    su = np.sum(intensity, axis=0)
    su_norm = su / su[0]

    return {
        "q": q_F_over_4kappa,
        "F_ratio": F_ratio,
        "kappa": kappa,
        "gamma": gamma,
        "sol": sol,
        "y": sol.y,
        "ww": ww,
        "su": su,
        "su_norm": su_norm,
    }


# ============================================================
# 4. Run four cases
# ============================================================
results = {}

for case_id, info in case_info.items():
    out = run_detuned_case(
        q_F_over_4kappa=info["q"],
        psi0=psi0,
        t_eval=t_eval,
        max_step=0.05,
    )
    results[case_id] = out

    print(
        f"case {case_id}: "
        f"F/(4*kappa)={out['q']:.4f}, "
        f"F_ratio={out['F_ratio']:.5f}, "
        f"kappa={out['kappa']:.5f}, "
        f"gamma={out['gamma']:.5f}, "
        f"population max={out['su_norm'].max():.3f}"
    )

#%%
# ============================================================
# 5. Plot Fig. S3(b)
# ============================================================



mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["mathtext.rm"] = "Times New Roman"
mpl.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axes = plt.subplots(
    2, 2,
    figsize=(8.2, 4.25),
    dpi=150,
    sharex=True,
    sharey=True,
)

axes_flat = axes.ravel()

# ==============================
# Real-space display window
# ==============================
site_min, site_max = 50, 150
site_idx = np.arange(site_min, site_max + 1)
X, Y = np.meshgrid(t_eval, site_idx)

# ==============================
# Common heatmap color scale
# ==============================
Z_all = [
    np.abs(results[case_id]["y"][site_min:site_max + 1, :])
    for case_id in panel_order
]

vmax = max(np.percentile(Z, 99.5) for Z in Z_all)

# ==============================
# Common population y-scale

pop_ymin = 0.0
pop_ymax = 8.0


for ax, case_id, Z in zip(axes_flat, panel_order, Z_all):
    out = results[case_id]
    q = out["q"]

    # ==============================
    # Heatmap
    # ==============================
    ax.pcolormesh(
        X,
        Y,
        Z,
        cmap="viridis",
        shading="gouraud",
        vmin=0.0,
        vmax=1.2 * vmax,
    )

    # ==============================
    # Yellow total-population curve
    # Use common right y-scale for all panels.
    # No clipping, no mapping to lattice axis.
    # ==============================
    ax2 = ax.twinx()

    ax2.plot(
        t_eval,
        out["su_norm"],
        color="yellow",
        lw=1.2,
        alpha=0.95,
        zorder=6,
    )

    ax2.set_ylim(pop_ymin, pop_ymax)

    # Hide right axis; keep only the curve
    ax2.set_yticks([])
    ax2.tick_params(axis="y", right=False, labelright=False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # ==============================
    # In-panel parameter label
    # ==============================
    ax.text(
        0.95,
        0.90,
        rf"$F/4\kappa={q:.4f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        color="white",
    )

    # ==============================
    # Numbered circular label
    # ==============================
    ax.text(
        0.045,
        0.86,
        f"{case_id}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        zorder=10,
        bbox=dict(
            boxstyle="circle,pad=0.22",
            facecolor="#8a5a48",
            edgecolor="none",
            alpha=0.9,
        ),
    )

    # ==============================
    # Axis settings
    # ==============================
    ax.set_xlim(0, 4)
    ax.set_ylim(site_min, site_max)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([50, 100, 150])
    ax.tick_params(
        axis="both",
        labelsize=13,
        top=False,
        right=False,
        length=3
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

# ==============================
# Axis labels
# ==============================
axes[0, 0].set_ylabel("Lattice number", fontsize=16, labelpad=4)
axes[1, 0].set_ylabel("Lattice number", fontsize=16, labelpad=4)

axes[1, 0].set_xlabel(r"$t/T$", fontsize=18, labelpad=2)
axes[1, 1].set_xlabel(r"$t/T$", fontsize=18, labelpad=2)

# ==============================
# Hide duplicated tick labels
# ==============================
axes[0, 0].tick_params(labelbottom=False)
axes[0, 1].tick_params(labelbottom=False)
axes[0, 1].tick_params(labelleft=False)
axes[1, 1].tick_params(labelleft=False)

# ==============================
# Population label
# ==============================
axes[0, 1].text(
    0.63,
    0.12,
    "Population",
    transform=axes[0, 1].transAxes,
    ha="left",
    va="center",
    fontsize=12,
    color="yellow",
)

fig.subplots_adjust(
    left=0.12,
    right=0.985,
    bottom=0.13,
    top=0.97,
    wspace=0.035,
    hspace=0.055,
)

plt.show()