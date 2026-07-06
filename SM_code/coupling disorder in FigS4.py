# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 15:53:34 2026

@author: 31690
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def run_kgapBO_process(
    psi0,
    t_eval,
    F,
    kappa,                 
    gamma,
    deltakappa=0.0,                                                     #disorder
    t_span=(0.0, 5.0),
    max_step=0.05,
    center_index=100,
    compute_fft=True,
):
    psi0 = np.asarray(psi0, dtype=complex)
    N = psi0.size


    if np.ndim(kappa) == 0:
        kappa_bond = np.full(N-1, kappa, dtype=complex)     
    else:
        kappa_bond = np.asarray(kappa, dtype=complex)
        if kappa_bond.shape != (N-1,):
            raise ValueError(f"kappa must be scalar or shape (N-1,), got {kappa_bond.shape}")

    def rhs(t, psi):
        H = np.zeros((N, N), dtype=complex)

        # on-site
        idx = np.arange(N)
        kn = (idx - center_index) * F
        H[idx, idx] = kn + 1j * ((-1)**idx) * gamma

        # hopping with bond-dependent kappa_bond[i] for (i <-> i+1)
        i = np.arange(N-1)
        # upper diagonal: H[i, i+1]
        H[i, i+1] = kappa_bond[i] + ((-1)**i) * deltakappa
        # lower diagonal: H[i+1, i]
        H[i+1, i] = np.conjugate(kappa_bond[i]) + ((-1)**(i+1)) * deltakappa

        return (-1j) * (H @ psi)

    sol = solve_ivp(rhs, t_span, psi0 * 1j, t_eval=t_eval, max_step=max_step)

    ww = np.abs(sol.y)
    intensity = ww**2
    su = intensity.sum(axis=0).astype(float)
    sueven = intensity[0::2, :].sum(axis=0).astype(float)
    suodd  = intensity[1::2, :].sum(axis=0).astype(float)

    out = {
        "sol": sol,
        "t": sol.t,
        "y": sol.y,
        "ww": ww,
        "intensity": intensity,
        "su": su,
        "sueven": sueven,
        "suodd": suodd,
        "kappa_bond": kappa_bond,        #disorder Array
        "params": dict(F=F, gamma=gamma, deltakappa=deltakappa,
                       t_span=t_span, max_step=max_step, center_index=center_index),
    }


    return out
#%%
t_eval = np.linspace(0,5,5001)
zpump0=np.hstack([np.zeros(100),1,np.zeros(101)],dtype=complex)
plt.rcParams['figure.facecolor'] = 'white'

k0=0.03
ksita=0
ksita1=-0.5*np.pi
bi=0
Zpump0=np.zeros(202,dtype=complex)
F =0.12991
kappa = 0.6*np.pi/F 
gainloss = 0.16*np.pi/F 

for i in range (202):
    zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita)
#zpump0[50]=np.exp((np.pi*0.5)*1j)
for i in range (202):
    Zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita1)
#Zpump0[50]=1
zpump=np.array(zpump0+bi*Zpump0,dtype=complex)

#%%
# Sweep coupling-disorder strength with the same seed
fixed_seed = 40

# You can change these three values freely
dkappa_ratio_list = [0.01, 0.03, 0.05]   # disorder strength in units of kappa

results = []

kappa0 = kappa
N = len(zpump)

# Fixed random pattern, only amplitude changes
rng = np.random.default_rng(fixed_seed)
disorder_base = rng.uniform(-1.0, 1.0, size=N - 1)

for ratio in dkappa_ratio_list:
    delta = ratio * disorder_base
    kappa_bond = kappa0 * (1.0 + delta)

    out = run_kgapBO_process(
        psi0=zpump,
        t_eval=t_eval,
        F=np.pi,
        kappa=kappa_bond,
        gamma=gainloss,
        deltakappa=0,
        compute_fft=False,
    )

    out["disorder_ratio"] = ratio
    out["disorder_seed"] = fixed_seed
    results.append(out)

# Separate arrays
out1, out2, out3 = results
ww1, ww2, ww3 = out1["ww"], out2["ww"], out3["ww"]
su1, su2, su3 = out1["su"], out2["su"], out3["su"]



#%% Plot 1x3 panels for coupling disorder
import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["mathtext.rm"] = "Times New Roman"
mpl.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), dpi=150, sharex=True, sharey=True)

y_idx = np.arange(50, 151)
X, Y = np.meshgrid(t_eval, y_idx)

# Common color scale
Z_all = [np.abs(out["y"][50:151, :]) for out in results]
vmax = max(np.percentile(Z, 99.5) for Z in Z_all)

for ax, out, Z in zip(axes, results, Z_all):
    # density plot
    ax.pcolormesh(
        X, Y, Z,
        cmap="viridis",
        shading="gouraud",
        vmin=0.0,
        vmax=1.2 * vmax
    )

    # in-panel title
    ax.text(
        0.5, 0.94,
        rf"$\Delta\kappa=\pm {100*out['disorder_ratio']:.0f}\%\,\kappa$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=20,
        color="white"
    )

    ax.set_xlim(0, 4)
    ax.set_ylim(50, 150)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([50, 100, 150])
    ax.tick_params(axis='both', labelsize=18, top=False, right=False, length=4)

    # yellow curve on right axis
    ax2 = ax.twinx()
    su = out["su"].astype(float)
    su_norm = su / su[0]

    ax2.plot(t_eval, su_norm, color="yellow", lw=1.3)
    ax2.set_ylim(0, 15)

    # hide right axis
    ax2.set_yticks([])
    ax2.tick_params(axis="y", right=False, labelright=False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

# labels
axes[0].set_ylabel("Lattice number", fontsize=18, labelpad=4)
for ax in axes:
    ax.set_xlabel(r"$t/T$", fontsize=20, labelpad=4)

# hide repeated y tick labels
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

# left-side row label
fig.text(
    0.05, 0.57,
    "Coupling\nDisorder",
    ha="center",
    va="center",
    fontsize=20
)

# spine width
for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

fig.subplots_adjust(
    left=0.16,
    right=0.985,
    bottom=0.22,
    top=0.92,
    wspace=0.14
)

plt.show()