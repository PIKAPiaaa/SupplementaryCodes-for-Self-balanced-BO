

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.unicode_minus']=False 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.unicode_minus']=False 
# Time sampling points used to record the evolution from t=0 to t=4
t_eval = np.linspace(0,4,4001)
# Initial real-space lattice array for the first Gaussian wave packet
zpump0=np.zeros(202,dtype=complex)
# Parameters of the initial wave packet: packet width k0, central momenta ksita and ksita1,
# and the superposition coefficient bi between the two wave packets
k0=0.03
ksita=1.57
ksita1=ksita-np.pi
bi=-1
# initial lattice-site array for the second wavepack
Zpump0=np.zeros(202,dtype=complex)
# Construct the two Gaussian wave packets defined by the above parameters
for i in range (202):
    zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita)
for i in range (202):
    Zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita1)
# Initial state formed by the superposition of the two Gaussian wave packets
zpump=1*np.array(zpump0+bi*Zpump0,dtype=complex)
# Four-case evolution data
# This section assumes t_eval, zpump, and zpump0 are already defined above.

def PUMP(t, zpump, F, k, gamma):
    """Time-evolution equation."""
    npump = len(zpump)
    kn = np.zeros(npump, dtype=complex)
    PUMPuse = np.zeros((npump, npump), dtype=complex)

    for i in range(npump):
        kn[i] = (i - 100) * F
        if i % 2 == 0:
            PUMPuse[i, i] = kn[i] + 1j * gamma
        else:
            PUMPuse[i, i] = kn[i] - 1j * gamma

    for i in range(1, npump):
        PUMPuse[i, i - 1] = np.conjugate(k)
    for i in range(0, npump - 1):
        PUMPuse[i, i + 1] = k

    return (-1j) * np.dot(PUMPuse, zpump)


def evolve_case(F_raw, z0, t_eval):
    """Run one case using PUMP + partial."""
    kappa = 0.6 * np.pi / F_raw
    gainloss = 0.16 * np.pi / F_raw

    # Keep F=pi so the time axis is in Bloch periods
    f = partial(PUMP, F=np.pi, k=kappa, gamma=gainloss)

    sol = solve_ivp(f, [0, 4], z0, t_eval=t_eval, max_step=0.01)

    ww = np.abs(sol.y)
    su = np.sum(ww**2, axis=0)
    su = su / su[0]

    return {
        "F_raw": F_raw,
        "F_used": np.pi,
        "kappa": kappa,
        "gamma": gainloss,
        "f": f,
        "sol": sol,
        "ww": ww,
        "su": su,
    }


# F values of four zero point
F_list = [0.98560, 0.27230, 0.07648, 0.03764]

# Run 4 cases
case1 = evolve_case(F_list[0], zpump, t_eval)
case2 = evolve_case(F_list[1], zpump, t_eval)
case3 = evolve_case(F_list[2], zpump, t_eval)
case4 = evolve_case(F_list[3], zpump, t_eval)

# Keep separate arrays
sol1, ww1, su1 = case1["sol"], case1["ww"], case1["su"]
sol2, ww2, su2 = case2["sol"], case2["ww"], case2["su"]
sol3, ww3, su3 = case3["sol"], case3["ww"], case3["su"]
sol4, ww4, su4 = case4["sol"], case4["ww"], case4["su"]

# Also keep everything in one list
case_list = [case1, case2, case3, case4]
ww_list = [case["ww"] for case in case_list]
su_list = [case["su"] for case in case_list]

#%%
# Plot 2x2 panels

fig, axes = plt.subplots(2, 2, figsize=(15, 7.2), sharex=True, sharey=True)

titles = [
    f"1th zero, F={case1['F_raw']:.3f}",
    f"3th zero, F={case2['F_raw']:.3f}",
    f"10th zero, F={case3['F_raw']:.3f}",
    f"20 th zero, F={case4['F_raw']:.3f}",
]


# Common color scale
vmax = max(np.percentile(case["ww"], 99.7) for case in case_list)

# Population line settings for each panel
pop_y0_list = [40.0, 45.0, 50.0, 50.0]
pop_amp_list = [30.0, 30.0, 30.0, 30.0]

for ax, ww, su, title,  pop_y0, pop_amp in zip(
    axes.ravel(), ww_list, su_list, titles, pop_y0_list, pop_amp_list
):
    ax.imshow(
        ww,
        origin="lower",
        aspect="auto",
        extent=[t_eval[0], t_eval[-1], 0, ww.shape[0] - 1],
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Population curve
    su_plot = pop_y0 + pop_amp * (su / np.max(su))
    ax.plot(t_eval, su_plot, color="yellow", lw=1.2)

    su_mean = pop_y0 + pop_amp * (np.average(su) / np.max(su))
    ax.hlines(
        su_mean, t_eval[0], t_eval[-1],
        color="yellow", lw=1.0, linestyles=(0, (1.5, 3.0))
    )



    # Title
    ax.text(
        0.98, 0.88, title,
        transform=ax.transAxes,
        fontsize=13,
        color="white",
        fontweight="bold",
        ha="right"
    )

    ax.set_xlim(0, 4)
    ax.set_ylim(50, 150)
    ax.tick_params(direction="in", top=True, right=True, labelsize=11)

# Axis labels
axes[0, 0].set_ylabel("Lattice number", fontsize=16)
axes[1, 0].set_ylabel("Lattice number", fontsize=16)
axes[1, 0].set_xlabel(r"$t/T$", fontsize=18)
axes[1, 1].set_xlabel(r"$t/T$", fontsize=18)

plt.tight_layout()
plt.show()
