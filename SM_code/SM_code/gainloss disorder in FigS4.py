import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Time-evolution function with gain/loss disorder
def run_kgapBO_process_gainloss_disorder(
    psi0,
    t_eval,
    F,
    kappa,
    gamma,
    deltakappa=0.0,
    disorder_strength=0.0,   # gain/loss disorder amplitude
    disorder_seed=0,         # fixed seed
    t_span=(0.0, 4.0),
    max_step=0.05,
    center_index=100,
):
    """
    Input: psi0 + parameters
    Output: evolution data in a dict
    """
    psi0 = np.asarray(psi0, dtype=complex)
    N = psi0.size

    # Fixed random pattern, only amplitude changes
    rng = np.random.default_rng(disorder_seed)
    disorder_base = rng.uniform(-1.0, 1.0, size=N)
    gainloss_disorder = disorder_strength * disorder_base

    def rhs(t, psi):
        H = np.zeros((N, N), dtype=complex)

        idx = np.arange(N)
        kn = (idx - center_index) * F

        # Gain/loss disorder on the diagonal
        gamma_site = gamma + gainloss_disorder
        sign = np.where(idx % 2 == 0, 1.0, -1.0)
        H[idx, idx] = kn + 1j * sign * gamma_site

        # Hopping
        i = np.arange(1, N)
        H[i, i - 1] = np.conjugate(kappa) + ((-1) ** i) * deltakappa
        i = np.arange(0, N - 1)
        H[i, i + 1] = kappa + ((-1) ** i) * deltakappa

        return (-1j) * (H @ psi)

    sol = solve_ivp(rhs, t_span, psi0 * 1j, t_eval=t_eval, max_step=max_step)

    ww = np.abs(sol.y)
    intensity = ww ** 2

    su = intensity.sum(axis=0).astype(float)
    sueven = intensity[0::2, :].sum(axis=0).astype(float)
    suodd = intensity[1::2, :].sum(axis=0).astype(float)

    out = {
        "sol": sol,
        "t": sol.t,
        "y": sol.y,
        "ww": ww,
        "intensity": intensity,
        "su": su,
        "sueven": sueven,
        "suodd": suodd,
        "disorder_strength": disorder_strength,
        "disorder_seed": disorder_seed,
        "gainloss_disorder": gainloss_disorder,
    }

    return out


#%%
# Initial state and parameters
t_eval = np.linspace(0, 4, 4001)

zpump0 = np.hstack([np.zeros(100), 1, np.zeros(101)]).astype(complex)
Zpump0 = np.zeros(202, dtype=complex)

k0 = 0.03
ksita = 0.0
ksita1 = -0.5 * np.pi
bi = 0.0

F0 = 0.13
kappa = 0.6 * np.pi / F0
gainloss = 0.16 * np.pi / F0

for i in range(202):
    zpump0[i] = np.exp(-(k0 * (100 - i) ** 2)) * np.exp(1.0j * i * ksita)

for i in range(202):
    Zpump0[i] = np.exp(-(k0 * (100 - i) ** 2)) * np.exp(1.0j * i * ksita1)

zpump = np.array(zpump0 + bi * Zpump0, dtype=complex)


#%%
# Sweep disorder strength with the same seed
fixed_seed = 30

# You can change these three values freely
dgamma_ratio_list = [0.01, 0.10, 0.15]   # disorder strength in units of gamma

results = []
for ratio in dgamma_ratio_list:
    out = run_kgapBO_process_gainloss_disorder(
        psi0=zpump,
        t_eval=t_eval,
        F=np.pi,
        kappa=kappa,
        gamma=gainloss,
        deltakappa=0.0,
        disorder_strength=ratio * gainloss,
        disorder_seed=fixed_seed,
    )
    out["disorder_ratio"] = ratio
    results.append(out)

# Separate arrays
out1, out2, out3 = results
ww1, ww2, ww3 = out1["ww"], out2["ww"], out3["ww"]
su1, su2, su3 = out1["su"], out2["su"], out3["su"]
#%%
# Plot 1x3 panels for gain/loss disorder

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.unicode_minus": False,
})

fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), dpi=150, sharex=True, sharey=True)

y_idx = np.arange(50, 151)
X, Y = np.meshgrid(t_eval, y_idx)

Z_all = [np.abs(out["y"][50:151, :]) for out in results]
vmax = max(np.percentile(Z, 99.5) for Z in Z_all)

for ax, out, Z in zip(axes, results, Z_all):
    ax.pcolormesh(X, Y, Z, cmap="viridis", shading="gouraud",
                  vmin=0.0, vmax=1.2 * vmax)

    ax.text(0.5, 0.94,
            rf"$\Delta\gamma=\pm {100*out['disorder_ratio']:.0f}\%\,\gamma$",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=20, color="white")

    ax.set_xlim(0, 4)
    ax.set_ylim(50, 150)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([50, 100, 150])
    ax.tick_params(axis="both", labelsize=18, top=False, right=False, length=4)

    ax2 = ax.twinx()
    su_norm = out["su"].astype(float) / out["su"][0]
    ax2.plot(t_eval, su_norm, color="yellow", lw=1.3)
    ax2.set_ylim(0, 15)
    ax2.set_yticks([])
    ax2.tick_params(axis="y", right=False, labelright=False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

axes[0].set_ylabel("Lattice number", fontsize=18, labelpad=4)
for ax in axes:
    ax.set_xlabel(r"$t/T$", fontsize=20, labelpad=4)

for ax in axes[1:]:
    ax.tick_params(labelleft=False)

fig.text(0.05, 0.57, "Gain/Loss\nDisorder",
         ha="center", va="center", fontsize=20)

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

fig.subplots_adjust(left=0.16, right=0.985, bottom=0.22, top=0.92, wspace=0.14)

plt.show()