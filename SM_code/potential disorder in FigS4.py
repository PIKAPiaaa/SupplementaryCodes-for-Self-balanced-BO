import matplotlib as mpl
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

def run_kgapBO_process(
    psi0,
    t_eval,
    F,
    kappa,
    gamma,
    deltakappa=0.0,
    disorder_strength=0.0,     # real on-site disorder strength
    disorder_seed=0,           # fixed seed
    t_span=(0.0, 4.0),
    max_step=0.05,
    center_index=100,
    compute_fft=False,
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
    onsite_disorder = disorder_strength * disorder_base

    def rhs(t, psi):
        H = np.zeros((N, N), dtype=complex)

        # on-site
        idx = np.arange(N)
        kn = (idx - center_index) * F
        H[idx, idx] = kn + onsite_disorder + 1j * ((-1) ** idx) * gamma

        # hopping
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
        "onsite_disorder": onsite_disorder,
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
fixed_seed = 1
W_list = [0.04, 0.06, 0.08] # disorder strength

results = []
for W in W_list:
    out = run_kgapBO_process(
        psi0=zpump,
        t_eval=t_eval,
        F=np.pi,
        kappa=kappa,
        gamma=gainloss,
        deltakappa=0.0,
        disorder_strength=W*kappa,# disorder strength with the unit kappa
        disorder_seed=fixed_seed,
        compute_fft=False,
    )
    results.append(out)

# Separate arrays
out1, out2, out3 = results
ww1, ww2, ww3 = out1["ww"], out2["ww"], out3["ww"]
su1, su2, su3 = out1["su"], out2["su"], out3["su"]


#%%

fig, axes = plt.subplots(
    1, 3,
    figsize=(19.2, 5.5),
    sharex=True,
    sharey=True
)

# lattice window
y_idx = np.arange(50, 151)
X, Y = np.meshgrid(t_eval, y_idx)

# Common color scale
Z_all = [np.abs(out["y"][50:151, :]) for out in results]
vmax = max(np.percentile(Z, 99.5) for Z in Z_all)
for ax, out, Z in zip(axes, results, Z_all):

    # Density plot
    im = ax.pcolormesh(
        X, Y, Z,
        cmap="viridis",
        shading="gouraud",
        vmin=0.0,
        vmax=1.2 * vmax
    )

    # Convert actual disorder strength back to percentage of kappa

    dV_percent = 100.0 * out["disorder_strength"] / kappa

    # White title inside each panel
    ax.text(
        0.5, 0.94,
        rf"$\Delta V=\pm {dV_percent:.0f}\%\,\kappa$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=20,
        color="white"
    )

    # Axes range and ticks
    ax.set_xlim(0, 4)
    ax.set_ylim(50, 150)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([50, 100, 150])

    ax.tick_params(
        axis="both",
        which="both",
        labelsize=20,
        direction="in",
        length=3.0,
        width=0.8
    )

    # Yellow total-intensity curve
    ax2 = ax.twinx()
    su = out["su"].astype(float)
    su_norm = su / su[0]

    ax2.plot(
        t_eval,
        su_norm,
        color="yellow",
        lw=1.1
    )

    # Adjust this if the yellow curve is too high/low
    ax2.set_ylim(0, 15)

    # Hide right axis
    ax2.set_yticks([])
    ax2.tick_params(axis="y", right=False, labelright=False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
axes[0].set_ylabel("Lattice number", fontsize=20)
for ax in axes:
    ax.set_xlabel(r"$t/T$", fontsize=20)
# Hide repeated y tick labels on the second and third panels
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

# Left row label
fig.text(
    0.045, 0.56,
    "Potential\nDisorder",
    ha="center",
    va="center",
    fontsize=20
)

# Layout close to your reference figure
fig.subplots_adjust(
    left=0.12,
    right=0.985,
    bottom=0.21,
    top=0.96,
    wspace=0.14
)

plt.show()