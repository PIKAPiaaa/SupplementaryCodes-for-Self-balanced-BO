
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from functools import partial
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
ksita1=-1.58
bi=-1
# initial lattice-site array for the second wavepack
Zpump0=np.zeros(202,dtype=complex)
# Dimensionless parameters of the Hamiltonian, 
# kappa and gainloss are rescaled by 2pi/F to preserve the parameter ratios when F is set to 2pi
F =0.13
kappa = 0.6*np.pi/F
gainloss = 0.16*np.pi/F
# Construct the two Gaussian wave packets defined by the above parameters
for i in range (202):
    zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita)
for i in range (202):
    Zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita1)
# Initial state formed by the superposition of the two Gaussian wave packets
zpump=1*np.array(zpump0+bi*Zpump0,dtype=complex)
# Define the time-evolution equation dz/dt = -i H z
# where H is the real-space tight-binding Hamiltonian including the linear potential kn,
# alternating gain/loss gamma, and nearest-neighbor hopping k
def PUMP(t,zpump,F,k,gamma):
    npump=len(zpump)
    kn=np.zeros(npump,dtype=complex)
    PUMPuse=np.zeros([npump,npump],dtype=complex)
    PUMPuse0=np.zeros([npump,npump],dtype=complex)
    for i in range(len(zpump0)):
        kn[i]=(i-100)*F
        if i%2==0:
            PUMPuse[i,i]=kn[i]+1j*gamma
        else:
            PUMPuse[i,i]=kn[i]-1j*gamma
    for i in range(1,npump):
        PUMPuse[i,i-1]=k.conjugate()
    for i in range(0,npump-1):
        PUMPuse[i,i+1]=k
    PUMPuse0=np.dot(PUMPuse,zpump)*(-1j)
    return PUMPuse0
# Use partial to fix the parameters and obtain an evolution function depending on t and zpump
# Set F = pi so that the time unit matches the Bloch-oscillation period
f = partial(PUMP,F=np.pi,k=kappa,gamma=gainloss)
# Numerically integrate the time evolution using solve_ivp;
#to avoid using a step size that is too large relative to the parameter scale, set the maximum step size to 0.05
sol1= solve_ivp(f, [0,4], zpump, t_eval=t_eval,max_step=0.05)
#ww stores the modulus (amplitude)
ww=abs(sol1.y)

#%% Fig. 2(f): snapshots at four times
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Time axis
Tarr = np.asarray(t_eval, dtype=float)

# Wavefunction amplitude
A = np.abs(np.asarray(sol1.y))

# Choose four times in units of T
target_times = [0.0, 0.5, 1.5, 1.0]
time_labels = [r"$t=0$", r"$T/2$", r"$3T/2$", r"$T$"]

time_ids = [
    np.argmin(np.abs(Tarr - tt))
    for tt in target_times
]

# Lattice-index window displayed in the figure
n_min, n_max = 80, 120
sites = np.arange(n_min, n_max + 1)

# Site convention:
# The Python array uses zero-based indexing.
# Even array indices correspond to the gain sublattice (+i*gamma),
# while odd array indices correspond to the loss sublattice (-i*gamma).
# Under one-based physical site numbering, the gain and loss sublattices
# correspond to odd- and even-numbered sites, respectively.
gain_sites = sites[sites % 2 == 0]
loss_sites = sites[sites % 2 == 1]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axes = plt.subplots(
    2, 2,
    figsize=(9.0, 4.8),
    dpi=150
)
axes = axes.ravel()

# Use the same vertical range in all four panels
y_max = 1.08 * np.max(A[sites][:, time_ids])

for ax, tid, lab in zip(axes, time_ids, time_labels):
    amp = A[:, tid]

    amp_gain = amp[gain_sites]
    amp_loss = amp[loss_sites]

    # Gain sublattice: even Python indices
    ax.vlines(
        gain_sites,
        0,
        amp_gain,
        color="red",
        lw=1.1
    )
    ax.scatter(
        gain_sites,
        amp_gain,
        s=42,
        facecolors="#d8a3a3",
        edgecolors="red",
        linewidths=1.1,
        zorder=3
    )

    # Loss sublattice: odd Python indices
    ax.vlines(
        loss_sites,
        0,
        amp_loss,
        color="#1f4ea8",
        lw=1.1
    )
    ax.scatter(
        loss_sites,
        amp_loss,
        s=42,
        facecolors="#8ea0c8",
        edgecolors="#1f4ea8",
        linewidths=1.2,
        zorder=3
    )

    ax.axhline(0, color="black", lw=1.6)

    ax.set_xlim(n_min - 0.8, n_max + 0.8)
    ax.set_ylim(0, y_max)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.text(
        0.50,
        -0.10,
        lab,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=24
    )

legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor="#d8a3a3",
        markeredgecolor="red",
        markeredgewidth=1.1,
        markersize=7,
        label="gain"
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor="#8ea0c8",
        markeredgecolor="#1f4ea8",
        markeredgewidth=1.2,
        markersize=7,
        label="loss"
    ),
]

axes[1].legend(
    handles=legend_handles,
    frameon=False,
    fontsize=20,
    loc="upper right",
    ncol=2,
    handlelength=0.0,
    columnspacing=1.4
)

plt.subplots_adjust(
    left=0.05,
    right=0.98,
    bottom=0.10,
    top=0.95,
    wspace=0.16,
    hspace=0.38
)

plt.show()