
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
F =0.12991
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




#%% Odd/even sublattice dynamics figure
import numpy as np
import matplotlib.pyplot as plt

# Time axis
x = np.asarray(t_eval, dtype=float)

# Sublattice data
odd_rows = np.abs(sol1.y[1::2, :])
even_rows = np.abs(sol1.y[::2, :])

# Lattice coordinates
y = np.arange(sol1.y.shape[0], dtype=float)
y_odd = y[1::2]
y_even = y[::2]

X_odd, Y_odd = np.meshgrid(x, y_odd)
X_even, Y_even = np.meshgrid(x, y_even)

# Color range
vmin = 0.0
vmax = np.percentile(np.abs(sol1.y), 99.5)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(
    2, 1,
    figsize=(5, 4.2),
    sharex=True,
    dpi=150
)

# Top: odd
ax[0].pcolormesh(
    X_odd, Y_odd, odd_rows,
    cmap="viridis",
    shading="auto",
    vmin=vmin,
    vmax=vmax*1.5
)
ax[0].set_xlim(0, 2)
ax[0].set_ylim(70, 130)
ax[0].set_yticks([70, 100, 130])
ax[0].tick_params(axis='both', labelsize=16, top=True, right=True, length=2.5)
ax[0].tick_params(labelbottom=False)

# Bottom: even
ax[1].pcolormesh(
    X_even, Y_even, even_rows,
    cmap="viridis",
    shading="auto",
    vmin=vmin,
    vmax=vmax*1.5
)
ax[1].set_xlim(0, 2)
ax[1].set_ylim(70, 130)
ax[1].set_xticks([0, 1, 2])
ax[1].set_yticks([70, 100])
ax[1].tick_params(axis='both', labelsize=16, top=True, right=True, length=2.5)
ax[1].set_xlabel(r"$t/T$", fontsize=20, labelpad=2)

# Common y label
fig.text(0.02, 0.6, "Lattice number", rotation=90,
         va="center", fontsize=20)

# Sublattice labels
ax[0].text(
    -0.15, 0.50, "odd",
    transform=ax[0].transAxes,
    rotation=90,
    ha="center", va="center",
    fontsize=20, color="#1f4ea8"
)

ax[1].text(
    -0.15, 0.50, "even",
    transform=ax[1].transAxes,
    rotation=90,
    ha="center", va="center",
    fontsize=20, color="red"
)
plt.subplots_adjust(
    left=0.22,
    right=0.98,
    bottom=0.12,
    top=0.98,
    hspace=0.00
)
plt.show()