
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
ksita=0
ksita1=0
bi=0
# initial lattice-site array for the second wavepack
Zpump0=np.zeros(202,dtype=complex)
# Dimensionless parameters of the Hamiltonian, 
# kappa and gainloss are rescaled by 2pi/F to preserve the parameter ratios when F is set to 2pi
F =0.13
kappa = 0.6*2*np.pi/F
gainloss = 0.0*np.pi/F
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
# Set F = 2pi so that the time unit matches the Bloch-oscillation period
f = partial(PUMP,F=2*np.pi,k=kappa,gamma=gainloss)
# Numerically integrate the time evolution using solve_ivp;
#to avoid using a step size that is too large relative to the parameter scale, set the maximum step size to 0.05
sol1= solve_ivp(f, [0,4], zpump, t_eval=t_eval,max_step=0.05)
#ww stores the modulus (amplitude)
ww=abs(sol1.y)




#%% figure code
T = np.asarray(t_eval, dtype=float)
Z = np.abs(np.asarray(sol1.y))
y = np.arange(Z.shape[0], dtype=float)

vmin = 0.0
vmax = np.percentile(Z, 99.7)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(9, 6))

im = ax.imshow(
    Z,
    extent=[T[0], T[-1], y[0], y[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest"
)

ax.set_xlim(0, 4)
ax.set_ylim(50, 150)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_yticks([50, 100, 150])

ax.set_xlabel(r"$t/T$", fontsize=28, labelpad=1)
ax.set_ylabel("Lattice number", fontsize=20, labelpad=6)

ax.tick_params(axis='both', labelsize=24, top=True, right=True, length=2.5)

ax.text(
    0.03, 0.94, "Normal BO",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=24,
    color="white"
)

ax.text(
    0.03, 0.10, r"No $k$-gap",
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=24,
    color="white"
)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.96)
plt.show()