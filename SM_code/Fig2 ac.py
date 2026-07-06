
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
bi=1
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

#%%Data processing
# Perform the Fourier transform along the lattice direction to obtain the momentum-space distribution
freq=np.fft.fftfreq(len(sol1.y[:,1]),1)
fft_result = np.fft.fft(sol1.y, axis=0)
magnitude = fft_result


# Get the sorting indices of the frequency array
sorted_indices = np.argsort(freq)
# Reorder the frequency array and the corresponding Fourier result
freq_sorted = freq[sorted_indices]
fft_result_sorted = fft_result[sorted_indices, :]


#%%code of dynamic in Fig. 2(a,c) 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


T = np.asarray(t_eval, dtype=float)

# upper panel: real-space evolution
Z_top = np.abs(np.asarray(sol1.y)[50:151, :])
y_top = np.arange(50, 151, dtype=float)

# lower panel: momentum-space evolution
ka = 2.0 * np.pi * np.asarray(freq_sorted, dtype=float)
Z_k = np.abs(np.asarray(fft_result_sorted))

# color ranges
vmin_top = 0.0
vmax_top = np.percentile(Z_top, 99.5)

vmin_k = 0.0
vmax_k = np.percentile(Z_k, 99.7)


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig = plt.figure(figsize=(6, 5.55), dpi=150)
gs = GridSpec(2, 1, height_ratios=[1.0, 0.82], hspace=0.30, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)


# panel (a): real-space evolution

im0 = ax0.imshow(
    Z_top,
    extent=[T[0], T[-1], y_top[0], y_top[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin_top,
    vmax=vmax_top*1.2,
    interpolation="nearest"
)

# vertical dashed line
ax0.axvline(1.0, color="white", ls=(0, (4, 4)), lw=1.0, alpha=0.9, zorder=7)

ax0.set_xlim(0, 4)
ax0.set_ylim(50, 150)
ax0.set_yticks([50, 100, 150])
ax0.set_xticks([0, 1, 2, 3, 4])
ax0.tick_params(axis='both', labelsize=16, top=True, right=True, length=2.5)
ax0.tick_params(labelbottom=True)
ax0.set_xlabel(r"$t/T$", fontsize=20, labelpad=1)
ax0.set_ylabel("Lattice number", fontsize=18, labelpad=8)

ax0.text(
    0.03, 0.95, "Seeding in gain channel",
    transform=ax0.transAxes,
    ha="left", va="top",
    fontsize=17,
    color="white"
)
# panel (c): momentum-space evolution
im1 = ax1.imshow(
    Z_k,
    extent=[T[0], T[-1], ka[0], ka[-1]],
    origin="lower",
    aspect="auto",
    cmap="plasma",
    vmin=vmin_k,
    vmax=vmax_k,
    interpolation="nearest"
)

# vertical dashed line
ax1.axvline(1.0, color="white", ls=(0, (4, 4)), lw=1.0, alpha=0.9, zorder=7)

ax1.set_xlim(0, 4)
ax1.set_ylim(-np.pi, np.pi)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

ax1.set_xlabel(r"$t/T$", fontsize=20, labelpad=1)
ax1.set_ylabel(r"$ka$", fontsize=18, labelpad=4)
ax1.tick_params(axis='both', labelsize=16, top=True, right=True, length=2.5)


# panel labels
fig.text(0.03, 0.94, "(a)", fontsize=20, fontweight="bold")
fig.text(0.03, 0.45, "(c)", fontsize=20, fontweight="bold")

for ax in [ax0, ax1]:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

fig.subplots_adjust(left=0.19, right=0.965, bottom=0.10, top=0.94, hspace=0.30)
plt.show()
