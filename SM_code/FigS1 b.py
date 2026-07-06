
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
k0=0.001
ksita=np.pi
ksita1=0
bi=0
# initial lattice-site array for the second wavepack
Zpump0=np.zeros(202,dtype=complex)
# Dimensionless parameters of the Hamiltonian
F =0
kappa = 0.6*np.pi
gainloss = 0.16*np.pi
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
f = partial(PUMP,F=F,k=kappa,gamma=gainloss)
# Numerically integrate the time evolution using solve_ivp;
#to avoid using a step size that is too large relative to the parameter scale, set the maximum step size to 0.05
sol1= solve_ivp(f, [0,4], zpump, t_eval=t_eval,max_step=0.05)
#ww stores the modulus (amplitude)
ww=abs(sol1.y)
# su stores the total intensity at each time,
su=np.zeros(len(t_eval),dtype=float)

for i in range(len(t_eval)):
    for j in range(202):
        su[i]=su[i]+abs(ww[j,i])**2
# Normalize the total intensity by its initial value 
su=su/su[0]

#%%Data processing


# Extract the normalized real part of the initial state for plotting
# for the present initial state,it has no imaginary part
psi0_amp = zpump.real
y0 = np.arange(psi0_amp.size, dtype=float)
# Perform the Fourier transform along the lattice direction to obtain the momentum-space distribution
freq=np.fft.fftfreq(len(sol1.y[:,1]),1)
fft_result = np.fft.fft(sol1.y, axis=0)
magnitude = fft_result


# Get the sorting indices of the frequency array
sorted_indices = np.argsort(freq)
# Reorder the frequency array and the corresponding Fourier result
freq_sorted = freq[sorted_indices]
fft_result_sorted = fft_result[sorted_indices, :]



#%%figure code
from matplotlib.gridspec import GridSpec

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

fig = plt.figure(figsize=(8.0, 4.2), dpi=150)
gs = GridSpec(1, 2, width_ratios=[1.0, 2.05], wspace=0.09, figure=fig)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)

# figure of initial state
ax0.fill_betweenx(
    y0, 0.0, psi0_amp,
    color="#1f6fb2",
    lw=0
)
ax0.plot(psi0_amp, y0, color="#1f6fb2", lw=1.0)
ax0.axvline(0.0, color="#1f6fb2", ls="--", lw=1.0, alpha=0.9)
ymax = max(y[-1], y0[-1])
ax0.set_xlim(-1, 1)
ax0.set_ylim(0, ymax)
if ymax >= 190:
    yticks = [0, 100, 200]
else:
    yticks = [0, int(round(ymax / 2)), int(round(ymax))]
ax0.set_xticks([-1, 0, 1])
ax0.set_yticks(yticks)

ax0.set_xlabel("Amplitude", fontsize=17, labelpad=4)
ax0.set_ylabel("Lattice number", fontsize=18, labelpad=10)

ax0.tick_params(axis='both', labelsize=13, top=True, right=True, length=2.0)

ax0.text(
    0.50, 0.90, "Initial state",
    transform=ax0.transAxes,
    ha="center", va="bottom",
    fontsize=14
)
# figure of dynamic
im = ax1.imshow(
    Z,
    extent=[T[0], T[-1], y[0], y[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest"
)
ax1.plot(T, su , color="#d8cf00", lw=0.8, zorder=5)
ax1.set_xlim(T[0], T[-1])
ax1.set_ylim(0, ymax)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xlabel(r"$t/T_b$", fontsize=20, labelpad=-1)
ax1.tick_params(axis='both', labelsize=13, top=True, right=False, length=2.0)
ax1.tick_params(labelleft=False)
ax1.text(
    0.50, 0.90, "Sysmetric",
    transform=ax1.transAxes,
    ha="center", va="bottom",
    fontsize=16,
    color="white"
)
for ax in [ax0, ax1]:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
fig.subplots_adjust(left=0.11, right=0.985, bottom=0.18, top=0.92, wspace=0.09)
plt.show()