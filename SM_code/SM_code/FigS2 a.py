
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
# kappa and gainloss are rescaled by pi/F to preserve the parameter ratios when F is set to pi
F =0.12
kappa = 0.6*np.pi/F
gainloss = 0.16*np.pi/F
# Construct the two Gaussian wave packets defined by the above parameters
for i in range (202):
    zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita)
for i in range (202):
    Zpump0[i]=np.exp(-(k0*(100-i)**2))*np.exp(1.0j*(i)*ksita1)
# Initial state formed by the superposition of the two Gaussian wave packets
zpump=1*np.array(zpump0+bi*Zpump0,dtype=complex)
#To match the Fig. S1(c),we normalize the initial state so that its maximum modulus is 1
zpump = zpump / np.max(np.abs(zpump))
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
# su stores the total intensity at each time,
su=np.zeros(len(t_eval),dtype=float)
#save the odd and even total intensity,with gain is even and loss is odd
suodd=np.zeros(len(t_eval),dtype=float)
sueven=np.zeros(len(t_eval),dtype=float)
for i in range(len(t_eval)):
    for j in range(202):
        su[i]=su[i]+abs(ww[j,i])**2
        if j%2==0:
            sueven[i]=sueven[i]+abs(ww[j,i])**2
        if j%2==1:
            suodd[i]=suodd[i]+abs(ww[j,i])**2
# Normalize the total intensity by its initial value 
su=su/su[0]

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
#Get the channel polarization C
C=np.zeros(len(t_eval),dtype=float)
for i in range(len(C)):
    C=(sueven-suodd)/(sueven+suodd)

#%% figure code
from matplotlib.gridspec import GridSpec
T = np.asarray(t_eval, dtype=float)
Z = np.abs(np.asarray(sol1.y))
y = np.arange(Z.shape[0], dtype=float)

vmin = 0.0
vmax = np.percentile(Z, 99.7)

# vertical guide lines: mark local extrema of |C(t)|
guide_centers = [0.5, 3.5]   # approximate expected locations
guide_window = 0.1         # search window around each center

def find_local_abs_extremum_time(T, C, center, window):
    mask = (T >= center - window) & (T <= center + window)
    if not np.any(mask):
        raise ValueError(f"No time points found around center={center}")
    idx_local = np.argmax(np.abs(C[mask]))
    return T[mask][idx_local]

t_guides = [
    find_local_abs_extremum_time(T, C, center, guide_window)
    for center in guide_centers
]

print("Guide-line times from |C(t)| extrema:", t_guides)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig = plt.figure(figsize=(6, 5.5), dpi=150)
gs = GridSpec(2, 1, height_ratios=[2.0, 1.15], hspace=0.20, figure=fig)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

# upper panel: evolution

im = ax0.imshow(
    Z,
    extent=[T[0], T[-1], y[0], y[-1]],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=1.4*vmax,
    interpolation="nearest"
)

# overlay total intensity
ax0r = ax0.twinx()
ax0r.plot(T, su, color="yellow", lw=1.2, zorder=6)
ax0r.set_ylim(0, 320)
ax0r.tick_params(axis='y', right=False, labelright=False)
ax0r.spines['right'].set_visible(False)

# dashed vertical lines
for tt in t_guides:
    ax0.axvline(tt, color="white", ls=(0, (4, 6)), lw=1.0, alpha=0.9, zorder=7)

ax0.set_xlim(0, 4)
ax0.set_ylim(50, 150)
ax0.set_yticks([50, 100, 150])
ax0.set_ylabel("Lattice number", fontsize=16, labelpad=8)
ax0.tick_params(axis='both', labelsize=12, top=True, right=False, length=2.5)
ax0.tick_params(labelbottom=False)

ax0.text(
    0.5, 1.02, "Amplified BO",
    transform=ax0.transAxes,
    ha="center", va="bottom",
    fontsize=16
)


# lower panel: C(t)

ax1.plot(T, C, color="#1f6fb2", lw=1.1)

for tt in t_guides:
    ax1.axvline(tt, color="0.3", ls=(0, (4, 6)), lw=1.0, alpha=0.9)

ax1.set_xlim(0, 4)
ax1.set_ylim(-1, 1)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels(["-1", "0", "1"])
ax1.set_xlabel(r"$t/T$", fontsize=18, labelpad=1)
ax1.set_ylabel(r"$C(t)$", fontsize=16, labelpad=8)
ax1.tick_params(axis='both', labelsize=12, top=True, right=False, length=2.5)

for ax in [ax0, ax1]:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

fig.subplots_adjust(left=0.18, right=0.955, bottom=0.12, top=0.92)
plt.show()