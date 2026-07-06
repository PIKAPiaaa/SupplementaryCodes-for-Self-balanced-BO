
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
# Initial parameter values; F will be scanned later
F=0.12991
kappa=0.6
gainloss=0.16
#Dimensionless parameters in the WSL formula
deltaf=2*kappa/F
gf=gainloss/F
n = deltaf
m = gf  
# Coefficient matrix in the generating-function equation,u(t)/F
def A(t, m, n):
    return np.array([[1j*m,n * np.cos(t)],
                     [n * np.cos(t),1j*-m]], dtype=complex)

   


# Propagator equation: dY/dt = -i A(t) Y
def dYdt(t, Y, m, n):
    # Reshape the propagator into a 2x2 matrix
    Y_matrix = Y.reshape(2, 2)  
    A_t = A(t, m, n)  
    # Time derivative of the propagator
    dY_matrix = -1j*A_t.dot(Y_matrix) 
    return dY_matrix.flatten() 
# This setup can gives the propagator evolution for one parameter set

#%% Scan F and compute U11(pi)-U22(pi)
kappa=0.6
gainloss=0.16
# Scan 1/F through the variable 4*kappa/F, from 1 to 20, which means F/(4*kappa) from 1/81 to 1
reciprocalofF_values = np.linspace(1, 81, 4001)/(4*kappa)  
farray=1/reciprocalofF_values
# Store U11(pi)-U22(pi) for each F
U11_U22_diff= np.zeros(len(farray),dtype=complex)
t_values = np.linspace(0, np.pi, 100)
# # Solve the propagator equation for each F
for i, Ff in enumerate(farray):
    deltaf=2*kappa/Ff
    gf=gainloss/Ff
    n = deltaf
    m = gf 
    # Initial propagator: 2x2 identity matrix
    Y0 = np.array([1, 0, 0, 1], dtype=complex)  
    # Solve the propagator equation
    solution = integrate.solve_ivp(dYdt, (t_values[0], t_values[-1]), Y0, args=(m, n), t_eval=t_values)

    # Restore the 2x2 propagator Y(t)
    Y_t_values = solution.y.T.reshape(-1, 2, 2)  

   # Extract U11(pi)-U22(pi)
    U11_U22_diff[i]=Y_t_values[-1, 0, 0]-Y_t_values[-1, 1, 1]  

# The time evolution of the propagator can also be visualized if needed

#%%
# Build the local WSL spectrum from U11(pi)-U22(pi)

q = np.arcsin(U11_U22_diff / (2j)) / np.pi   # q = arcsin[(U11-U22)/2i]/pi
# Choose one reduced WSL cell for plotting.
# Only the imaginary part of E/kappa is used in Fig. 3.
# Therefore, the integer ladder offsets in Re(E/F) do not affect the plotted spectrum.
# Here we use two adjacent reduced branches, E_{0,-}/F and E_{1,+}/F.
Eresult = np.zeros((2, len(U11_U22_diff)), dtype=complex)
Fdiff=np.zeros(2)
result = np.arcsin(U11_U22_diff/(2j))/np.pi
n_minus = 0
n_plus = 1

Eresult[0, :] = -q + 2 * n_minus             # E_{n,-}/F = -q + 2n
Eresult[1, :] =  q + (2 * n_plus - 1)        # E_{n,+}/F =  q + (2n - 1)
#%% Plot Fig. 3(a) with F/(4kappa) as the x axis
# x axis: F/(4kappa)
xvals = farray / (4 * kappa)

# Sort by x for smooth lines
sort_idx = np.argsort(xvals)
xplot = xvals[sort_idx]

# Im(E)/kappa, since Eresult is E/F
yplot_0 = (Eresult[0, :].imag * farray / kappa)[sort_idx]
yplot_1 = (Eresult[1, :].imag * farray / kappa)[sort_idx]

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(7.0, 4), dpi=150)

ax.plot(xplot, yplot_0, color="#1f4ea8", lw=1.6)
ax.plot(xplot, yplot_1, color="#1f4ea8", lw=1.6)

# zero line
ax.axhline(0, color="black", lw=1.0, ls=(0, (4, 4)), alpha=0.85)

# axes
ax.set_xlabel(r"$F/4\kappa$", fontsize=18, labelpad=3)
ax.set_ylabel(r"$Im(E)/\kappa$", fontsize=18, labelpad=4)

ax.set_xlim(0, 1.0)
ax.set_ylim(-0.25, 0.25)

ax.set_xticks(np.linspace(0, 1.0, 6))
ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
ax.set_yticklabels([r"$-0.2$", r"$-0.1$", r"$0$", r"$0.1$", r"$0.2$"])

ax.tick_params(axis='both', labelsize=16, top=True, right=True, length=3)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()


#%% Plot a selected x range near 0 for the finer spectral structure, like the inset of Fig. 3(a)

# Set the x range here
x_start = 0.02
x_end = 0.075

fig, ax = plt.subplots(figsize=(4.0, 4), dpi=150)

ax.plot(xplot, yplot_0, color="#1f4ea8", lw=1.6)
ax.plot(xplot, yplot_1, color="#1f4ea8", lw=1.6)

# Zero line
ax.axhline(0, color="black", lw=1.0, ls=(0, (4, 4)), alpha=0.85)

# Axes
ax.set_xlabel(r"$F/4\kappa$", fontsize=18, labelpad=3)
ax.set_ylabel(r"$Im(E)/\kappa$", fontsize=18, labelpad=4)

ax.set_xlim(x_start, x_end)
ax.set_ylim(-0.1, 0.1)

ax.set_yticks([ -0.1, 0, 0.1,])
ax.set_xticks([ 0.025, 0.05, 0.075,])
ax.set_yticklabels([ r"$-0.1$", r"$0$", r"$0.1$"])
ax.tick_params(axis='x', labelsize=14, top=True, length=3)
ax.tick_params(axis='y', labelsize=16, right=True, length=3)

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()