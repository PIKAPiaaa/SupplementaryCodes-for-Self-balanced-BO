
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
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
gainloss=0.3
# Scan 1/F through the variable 4*kappa/F, from 1 to 91, which means F/(4*kappa) from 1/91 to 1
reciprocalofF_values = np.linspace(1, 91, 4501)/(4*kappa)  
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


# Build the reduced local WSL branches E/F
# Only the imaginary part is used in Fig.3, so the real offsets do not affect the plot
Eresult=np.zeros((2,len(U11_U22_diff)),dtype=complex)
Fdiff=np.zeros(2)
result = np.arcsin(U11_U22_diff/(2j))/np.pi
for i in range(Eresult.shape[0]):
    Fdiff[i]=1*i
for i in range(Eresult.shape[1]):
    for j in range(Eresult.shape[0]): 
        Eresult[j, i] = Fdiff[j] +(-1)**(j+1)*result[i]





#%% get the zero point of Eimag
zeros = []
x = farray/kappa
y = Eresult.T.imag[:, 0] * (farray/kappa)

for j in range(len(y)-1):
    if y[j]*y[j+1] < 0:  # 异号
        # 线性插值求零点
        x0 = x[j] - y[j]*(x[j+1]-x[j])/(y[j+1]-y[j])
        zeros.append(x0)
zeros=np.array(zeros)
#%% Plot zeros vs order

from scipy.optimize import curve_fit
# Order of zeros
order = np.arange(1, len(zeros) + 1)

# Main-axis data: F/(4*kappa)
zeros_y = zeros / 4.0

# Main fit
def fit_func(n, a, b, c):
    return a / (n + b)**c

popt, _ = curve_fit(fit_func, order, zeros_y, p0=[0.5, 0.0, 1.0], maxfev=20000)
order_fit = np.linspace(0.5, len(order), 400)
zeros_fit = fit_func(order_fit, *popt)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=150)

# Main plot
ax.scatter(order, zeros_y, s=30, color="red", label="Zeros", zorder=3)
ax.plot(order_fit, zeros_fit, color="#1f4ea8", lw=1.8, ls=(0, (1.2, 1.2)), label="Fit.")
ax.set_xlabel("Order of zeros", fontsize=24, labelpad=4)
ax.set_ylabel(r"$F/4\kappa$", fontsize=24, labelpad=6)
ax.set_xlim(0, 25)
ax.set_ylim(0, 0.5)
ax.set_xticks([0, 5, 10, 15, 20, 25])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels([r"$0$", r"$0.1$", r"$0.2$", r"$0.3$", r"$0.4$", r"$0.5$"])
ax.tick_params(axis='both', labelsize=18, top=True, right=True, length=4)

# Legend position
ax.legend(
    frameon=False,
    fontsize=22,
    loc="upper left",
    bbox_to_anchor=(0.14, 0.98),
    handlelength=1.3,
    handletextpad=0.4,
    borderaxespad=0.0
)


# Inset: 4*kappa/F vs order
# use the 10th to 19th zeros
idx0, idx1 = 9, 19   # Python indices: 10th to 19th
order_sub = order[idx0:idx1]
inv_sub = 1.0 / zeros_y[idx0:idx1]   # = 4*kappa/F

# Linear fit in the inset
coef = np.polyfit(order_sub, inv_sub, 1)
order_sub_fit = np.linspace(order_sub.min(), order_sub.max(), 200)
inv_fit = np.polyval(coef, order_sub_fit)

axins = ax.inset_axes([0.64, 0.42, 0.30, 0.34])
axins.scatter(order_sub, inv_sub, s=28, color="red", zorder=3)
axins.plot(order_sub_fit, inv_fit, color="#1f4ea8", lw=1.6, ls=(0, (1.2, 1.2)))
axins.set_xlabel("Order", fontsize=12, labelpad=1)
axins.set_ylabel(r"$4\kappa/F$", fontsize=12, labelpad=2)
axins.tick_params(axis='both', labelsize=10, top=True, right=True, length=2.5)
# Clean inset ticks
axins.set_xticks([])
axins.set_yticks([])

# Adjust spine widths
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
for spine in axins.spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()