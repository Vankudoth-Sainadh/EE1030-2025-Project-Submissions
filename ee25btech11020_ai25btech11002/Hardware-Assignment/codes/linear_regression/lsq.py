import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

A = np.loadtxt('trainingdata.dat')
Tt = A[:, 1]
Vt = A[:, 0]

Xt = np.column_stack((np.ones(Tt.shape[0]), Tt, Tt**2))
a, residuals, rank, s = LA.lstsq(Xt, Vt, rcond=None)
print("Coefficients [a0, a1, a2] for V(T):", a)

plt.figure()
plt.plot(Tt, Xt @ a, label='Fitted Model')
plt.scatter(Tt, Vt, label='Training Data', color='#A0A')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel(r'Temperature ($^{\circ}$C)')
plt.legend()
plt.tight_layout()
plt.savefig('../../figs/model.png')

B = np.loadtxt('validationdata.dat')
Tv = B[:, 1]
Vv = B[:, 0]

plt.figure()
plt.plot(Tt, Xt @ a, label='Fitted Training Model', color='#0AA')
plt.scatter(Tv, Vv, label='Validation Data', color="#660")
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel(r'Temperature ($^{\circ}$C)')
plt.legend()
plt.tight_layout()
plt.savefig('../../figs/validation.png')

# --- Inversion using quadratic formula ---
a0, a1, a2 = a
disc = a1**2 - 4*a2*(a0 - Vt)
T1 = (-a1 + np.sqrt(disc)) / (2*a2)
T2 = (-a1 - np.sqrt(disc)) / (2*a2)

# Choose the branch that matches measured T range
T_calc = np.where(abs(T1 - Tt) < abs(T2 - Tt), T1, T2)

plt.figure()
plt.scatter(Tt, Vt, label='Measured Data', color='k')
plt.scatter(T_calc, Vt, label='Inverted (Quadratic Formula)', color='r', s=15)
plt.xlabel("Temperature (°C)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('../../figs/quadratic_inversion.png')

# Compare predicted T(V) numerically
plt.figure()
plt.plot(Tt, T_calc, 'o', label='Predicted T(V)')
plt.plot([min(Tt), max(Tt)], [min(Tt), max(Tt)], 'k--', label='Ideal (y=x)')
plt.xlabel("Measured Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('../../figs/T_comparison.png')

