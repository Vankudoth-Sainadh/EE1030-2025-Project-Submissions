#Code by Harshith Chiruvella and Abhiram Reddy

import numpy as np
import matplotlib.pyplot as plt

# The coefficients (n_lsq) are needed for this plot, so the calculation
# block for n_lsq should ideally be included or imported here, but based
# strictly on your original structure:

# --- RECALCULATE COEFFICIENTS (needed for n_lsq) ---
A = np.loadtxt('training.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
v, av, bv = np.linalg.lstsq(X, A[:,[1]], rcond=None)[0]
n_lsq = np.zeros((3,1))
n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv
# ----------------------------------------------------

#Plot for validation
B = np.loadtxt('validation.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2))
Vv = B[:,[1]]
Tv = B[:,[0]]

# Plotting the model prediction (Xv@n_lsq) vs. actual validation data (Cv)
plt.figure()
plt.plot(Tv, Xv@n_lsq)
plt.plot(Tv, Vv, 'k.')
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\\circ}$C)')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/validation.png')