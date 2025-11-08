import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training_data.txt')
T = A[:, [0]]
C = A[:, [1]]

X = np.hstack((np.ones((A.shape[0], 1)), T, T**2, T**3))

v, av, bv, cv = np.linalg.lstsq(X, C, rcond=None)[0]

n_lsq = np.zeros((4, 1))
n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv
n_lsq[3][0] = cv
print(n_lsq)

plt.plot(T, X @ n_lsq, label='Cubic Fit')
plt.plot(T, C, 'k.', label='Training Data')
plt.grid()

plt.ylabel('Temperature ($^{\circ}$C)')
plt.xlabel('Output Voltage (V)')
plt.legend()
plt.tight_layout()
plt.savefig('/home/nipun-dasari/EE1030-2025-Project-Submissions/ee25btech11042_ee25btech11048/Hardware_Assignment/figs/train_cubic.png')

plt.close('all')

B = np.loadtxt('validation_data.txt')
Tv = B[:, [0]]
Cv = B[:, [1]]

Xv = np.hstack((np.ones((B.shape[0], 1)), Tv, Tv**2, Tv**3))

plt.plot(Tv, Xv @ n_lsq, label='Cubic Fit')
plt.plot(Tv, Cv, 'k.', label='Validation Data')
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('/home/nipun-dasari/EE1030-2025-Project-Submissions/ee25btech11042_ee25btech11048/Hardware_Assignment/figs/valid_cubic.png')

