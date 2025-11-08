import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[1]],A[:,[1]]**2))
T = A[:,[0]]
C = A[:,[1]]

#Least squares method
t, at, bt = np.linalg.lstsq(X, T, rcond=None)[0]
n_lsq = np.zeros((3,1))
n_lsq[0][0] = t
n_lsq[1][0] = at
n_lsq[2][0] = bt
print(n_lsq)

plt.plot(T, X@n_lsq)
plt.plot(T, C, 'k.')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')
plt.tight_layout()
plt.savefig('../figs/train.png')

#Close current figure(s)
plt.close('all')

#Plot for validation
B = np.loadtxt('validation_data.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2))
Cv = B[:,[1]]
Tv = B[:,[0]]
plt.plot(Tv, Xv@n_lsq)
plt.plot(Tv, Cv, 'k.')
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/valid.png')
