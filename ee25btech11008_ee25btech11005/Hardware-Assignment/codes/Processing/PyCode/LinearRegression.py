import numpy as np
import matplotlib.pyplot as plt
import padasip as pa

A = np.loadtxt('../Data/training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
T = A[:,[0]]
C = A[:,[1]]

#Least squares method
v, av, bv = np.linalg.lstsq(X, C, rcond=None)[0]
n_lsq = np.zeros((3,1))
v = np.array(v).item()     
av = np.array(av).item()
bv = np.array(bv).item()

n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv

print(n_lsq)

#Plot both the results
idx = np.argsort(T.flatten())
plt.plot(T[idx], (X @ n_lsq)[idx])
plt.plot(T[idx], C[idx], 'k.')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature (\u00B0C)')
plt.tight_layout()
plt.savefig('../../../figs/train.png')

#Close current figure(s)
plt.close('all')

#Plot for validation
B = np.loadtxt('../Data/validation_data.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2))
Cv = B[:,[1]]
Tv = B[:,[0]]
plt.clf()
idxv = np.argsort(Tv.flatten())
plt.plot(Tv[idxv], (Xv @ n_lsq)[idxv])
plt.plot(Tv[idxv], Cv[idxv], 'k.')
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature (\u00B0C)')
plt.grid()
plt.tight_layout()
plt.savefig('../../../figs/valid.png')
