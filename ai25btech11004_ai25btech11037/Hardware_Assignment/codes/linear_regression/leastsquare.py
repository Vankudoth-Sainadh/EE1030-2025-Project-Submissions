import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2,A[:,[0]]**3))
T = A[:,[0]]
C = A[:,[1]]

#Least squares method
v, av, bv, cv = np.linalg.lstsq(X, C, rcond=None)[0]
n_lsq = np.zeros((4,1))
n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv
n_lsq[3][0] = cv
print(n_lsq)

#Plot both the results
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
B = np.loadtxt('validation.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2,B[:,[0]]**3))
Cv = B[:,[1]]
Tv = B[:,[0]]
plt.plot(Tv, Xv@n_lsq)
plt.plot(Tv, Cv, 'k.')
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/valid.png')

#for finding the function for T in terms of V

A2 = np.loadtxt('training.txt')
X2 = np.hstack((np.ones((A2.shape[0],1)),A2[:,[1]],A2[:,[1]]**2,A2[:,[1]]**3))
V = A[:,[1]]
C2 = A[:,[0]]

#Least squares method
t, at, bt, ct = np.linalg.lstsq(X2, C2, rcond=None)[0]
n2_lsq = np.zeros((4,1))
n2_lsq[0][0] = t
n2_lsq[1][0] = at
n2_lsq[2][0] = bt
n2_lsq[3][0] = ct
print(n2_lsq)
