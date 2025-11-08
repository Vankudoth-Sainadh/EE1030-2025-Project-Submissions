import numpy as np
import matplotlib.pyplot as plt
import padasip as pa

A = np.loadtxt('training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
T = A[:,[0]]
C = A[:,[1]]

#Least squares method
v, av, bv = np.linalg.lstsq(X, C, rcond=None)[0]
n_lsq = np.zeros((3,1))
n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv
print(n_lsq)

#Plot both the results
plt.plot(T, X@n_lsq)
plt.plot(T, C, 'k.')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')
plt.tight_layout()
plt.savefig('train.png')

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
plt.savefig('valid.png')

# For finding the function for T in terms of V
A2 = np.loadtxt('training_data.txt')
X2 = np.hstack((np.ones((A2.shape[0],1)), A2[:,[1]], A2[:,[1]]**2))
V = A[:,[1]]
C2 = A[:,[0]]

# Least squares Method
t,at,bt = np.linalg.lstsq(X2,C2, rcond=None)[0]
n2_lsq = np.zeros((3,1))
n2_lsq[0][0] = t
n2_lsq[1][0] = at
n2_lsq[2][0] = bt
print(n2_lsq)
