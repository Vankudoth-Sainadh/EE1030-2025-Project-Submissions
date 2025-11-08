import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training_data1.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
T = A[:,[0]]
Y = A[:,[1]]

#Least squares method
v, av, bv = np.linalg.lstsq(X, Y, rcond=None)[0]
theta = np.zeros((3,1))
theta[0][0] = v
theta[1][0] = av
theta[2][0] = bv
print(theta)

#Plot both the results
plt.plot(T, X@theta)
plt.plot(T, Y, 'k.')
plt.grid()
plt.ylabel('Temperature ($^{\circ}$C)')
plt.xlabel('Output Voltage(V)')
plt.tight_layout()
plt.savefig('../../figs/train1.png')

#Close current figure(s)
plt.close('all')

#Plot for validation
B = np.loadtxt('validation_data1.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2))
Yv = B[:,[1]]
Tv = B[:,[0]]
plt.plot(Tv, Xv@theta)
plt.plot(Tv, Yv, 'k.')
plt.ylabel('Temperature ($^{\circ}$C)')
plt.xlabel('Output Voltage (V)')
plt.grid()
plt.tight_layout()
plt.savefig('../../figs/valid1.png')
