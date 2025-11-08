import numpy as np
import matplotlib.pyplot as plt
import padasip as pa

A = np.loadtxt('/home/shriyasnh/Desktop/Hardware_Assignment/codes/linear_regression/training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
T = A[:,[0]]
C = A[:,[1]]

#Least squares method
n_lsq = np.linalg.lstsq(X, C, rcond=None)[0].reshape((3,1))
print(n_lsq)

#Plot both the results
plt.plot(T, X@n_lsq)
plt.plot(T, C, 'k.')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel(r'Temperature ($^{\circ}$C)')
plt.tight_layout()
plt.savefig('/home/shriyasnh/Desktop/Hardware_Assignment/figs/train.png')


#Plot for validation
B = np.loadtxt('/home/shriyasnh/Desktop/Hardware_Assignment/codes/linear_regression/validating_data.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2))
Cv = B[:,[1]]
Tv = B[:,[0]]
plt.figure()
plt.plot(Tv, Xv@n_lsq)
plt.plot(Tv, Cv, 'k.')
plt.ylabel('Output Voltage (V)')
plt.xlabel(r'Temperature ($^{\circ}$C)')
plt.grid()
plt.tight_layout()
plt.savefig('/home/shriyasnh/Desktop/Hardware_Assignment/figs/valid.png')

# For finding the function for T in terms of V
A2 = np.loadtxt('/home/shriyasnh/Desktop/Hardware_Assignment/codes/linear_regression/training_data.txt')
X2 = np.hstack((np.ones((A2.shape[0],1)), A2[:,[1]], A2[:,[1]]**2))
V = A[:,[1]]
C2 = A[:,[0]]

# Least squares Method
n2_lsq = np.linalg.lstsq(X2, C2, rcond=None)[0].reshape((3,1))
print(n2_lsq)