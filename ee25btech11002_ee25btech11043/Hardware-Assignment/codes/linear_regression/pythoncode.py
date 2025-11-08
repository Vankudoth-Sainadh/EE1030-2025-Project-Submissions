import numpy as np
import matplotlib.pyplot as plt


A = np.loadtxt('trainingdata.txt')
T_train = A[:,0]
Y_train = A[:,1]


X_train = np.column_stack((np.ones(A.shape[0]), T_train, T_train**2))


theta = np.linalg.lstsq(X_train, Y_train, rcond=None)[0].reshape(-1, 1)
print("The value of n is :")
print(theta)

plt.figure()
plt.plot(T_train, X_train @ theta, label='Fitted Model')
plt.plot(T_train, Y_train, 'k.', label='Training Data')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel(r'Temperature ($^{\circ}$C)')
plt.legend()
plt.tight_layout()
plt.savefig('train.png')
plt.close()

B = np.loadtxt('validationdata.txt')
T_valid = B[:, 0]
Y_valid = B[:, 1]

X_valid = np.column_stack((np.ones(B.shape[0]), T_valid, T_valid**2))

plt.figure()
plt.plot(T_valid, X_valid @ theta, label='Fitted Model')
plt.plot(T_valid, Y_valid, 'k.', label='Validation Data')
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel(r'Temperature ($^{\circ}$C)')
plt.legend()
plt.tight_layout()

plt.savefig('valid.png')


