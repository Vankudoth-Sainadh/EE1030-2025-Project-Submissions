import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('../../tables/training_data.txt')

X = np.hstack((np.ones((A.shape[0], 1)), A[:, [0]], A[:, [0]]**2))
T = A[:, [0]]
Y = A[:, 1]

XTX = X.T @ X

inv = np.linalg.inv(XTX)

XTY = X.T @ Y

C, B, A_coeff = inv @ XTY

theta = np.array([[C], [B], [A_coeff]])

T_fit = np.linspace(T.min(), T.max(), 400)
X_fit = np.hstack((np.ones((T_fit.shape[0], 1)), T_fit.reshape(-1, 1), T_fit.reshape(-1, 1)**2))
Y_fit = X_fit @ theta

plt.figure(figsize=(10, 6))

plt.plot(T_fit, Y_fit, 'r-', label='Least Squares Parabola') 
plt.plot(T, Y, 'bo', label='Training Data Points') 

plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')

plt.title('Training Data vs. Least Squares Parabola') 
plt.legend()
plt.tight_layout()

plt.savefig('../../figs/training_data.png') 
plt.close('all')

B = np.loadtxt('../../tables/validation_data.txt')

Xv = np.hstack((np.ones((B.shape[0], 1)), B[:, [0]], B[:, [0]]**2))
Yv = B[:, 1]
Tv = B[:, [0]]

Tv_fit = np.linspace(Tv.min(), Tv.max(), 400)
Xv_fit = np.hstack((np.ones((Tv_fit.shape[0], 1)), Tv_fit.reshape(-1, 1), Tv_fit.reshape(-1, 1)**2))
Yv_fit = Xv_fit @ theta

plt.figure(figsize=(10, 6))
plt.plot(Tv_fit, Yv_fit, 'r--', label='Least Squares Parabola') 
plt.plot(Tv, Yv, 'ms', label='Validation Data Points') 

plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')

plt.title('Validation Data vs. Least Squares Parabola') 
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('../../figs/validation_data.png')
