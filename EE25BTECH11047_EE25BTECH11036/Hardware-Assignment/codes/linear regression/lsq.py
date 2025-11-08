import numpy as np
import matplotlib.pyplot as plt


A = np.loadtxt('training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)), A[:,[0]], A[:,[0]]**2, A[:,[0]]**3))
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
plt.xlabel('Temperature ($^{\\circ}$C)')
plt.tight_layout()
plt.savefig('../figs/train.png')

#Close current figure(s)
plt.close('all')

#Plot for validation
B = np.loadtxt('validation_data.txt')
Xv = np.hstack((np.ones((B.shape[0],1)), B[:,[0]], B[:,[0]]**2, B[:,[0]]**3))
Cv = B[:,[1]]
Tv = B[:,[0]]
plt.plot(Tv, Xv@n_lsq)
plt.plot(Tv, Cv, 'k.')
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\circ}$C)')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/valid.png')

A2 = np.loadtxt('training_data.txt')

T = A2[:, 0].astype(float)   # shape (N,)
V = A2[:, 1].astype(float)   # shape (N,)

X2 = np.vstack((np.ones_like(V), V, V**2, V**3)).T  # shape (N,4)

sol_asc, *_ = np.linalg.lstsq(X2, T, rcond=None)    # shape (4,)
coeffs_array = sol_asc.reshape((4,1))               # column vector like your earlier n_lsq

p_desc = np.polyfit(V, T, 3)  # [b3, b2, b1, b0]
p_from_polyfit_asc = np.array([p_desc[3], p_desc[2], p_desc[1], p_desc[0]])

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

pred_lstsq = X2 @ sol_asc
pred_polyfit = np.polyval(p_desc, V)
print("Coefficients of temperature polynomial")
print(coeffs_array)

