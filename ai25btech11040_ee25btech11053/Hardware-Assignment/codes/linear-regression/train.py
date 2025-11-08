import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('./training.csv', delimiter=',')
A = A[A[:,1].argsort()]
X = np.hstack((np.ones((A.shape[0],1)),A[:,[1]],A[:,[1]]**2))
T = A[:,[0]]
C = A[:,[1]]

#Least squares method
a, b, c = map(lambda x: x[0], np.linalg.lstsq(X, T, rcond=None)[0])
lsq = np.array([
    [a],
    [b],
    [c]
])
print(f"Coefficients: {a}, {b}, {c}")

min_v = np.min(C)
max_v = np.max(C)

rng = np.arange(min_v, max_v, step=0.01).reshape(-1, 1)

#Plot both the results
plt.plot(C, X@lsq, 'bx', label='Fitting')
plt.plot(C, T, 'k.', label='Data')
plt.plot(rng, np.hstack((np.ones((rng.shape[0],1)), rng, rng**2))@lsq, label='Model')
plt.legend()
plt.grid()
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature ($^{\\circ}$C)')
plt.savefig('../../figs/training.png')

#Close current figure(s)
plt.close('all')

# Plot for validation
B = np.loadtxt('validation.csv', delimiter=',')
B = B[B[:,1].argsort()]
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[1]],B[:,[1]]**2))
Cv = B[:,[1]]
Tv = B[:,[0]]
plt.plot(Cv, Xv@lsq, 'rx', label="predictions")
plt.plot(Cv, Tv, 'k.', label="actual data")
plt.xlabel('Output Voltage (V)')
plt.ylabel('Temperature ($^{\\circ}$C)')
plt.legend()
plt.grid()
plt.savefig('../../figs/validation.png')