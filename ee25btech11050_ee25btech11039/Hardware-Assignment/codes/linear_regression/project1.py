import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
T = A[:,[0]]
C = A[:,[1]]

# ------------------------------------------------------------
# Least squares method (V in terms of T)
# ------------------------------------------------------------
v, av, bv = np.linalg.lstsq(X, C, rcond=None)[0]
n_lsq = np.zeros((3,1))
n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv
print("\nCoefficients of V in terms of T:")
print(n_lsq)

# ------------------------------------------------------------
# Function to compute T in terms of V (quadratic least squares)
# ------------------------------------------------------------
def fit_T_in_terms_of_V(V, T):
    """
    Fit T = c0 + c1*V + c2*V^2 using least squares method.
    Returns coefficients (c0, c1, c2) and the design matrix.
    """
    X_TV = np.hstack((np.ones((V.shape[0], 1)), V, V**2))
    coeffs = np.linalg.lstsq(X_TV, T, rcond=None)[0]

    # Convert from 1x1 numpy arrays to float
    c0, c1, c2 = coeffs.flatten().astype(float)
    return c0, c1, c2, X_TV


# ------------------------------------------------------------
# Fit T(V) and print coefficients
# ------------------------------------------------------------
c0, c1, c2, X_TV = fit_T_in_terms_of_V(C, T)
print("\nCoefficients of T in terms of V:")
print(f"c0 = {c0:.6f},  c1 = {c1:.6f},  c2 = {c2:.6e}")
print(f"\nEquation of T in terms of V:\nT = ({c0:.6f}) + ({c1:.6f})*V + ({c2:.6e})*V^2")

# ------------------------------------------------------------
# Plot both the results (V vs T)
# ------------------------------------------------------------
plt.plot(T, X @ n_lsq, label="Fitted V(T)")
plt.plot(T, C, 'k.', label="Training Data")
plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\\circ}$C)')
plt.legend()
plt.tight_layout()
plt.savefig('../figs/train.png')

#Close current figure(s)
plt.close('all')

# ------------------------------------------------------------
# Plot for validation (original)
# ------------------------------------------------------------
B = np.loadtxt('validation_data.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2))
Cv = B[:,[1]]
Tv = B[:,[0]]
plt.plot(Tv, Xv @ n_lsq, label="Fitted V(T)")
plt.plot(Tv, Cv, 'k.', label="Validation Data")
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\\circ}$C)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('../figs/valid.png')

#Close current figure(s)
plt.close('all')

# ------------------------------------------------------------
# Plot for T(V) (new)
# ------------------------------------------------------------
V_range = np.linspace(np.min(C), np.max(C), 200).reshape(-1,1)
T_fit = c0 + c1*V_range + c2*(V_range**2)

plt.plot(C, T, 'k.', label="Training Data (T vs V)")
plt.plot(V_range, T_fit, 'r-', label="Fitted T(V)")
plt.xlabel('Voltage (V)')
plt.ylabel('Temperature ($^{\\circ}$C)')
plt.title('T in terms of V (Least Squares Fit)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('../figs/T_vs_V.png')
plt.show()



