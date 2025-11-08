import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training_data.txt')

T_out = A[:,[0]]
V_in = A[:,[1]]   

X_design = np.hstack((np.ones((A.shape[0],1)), V_in, V_in**2))


coeffs = np.linalg.lstsq(X_design, T_out, rcond=None)[0]

print("Coefficients for T = a + b*V + c*V^2:")
print(coeffs)


plt.plot(V_in, X_design @ coeffs, label='Best Fit Curve')
plt.plot(V_in, T_out, 'k.', label='Data Points')
plt.grid()
plt.ylabel(r'Temperature ($^{\circ}$C)')
plt.xlabel('Output Voltage (V)')
plt.legend()
plt.tight_layout()
plt.savefig("/media/indhiresh-s/New Volume/Matrix2/EE1030-2025-Project-Submissions/ee25btech11027_ee25btech11038/Hardware_Assignment/figs/training.png")
plt.close('all')

B = np.loadtxt('validation_data.txt')

Tv_out = B[:,[0]]
Vv_in = B[:,[1]]

Xv_design = np.hstack((np.ones((B.shape[0],1)), Vv_in, Vv_in**2))

plt.plot(Vv_in, Xv_design @ coeffs, label='Best Fit Curve')
plt.plot(Vv_in, Tv_out, 'k.')
plt.grid()
plt.ylabel(r'Temperature ($^{\circ}$C)')
plt.xlabel('Output Voltage (V)')
plt.legend()
plt.tight_layout()
plt.savefig("/media/indhiresh-s/New Volume/Matrix2/EE1030-2025-Project-Submissions/ee25btech11027_ee25btech11038/Hardware_Assignment/figs/validation.png")
print("\nSuccessfully generated 'train_voltage_input.png' and 'valid_voltage_input.png'.")
