import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x_train, y_train = np.loadtxt('../../tables/training_data.txt', unpack=True)

x_val, y_val = np.loadtxt('../../tables/validation_data.txt', unpack=True)

A = np.vstack([np.ones_like(x_train), x_train, x_train**2, x_train**3]).T
b = y_train.reshape(-1, 1)

ATA = A.T @ A
ATb = A.T @ b

ATA_sympy = sp.Matrix(ATA)
ATA_inv_sympy = ATA_sympy.inv()
ATA_inv = np.array(ATA_inv_sympy)

coeffs = (ATA_inv @ ATb).flatten()

print("Estimated cubic coefficients (from training data):")
for i, c in enumerate(coeffs):
    print(f"  c{i} = {c:.6f}")

xx = np.linspace(24, 96, 200)
yy_fit = coeffs[0] + coeffs[1]*xx + coeffs[2]*xx**2 + coeffs[3]*xx**3

plt.figure(1, figsize=(8, 6)) 
plt.scatter(x_train, y_train, color='blue', label="Training Data Points")
plt.plot(xx, yy_fit, color='red', label="Least squares cubic fit")
plt.xlabel("Temperature(C)")
plt.ylabel("Voltage(V)")
plt.title("Least Squares Fit on Training Data")
plt.legend()
plt.grid(True)
plt.savefig("../../figs/Volt_Temp/training.png")

plt.figure(2, figsize=(8, 6)) 
plt.scatter(x_val, y_val, color='green', label="Validation Data Points")
plt.plot(xx, yy_fit, color='red', label="Least square prediction")
plt.xlabel("Temperature(C)")
plt.ylabel("Voltage(V)")
plt.title("Validation Data")
plt.legend()
plt.grid(True)
plt.savefig("../../figs/Volt_Temp/validation.png")
plt.show()