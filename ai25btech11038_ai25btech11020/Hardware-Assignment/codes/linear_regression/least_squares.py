import numpy as np

temperatures = []
voltages = []

print("Enter 12 temperatures (T):")
for i in range(12):
    temperatures.append(float(input()))

print("Enter 12 corresponding voltages (V):")
for i in range(12):
    voltages.append(float(input()))

A = np.zeros((12, 3))
for i in range(12):
    A[i][0] = 1
    A[i][1] = voltages[i]
    A[i][2] = voltages[i]**2

b = np.array(temperatures)

p = A.T
q = p @ A
r = np.linalg.inv(q)
x = r @ p @ b

print("\nConstants:")
print("(for T = a0 + a1 * V + a2 * V^2)")
print(f"a0 = {x[0]}")
print(f"a1 = {x[1]}")
print(f"a2 = {x[2]}")
