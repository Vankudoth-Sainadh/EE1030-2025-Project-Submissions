import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("trainingdata.txt")
v = data[:, 0]
t = data[:, 1]

# Design matrix for voltage as a function of temperature
t_mat = np.column_stack((np.ones(len(t)), t, t**2))
coef_v = np.linalg.lstsq(t_mat, v, rcond=0)[0]

# Design matrix for temperature as a function of voltage
v_mat = np.column_stack((np.ones(len(v)), v, v**2))
coef_t = np.linalg.lstsq(v_mat, t, rcond=0)[0]
print(coef_t)

# Generate model
t_range = np.linspace(25, 100, 10000)
t_pred = np.column_stack((np.ones(len(t_range)), t_range, t_range**2))
v_pred = t_pred @ coef_v

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(t, v, marker='o', label="Data")
ax.plot(t_range, v_pred, label="Model")
ax.grid(True)
ax.set_xlabel(f"Temperature(\u00B0C)") 
ax.set_ylabel("Voltage (V)")
ax.set_title("Training Data Plot")
ax.legend()
plt.savefig("../../figs/training.png")
plt.show()









