import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt("trainingdata.txt")
v = train[:, 0]
t = train[:, 1]

# Fit voltage as a quadratic function of temperature
t_mat = np.column_stack((np.ones(len(t)), t, t**2))
coef_v = np.linalg.lstsq(t_mat, v, rcond=0)[0]

# Fit temperature as a quadratic function of voltage
v_mat = np.column_stack((np.ones(len(v)), v, v**2))
coef_t = np.linalg.lstsq(v_mat, t, rcond=0)[0]
print(coef_t)

# Predict voltage for a range of temperatures
t_range = np.linspace(25, 100, 10000)
t_pred = np.column_stack((np.ones(len(t_range)), t_range, t_range**2))
v_pred = t_pred @ coef_v
# Load validation data
val = np.loadtxt("Validation.txt")
v_val = val[:, 0]
t_val = val[:, 1]
print(t_val)
v_exp=np.column_stack((np.ones(len(v_val)), v_val, v_val**2))
t_exp=v_exp@coef_t
print(t_exp)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(t_val, v_val, marker='o', label="Data")
ax.plot(t_range, v_pred, label="Model")
ax.grid(True)
ax.set_xlabel(f"Temperature(\u00B0C)")
ax.set_ylabel("Voltage (V)")
ax.set_title("Validation Data Plot")
ax.legend()
plt.savefig("../../figs/validation.png")
plt.show()











