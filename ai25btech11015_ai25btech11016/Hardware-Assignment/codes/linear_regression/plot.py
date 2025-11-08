import numpy as np
import matplotlib.pyplot as plt

Vw= np.array([0.19,0.225,0.26,0.293,0.3177,0.3372,0.3519,0.3715,0.322,0.3566,0.367,0.386,0.3959,0.4015,0.4106])
Tw = np.array([31.3,42.1,51.9,59.5,66.7,73.3,78,83.6,69.3,78.7,83.1,88.2,91.3,93.8,97])

a = 156.712  
b = 197.683 
c = -11.134 

V_fit = np.linspace(Vw.min(), Vw.max(), 200)
T_fit = a * V_fit**2 + b * V_fit + c

plt.figure(figsize=(8,6))
plt.scatter(Vw, Tw, color='b', label='Measured Data', marker='o')


plt.plot(V_fit, T_fit, color='r', label='Quadratic Fit: T = aV² + bV + c')
plt.title("Voltage vs Temperature with Quadratic Fit")
plt.xlabel("Voltage (V)")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
plt.savefig("volt_vs_temp_with_fit.png", dpi=300)
plt.show()
