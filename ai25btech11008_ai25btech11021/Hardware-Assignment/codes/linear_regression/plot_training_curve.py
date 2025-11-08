#Code by Harshith Chiruvella and Abhiram Reddy

import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('training.txt')

plt.scatter(A[:,0], A[:,1])

plt.grid()
plt.ylabel('Output Voltage (V)')
plt.xlabel('Temperature ($^{\\circ}$C)')
plt.tight_layout()

plt.savefig('../figs/training.png')