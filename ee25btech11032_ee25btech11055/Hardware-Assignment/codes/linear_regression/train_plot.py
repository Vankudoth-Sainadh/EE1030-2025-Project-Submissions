import numpy as np
import numpy.linalg as LA
import sys  # for command line input
from pathlib import Path
import matplotlib.pyplot as plt

data = {}
with open(Path(__file__).resolve().parent.parent.parent / "tables" / "training_data.txt") as fp:
    for line in fp:
        data[line.split()[0]] = line.split()[1]
try:
    n = int(sys.argv[1])  # first argument, degree of relation
except IndexError:
    n = 2

T = np.empty((len(data), 1))
V = np.empty((len(data), n + 1))

i = 0
for temp, volt in data.items():
    T[i] = float(temp)
    volts = []
    volt = float(volt)
    for j in range(n + 1):
        if j == 0:
            volts.append(1)
        else:
            volts.append(volts[-1] * volt)
    V[i] = volts
    i += 1

# Now using least squares
lst_square = LA.inv(V.T @ V) @ V.T @ T
print(lst_square)

#Least squares method
print(np.linalg.lstsq(V, T, rcond=None)[0])

# np.linalg.lstsq returns Tuple (coeffs, residuals, rank, singular_values)
# np.linalg.lstsq(X, C, rcond=None)[0] , unpacks coeffs into v , av,bv

#Using SVD
U, s, VT = np.linalg.svd(V, full_matrices=False)
n_svd = VT.T @ np.linalg.inv(np.diag(s)) @ U.T @ T
print(n_svd)

#Plot both the results
plt.plot(V[:,[1]] , V@lst_square , color="red" , label = "Predicted Temperature")
plt.plot(V[:,[1]] , T , 'k.',color="green" , label = "Actual Temperature" )
plt.grid()
plt.xlim(2.2,2.8)
plt.ylim(30,90)
plt.ylabel('Output Temperature ($^{\circ}$C)')
plt.xlabel('Voltage Reading')
plt.tight_layout()
plt.legend(loc='best')
plt.savefig(Path(__file__).resolve().parent.parent.parent / "figs" / "train.png")


#Close current figure(s)
plt.close('all')


#Plot for validation
valid_data = {}
with open(Path(__file__).resolve().parent.parent.parent / "tables" / "validation_data.txt") as fp:
    for line in fp:
        valid_data[line.split()[1]] = [line.split()[2],line.split()[0]]

T_thermo = np.empty((len(valid_data), 1))
Vv = np.empty((len(valid_data), n + 1))

T_pt =  np.empty((len(valid_data), 1))
i = 0
for temp, volt in valid_data.items():
    T_thermo[i] = float(temp)
    # print(volt)
    T_pt[i] = float(volt[1])
    volts = []
    volt = float(volt[0])
    for j in range(n + 1):
        if j == 0:
            volts.append(1)
        else:
            volts.append(volts[-1] * volt)
    Vv[i] = volts
    i += 1    

plt.plot( Vv[:,[1]] , T_pt , label = "Predicted Model")
plt.plot( Vv[:,[1]], T_thermo , 'k.' , label = "Actual Temp" )
plt.xlabel(' Output Voltage (V) ');plt.ylabel('Temperature ($^{\circ}$C)')
plt.grid()
# plt.tight_layout()
plt.legend(loc = 'best')
plt.savefig(Path(__file__).resolve().parent.parent.parent / "figs" / "valid.png")


