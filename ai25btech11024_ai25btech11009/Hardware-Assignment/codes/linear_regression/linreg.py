import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

V=np.array([2.3509,2.3265,2.3216,2.3069,2.2987,2.2923,2.2874,2.2651,2.2581,2.2483,2.2532,2.2434,2.2397,2.2385,2.1945,2.1310,2.1065]).reshape(-1,1)

A=np.array([41.1,45.5,58.2,51.1,54.9,58.5,60.8,64.3,66.9,70.0,72.1,75.9,77.3,80.5,83.2,86.1,88.8]).reshape(-1,1)
T=np.column_stack((np.ones(17),A,A**2))

P=T.T @ T
Q=la.inv(P)
R=T.T @ V
X=Q @ R
print(X)

K=np.column_stack((np.ones(17),V,V**2))

D=K.T @ K
E=la.inv(D)
F=K.T @ A
Y=E @ F
print(Y)

t=np.linspace(35,100,500)
v=X[0]+X[1]*t+X[2]*t*t

V1=np.array([2.3509,2.3265,2.3216,2.3069,2.2987,2.2923,2.2874,2.2651,2.2581,2.2483,2.2532,2.2434,2.2397,2.2385,2.1945,2.1310,2.1065])
T1=np.array([41.1,45.5,58.2,51.1,54.9,58.5,60.8,64.3,66.9,70.0,72.1,75.9,77.3,80.5,83.2,86.1,88.8])
coords=np.vstack((T1,V1))

plt.plot(t,v)
plt.scatter(coords[0,:],coords[1,:],c='r',label='Data points')
plt.xlabel('Temperature (C)', fontsize=16, fontweight='bold')
plt.ylabel('Output Voltage (V)', fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend()
plt.show()
