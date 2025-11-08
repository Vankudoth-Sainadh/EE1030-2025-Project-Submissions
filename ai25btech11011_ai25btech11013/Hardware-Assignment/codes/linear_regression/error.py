import numpy as np 
A=np.loadtxt("Validation.txt")
V=A[:,0]
Tact=A[:,1]
V1=np.zeros((len(V),3))
V1[:,0]=np.ones(len(V))
V1[:,1]=V
V1[:,2]=V*V
X=np.array([2561.69352124,-1579.0764281,243.53729495])
Texp=V1@X
print(Texp)
print(Tact)
mae=np.mean(np.abs(Texp-Tact))
rmse = np.sqrt(np.mean((Texp - Tact)**2))
maxae = np.max(np.abs(Texp - Tact))
print("Root mean square error:", rmse)
print("Maximum Absolute error:", maxae)
print("Mean Absolute error",mae)
