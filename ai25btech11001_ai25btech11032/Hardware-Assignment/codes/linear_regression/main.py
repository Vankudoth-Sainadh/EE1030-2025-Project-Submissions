import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import openpyxl

df = openpyxl.load_workbook("../../tables/data.xlsx")
df1 = df.active

data = [[float(j.value) for j in i[1:20]] for i in df1.iter_cols(5,6)]

temp = np.asarray(data[1])
volt = np.asarray(data[0])
volt = volt*5.0/1023.0

X= np.block([[np.ones(temp.shape)],[temp],[temp*temp]]).T

Y = volt.reshape(-1,1)

N = LA.inv(X.T@(X))@(X.T@Y)
print(N)
plt.scatter(temp,volt)

temp = np.linspace(20.0,90.0,1000)

X= np.block([[np.ones(temp.shape)],[temp],[temp*temp]]).T

Y = X@N

plt.plot(temp,Y)

plt.xlabel("T")
plt.ylabel("V")
plt.savefig("../../figs/VvsT.jpg")


