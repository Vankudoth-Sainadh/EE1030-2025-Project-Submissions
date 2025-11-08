import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import openpyxl

df = openpyxl.load_workbook("../../tables/data.xlsx")
df1 = df.active

data = [[float(j.value) for j in i[2:20]] for i in df1.iter_cols(1,2)]

temp = np.asarray(data[0])
volt = np.asarray(data[1])
#volt = volt*5.0/1023.0
N = np.array([[794.88620258 ],[-384.63936743],[26.03817558]])

X= np.block([[np.ones(volt.shape)],[volt],[volt*volt]]).T
Y = X@N

for i in range(Y[:,0].shape[0]):
    print(f"{volt[i]}\t{temp[i]}\t{Y[i,0]}")

err = np.subtract(temp,Y[:,0])
print("Max error :",np.amax(err))
print("Mean absolute :",np.mean(np.abs(err)))
print("RMSE :",np.sqrt(np.mean(err*err)))
plt.scatter(volt,temp)

volt = np.linspace(430,500,1000)
volt = volt*5.0/1023.0

X= np.block([[np.ones(volt.shape)],[volt],[volt*volt]]).T

Y = X@N
plt.plot(volt,Y)

plt.ylabel("T")
plt.xlabel("V")
plt.title("Testing")
plt.savefig("../../figs/Testing.jpg")



