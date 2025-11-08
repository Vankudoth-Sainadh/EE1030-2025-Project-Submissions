import numpy as np
import numpy.linalg as LA
import sys  # for command line input
from pathlib import Path
import ctypes as ct


data = {}
with open(Path(__file__).resolve().parent.parent.parent / "tables" / "training_data.txt") as fp:
    for line in fp:
        data[line.split()[0]] = line.split()[1]
try:
    n = int(sys.argv[1])  # first argument, degree of relation
except IndexError:
    n = 2

length = len(data)

T = np.empty((length, 1))
V = np.empty((length, n + 1))

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

libmatrix = ct.CDLL("./matrix.so")
find_inv = libmatrix.find_inverse

find_inv.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ct.c_int,
]

find_inv.restype = ct.c_int
matmul = libmatrix.mul

matmul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ct.c_int,
    ct.c_int,
    ct.c_int,
]
libmatrix.mul.restype = None

vt_v = np.zeros((n+1)*(n+1))

matmul(V.T.flatten(),V.flatten(),vt_v,(n+1),length,(n+1))

inv = np.empty_like(vt_v)
find_inv(vt_v,inv,(n+1))

X = np.zeros((n+1) * length)
matmul(inv , V.T.flatten() , X , (n+1) , (n+1) , length)
prod = np.zeros((n+1) * 1)
matmul(X, T.flatten() , prod , (n+1) , length, 1)
print(prod)



