import numpy as np
import math

Vw = np.array([0.19,0.225,0.26,0.293,0.3177,0.3372,0.3519,0.3715,0.322,0.3566,0.367,0.386,0.3959,0.4015,0.4106])
Tw = np.array([31.3,42.1,51.9,59.5,66.7,73.3,78,83.6,69.3,78.7,83.1,88.2,91.3,93.8,97])
Xw = np.column_stack((Vw**2, Vw, np.ones_like(Vw)))
# (X^T X)^(-1) X^T T    --> least sq formual
a,b,c = np.linalg.inv(Xw.T @ Xw) @ (Xw.T @ Tw)


