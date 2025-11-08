import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

A = np.loadtxt('../Data/training_data.txt')
X = np.hstack((np.ones((A.shape[0],1)),A[:,[0]],A[:,[0]]**2))
T = A[:,[0]]
C = A[:,[1]]

B = np.loadtxt('../Data/validation_data.txt')
Xv = np.hstack((np.ones((B.shape[0],1)),B[:,[0]],B[:,[0]]**2))
Cv = B[:,[1]]
Tv = B[:,[0]]

v, av, bv = np.linalg.lstsq(X, C, rcond=None)[0]
n_lsq = np.zeros((3,1))
v = np.array(v).item()     
av = np.array(av).item()
bv = np.array(bv).item()

n_lsq[0][0] = v
n_lsq[1][0] = av
n_lsq[2][0] = bv

y_train_pred = X @ n_lsq
y_valid_pred = Xv @ n_lsq

mse_train = mean_squared_error(C, y_train_pred)
mae_train = mean_absolute_error(C, y_train_pred)
r2_train = r2_score(C, y_train_pred)

mse_valid = mean_squared_error(Cv, y_valid_pred)
mae_valid = mean_absolute_error(Cv, y_valid_pred)
r2_valid = r2_score(Cv, y_valid_pred)

print("MSE (train, valid):", np.round(mse_train, 6), np.round(mse_valid, 6))
print("MAE (train, valid):", np.round(mae_train, 6), np.round(mae_valid, 6))
print("R2 (train, valid):", np.round(r2_train, 6), np.round(r2_valid, 6))

res_train = abs(C - y_train_pred)
res_valid = abs(Cv - y_valid_pred)

plt.plot(T, res_train, 'bo-', label='Train Residuals')
plt.plot(Tv, res_valid, 'ro-', label='Validation Residuals')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Temperature (\u00B0C)')
plt.ylabel('Residual (V)')
plt.legend()
plt.grid()
plt.ylim(0, 0.04)
plt.tight_layout()
plt.savefig('../../../figs/residuals.png')




