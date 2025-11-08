import pandas as pd
import numpy as np

# Load the testing.csv file
data = pd.read_csv('testing.csv')

# Define function to calculate MAE
def calculate_mae(predicted, actual):
    absolute_errors = np.abs(predicted - actual)
    mae = np.mean(absolute_errors)
    return mae

# Extract predicted and actual temperature columns
predicted_tp = data['T_p']
actual_t = data['T']

# Calculate MAE between predicted tp and actual t
tp_mae = calculate_mae(predicted_tp, actual_t)

print(f"Absolute Mean Error between predicted tp and actual t: {tp_mae}")

