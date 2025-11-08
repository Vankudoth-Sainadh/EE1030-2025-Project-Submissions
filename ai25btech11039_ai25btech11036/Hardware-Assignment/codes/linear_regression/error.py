from sklearn.metrics import mean_absolute_error

# assuming T_actual and T_pred are already defined
mae = mean_absolute_error(T_actual, T_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
