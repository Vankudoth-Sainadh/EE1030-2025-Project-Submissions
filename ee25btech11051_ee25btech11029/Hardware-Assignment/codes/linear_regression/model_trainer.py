import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import joblib
import os

print("--- Part 1: Model Trainer & Equation Generator ---")

# Check if model files already exist
MODEL_FILES = ['poly_transformer.joblib', 'scaler.joblib', 'model.joblib']
if all(os.path.exists(f) for f in MODEL_FILES):
    print("Model files already exist. Skipping training.")
else:
    try:
        print("Loading 'training_data.txt'...")
        A = np.loadtxt('training_data.txt')
        T_train = A[:,[0]]
        C_train = A[:,[1]]
    except Exception as e:
        print(f"Error: Could not load 'training_data.txt'. {e}")
        exit()

    print("Training model T = f(V)...")

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(C_train.reshape(-1, 1))
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_poly_train)

    model = SGDRegressor(eta0=0.01)
    model.fit(X_scaled_train, T_train.ravel())

    print("Training complete.")

    try:
        joblib.dump(poly, 'poly_transformer.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(model, 'model.joblib')
        print("Successfully saved models to:")
        print("  - poly_transformer.joblib")
        print("  - scaler.joblib")
        print("  - model.joblib")
    except Exception as e:
        print(f"Error saving models: {e}")
        
# --- Display the equation (this part is just for you) ---
try:
    # Load the models to print the equation
    poly = joblib.load('poly_transformer.joblib')
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')

    if hasattr(model, 'coef_'):
        w0 = model.intercept_[0]
        w1 = model.coef_[0]
        w2 = model.coef_[1]
        
        V_mean = scaler.mean_[0]
        V_scale = scaler.scale_[0]
        V2_mean = scaler.mean_[1]
        V2_scale = scaler.scale_[1]
        
        print("\n--- FINAL MODEL EQUATION ---")
        print("T = Temperature, V = Voltage\n")
        print(f"Model: T = w0 + w1*((V - V_max)/V_scale) + w2*((V^2 - V2_max)/V2_scale)\n")
        print("Your final equation is:")
        # --- THIS IS THE FIX ---
        # It was V2_max before, which was undefined. It should be V2_mean.
        print(f"T \u2248 {w0:.6f} + {w1:.6f} * ((V - {V_mean:.6f}) / {V_scale:.6f}) + {w2:.6f} * ((V*V - {V2_mean:.6f}) / {V2_scale:.6f})")
    else:
        print("Error: Model was not trained successfully.")
except Exception as e:
    print(f"Could not print equation: {e}")

