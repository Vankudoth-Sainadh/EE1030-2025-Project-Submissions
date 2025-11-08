import numpy as np
import matplotlib.pyplot as plt
import serial
import time
from collections import deque
import joblib 
import os 

SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 9600
MAX_POINTS = 100
MODEL_FILES = ['poly_transformer.joblib', 'scaler.joblib', 'model.joblib']


print("Loading pre-trained models")
if not all(os.path.exists(f) for f in MODEL_FILES):
    print(f"Error: Model files not found ({MODEL_FILES})")
    print("Please run 'train_and_get_equation.py' first.")
    exit()

try:
    poly = joblib.load('poly_transformer.joblib')
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()
    
V_live = deque(maxlen=MAX_POINTS) 
T_actual_live = deque(maxlen=MAX_POINTS) 
T_pred_live = deque(maxlen=MAX_POINTS)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    time.sleep(2) 
    ser.flushInput()
except Exception as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}.")
    print(e)
    exit()


plt.ion() 
fig, ax = plt.subplots()
line_actual, = ax.plot([], [], 'k.', label='Actual Sensor Data')
line_pred, = ax.plot([], [], 'r-', label=f'Live Model Fit (T = f(V))')
ax.set_xlabel('Output Voltage (V)')
ax.set_ylabel(r'Temperature ($^{\circ}$C)')
ax.legend()
ax.grid(True)
plt.title('Live Model (T vs V) and Sensor Data')

print("Starting live data acquisition... Press Ctrl+C to stop.")
try:
    while True:
        try:
            line_bytes = ser.readline()
            print(f"Raw bytes: {line_bytes}")

            if not line_bytes:
                continue 
            line_str = line_bytes.decode('utf-8').strip()


            print(f"Decoded string: '{line_str}'")
            
            if ',' in line_str:
                print("Comma found, attempting to parse...")
                
                T_str, C_str = line_str.split(',')
                T_val = float(T_str) 
                C_val = float(C_str) 
                
                C_new = np.array([[C_val]]) 
                T_new_target = np.array([T_val])
                X_poly_new = poly.transform(C_new)
                X_scaled_new = scaler.transform(X_poly_new)

                T_pred_val = model.predict(X_scaled_new)[0]
                
                model.partial_fit(X_scaled_new, T_new_target)
                
                V_live.append(C_val) 
                T_actual_live.append(T_val)
                T_pred_live.append(T_pred_val)
                
                line_actual.set_data(V_live, T_actual_live)
                line_pred.set_data(V_live, T_pred_live)
                
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

            else:
                print("No comma in string, skipping line.")

        except Exception as e:
            print(f"--- DATA ERROR ---")
            print(f"Error: {e}")
            print(f"Raw data that caused error: {line_bytes}")
            print(f"--------------------")

except KeyboardInterrupt:
    print("\n\n--- STOPPING SCRIPT (Ctrl+C) ---")
    print("Note: Live model tuning will not be saved.")

finally:
    ser.close()
    print("Serial port closed.")
    plt.ioff() 
    print("Final plot window closed.")
    plt.show()

