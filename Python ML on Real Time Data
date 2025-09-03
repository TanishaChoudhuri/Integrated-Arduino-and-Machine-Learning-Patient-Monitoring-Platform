# real_time_patient_monitoring_static_plot_labeled.py


import serial
import serial.tools.list_ports
import json
import joblib
import time
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt


# -----------------------------
# 1. Auto-detect Arduino COM port
# -----------------------------
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "Arduino" in p.description or "CH340" in p.description or "USB Serial Device" in p.description:
            return p.device
    return None


SERIAL_PORT = find_arduino_port()
if not SERIAL_PORT:
    raise Exception("Arduino not found. Check USB connection and COM ports.")
print(f"Arduino detected on port: {SERIAL_PORT}")


BAUD_RATE = 9600


# -----------------------------
# 2. Setup Serial Connection
# -----------------------------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
time.sleep(3)  # allow Arduino to reset


# -----------------------------
# 3. Load Trained ML Models
# -----------------------------
clf_event = joblib.load("clf_event.pkl")
clf_fever = joblib.load("clf_fever.pkl")
clf_resp = joblib.load("clf_resp.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# 4. Real-time Data Reading & Prediction
# -----------------------------
print("Reading data from Arduino and running ML predictions...\n")


table_data = []


# Lists to store readings for static plot
temp_vals, hr_vals, resp_vals = [], [], []
temp_alerts, hr_alerts, resp_alerts = [], [], []


try:
    while True:
        raw_line = ser.readline()
        if not raw_line:
            continue


        # --- Handle decoding safely ---
        try:
            line = raw_line.decode('utf-8').strip()
        except UnicodeDecodeError:
            continue  # skip corrupted line


        if not line:
            continue


        # Parse JSON from Arduino
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip invalid JSON


        temp = float(data.get("temperature", 0))
        hr = float(data.get("heart_rate", 0))
        rr = float(data.get("resp_rate", 0))


        # Prepare features for ML
        X_df = pd.DataFrame([{
            "temperature": temp,
            "heart_rate": hr,
            "resp_rate": rr
        }])
        X_scaled = scaler.transform(X_df)


        # Make Predictions
        event_pred = int(clf_event.predict(X_scaled)[0])
        fever_pred = int(clf_fever.predict(X_scaled)[0])
        resp_pred = int(clf_resp.predict(X_scaled)[0])


        # Append to table
        table_data.append([
            round(temp, 1),
            round(hr, 0),
            round(rr, 0),
            "Alert" if event_pred else "Normal",
            "Fever" if fever_pred else "No Fever",
            "Resp Anomaly" if resp_pred else "Normal Resp"
        ])
        if len(table_data) > 10:
            table_data.pop(0)


        # Terminal table (live update)
        print("\033c", end="")  # clear terminal
        print(tabulate(table_data, headers=["Temperature","Heart Rate","Resp Rate","Event","Fever","Respiratory"], tablefmt="grid"))


        # Store readings for static plot
        temp_vals.append(temp)
        hr_vals.append(hr)
        resp_vals.append(rr)


        if fever_pred:
            temp_alerts.append(len(temp_vals)-1)
        if event_pred:
            hr_alerts.append(len(temp_vals)-1)
        if resp_pred:
            resp_alerts.append(len(temp_vals)-1)


        # --- 5. Send ML alert back to Arduino ---
        alert_str = "NORMAL"
        if event_pred:
            alert_str = "EVENT ALERT"
        elif fever_pred:
            alert_str = "FEVER"
        elif resp_pred:
            alert_str = "RESP ALERT"


        ser.write((alert_str + "\n").encode())  # send alert


except KeyboardInterrupt:
    print("\nStopping...")
    ser.close()


# -----------------------------
# 6. Static plot with labeled alerts
# -----------------------------
plt.figure(figsize=(10,8))


# Temperature
plt.subplot(3,1,1)
plt.plot(temp_vals, color='blue', label='Temperature')
for i in temp_alerts:
    plt.scatter(i, temp_vals[i], color='red')
    plt.text(i, temp_vals[i]+0.1, f"{temp_vals[i]:.1f}", color='red', fontsize=8)
plt.ylabel("Temp (°C)")
plt.legend()


# Heart Rate
plt.subplot(3,1,2)
plt.plot(hr_vals, color='blue', label='Heart Rate')
for i in hr_alerts:
    plt.scatter(i, hr_vals[i], color='red')
    plt.text(i, hr_vals[i]+0.5, f"{hr_vals[i]:.0f}", color='red', fontsize=8)
plt.ylabel("HR (bpm)")
plt.legend()


# Respiratory Rate
plt.subplot(3,1,3)
plt.plot(resp_vals, color='blue', label='Resp Rate')
for i in resp_alerts:
    plt.scatter(i, resp_vals[i], color='red')
    plt.text(i, resp_vals[i]+0.2, f"{resp_vals[i]:.0f}", color='red', fontsize=8)
plt.ylabel("RR (bpm)")
plt.xlabel("Reading #")
plt.legend()


plt.tight_layout()
plt.show()
