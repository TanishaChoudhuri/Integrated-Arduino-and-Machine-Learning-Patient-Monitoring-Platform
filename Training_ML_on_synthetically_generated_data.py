# full_patient_monitoring_ml_autoport_fixed_plot.py


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import threading
import time
import requests
import socket
from tabulate import tabulate
import uvicorn
import matplotlib.pyplot as plt


# ------------------------------
# 1. Generate synthetic dataset
# ------------------------------
n_samples = 1000
np.random.seed(42)


temperature = np.random.normal(37, 0.7, n_samples)
heart_rate = np.random.normal(75, 10, n_samples)
resp_rate = np.random.normal(18, 4, n_samples)


event = ((temperature>38) | (temperature<36) |
         (heart_rate>100) | (heart_rate<50) |
         (resp_rate>25) | (resp_rate<12)).astype(int)


fever = (temperature > 38).astype(int)
resp_anomaly = ((resp_rate>25) | (resp_rate<10)).astype(int)


df = pd.DataFrame({
    'temperature': temperature,
    'heart_rate': heart_rate,
    'resp_rate': resp_rate,
    'event': event,
    'fever': fever,
    'resp_anomaly': resp_anomaly
})


# ------------------------------
# 2. Preprocess
# ------------------------------
X = df[['temperature', 'heart_rate', 'resp_rate']]
y_event = df['event']
y_fever = df['fever']
y_resp = df['resp_anomaly']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split dataset
X_train, X_test, y_train_event, y_test_event = train_test_split(X_scaled, y_event, test_size=0.2, random_state=42)
_, _, y_train_fever, y_test_fever = train_test_split(X_scaled, y_fever, test_size=0.2, random_state=42)
_, _, y_train_resp, y_test_resp = train_test_split(X_scaled, y_resp, test_size=0.2, random_state=42)


# ------------------------------
# 3. Train ML Models
# ------------------------------
clf_event = RandomForestClassifier(n_estimators=100, random_state=42)
clf_event.fit(X_train, y_train_event)


clf_fever = RandomForestClassifier(n_estimators=100, random_state=42)
clf_fever.fit(X_train, y_train_fever)


clf_resp = RandomForestClassifier(n_estimators=100, random_state=42)
clf_resp.fit(X_train, y_train_resp)


# ------------------------------
# 4. Evaluate
# ------------------------------
print("Event Detection Accuracy:", accuracy_score(y_test_event, clf_event.predict(X_test)))
print("Fever Prediction Accuracy:", accuracy_score(y_test_fever, clf_fever.predict(X_test)))
print("Respiratory Anomaly Accuracy:", accuracy_score(y_test_resp, clf_resp.predict(X_test)))


# ------------------------------
# 5. Save models & scaler
# ------------------------------
joblib.dump(clf_event, 'clf_event.pkl')
joblib.dump(clf_fever, 'clf_fever.pkl')
joblib.dump(clf_resp, 'clf_resp.pkl')
joblib.dump(scaler, 'scaler.pkl')


# ------------------------------
# 6. FastAPI Inference Server
# ------------------------------
app = FastAPI(title="Patient Monitoring ML API")


class SensorData(BaseModel):
    temperature: float
    heart_rate: float
    resp_rate: float


# Load models
clf_event = joblib.load('clf_event.pkl')
clf_fever = joblib.load('clf_fever.pkl')
clf_resp = joblib.load('clf_resp.pkl')
scaler = joblib.load('scaler.pkl')


@app.post("/predict")
def predict(data: SensorData):
    features_df = pd.DataFrame([{
        'temperature': data.temperature,
        'heart_rate': data.heart_rate,
        'resp_rate': data.resp_rate
    }])
    features_scaled = scaler.transform(features_df)


    event_pred = int(clf_event.predict(features_scaled)[0])
    fever_pred = int(clf_fever.predict(features_scaled)[0])
    resp_pred = int(clf_resp.predict(features_scaled)[0])


    return {
        "event_alert": event_pred,
        "fever_alert": fever_pred,
        "resp_alert": resp_pred
    }


# ------------------------------
# 7. Simulation client with plotting
# ------------------------------
def simulation_client(port, max_iterations=30):
    url = f"http://127.0.0.1:{port}/predict"
    table_data = []


    temp_vals, hr_vals, resp_vals = [], [], []
    temp_alerts, hr_alerts, resp_alerts = [], [], []


    for i in range(max_iterations):
        sample = {
            "temperature": np.random.normal(37, 1),
            "heart_rate": np.random.normal(75, 10),
            "resp_rate": np.random.normal(18, 4)
        }
        try:
            response = requests.post(url, json=sample)
            res = response.json()
            table_data.append([
                round(sample['temperature'], 2),
                round(sample['heart_rate'], 2),
                round(sample['resp_rate'], 2),
                "Alert" if res['event_alert'] else "Normal",
                "Fever" if res['fever_alert'] else "No Fever",
                "Resp Anomaly" if res['resp_alert'] else "Normal Resp"
            ])


            temp_vals.append(sample['temperature'])
            hr_vals.append(sample['heart_rate'])
            resp_vals.append(sample['resp_rate'])


            if res['fever_alert']:
                temp_alerts.append(i)
            if res['event_alert']:
                hr_alerts.append(i)
            if res['resp_alert']:
                resp_alerts.append(i)


        except:
            print("Server not ready yet...")
        time.sleep(1)


    # Print table
    print("\nSimulation Results (30 readings, human-readable):\n")
    print(tabulate(table_data, headers=["Temperature", "Heart Rate", "Resp Rate", "Event", "Fever", "Respiratory"], tablefmt="grid"))


    # --------------------------
    # Static plots after simulation
    # --------------------------
    plt.figure(figsize=(10, 8))


    # Temperature
    plt.subplot(3, 1, 1)
    plt.plot(temp_vals, color='blue', label='Temperature')
    if temp_alerts:
        plt.scatter(temp_alerts, [temp_vals[i] for i in temp_alerts], color='red', label='Fever Alert')
    plt.ylabel("Temperature (°C)")
    plt.legend()


    # Heart Rate
    plt.subplot(3, 1, 2)
    plt.plot(hr_vals, color='blue', label='Heart Rate')
    if hr_alerts:
        plt.scatter(hr_alerts, [hr_vals[i] for i in hr_alerts], color='red', label='Event Alert')
    plt.ylabel("Heart Rate (bpm)")
    plt.legend()


    # Respiratory Rate
    plt.subplot(3, 1, 3)
    plt.plot(resp_vals, color='blue', label='Resp Rate')
    if resp_alerts:
        plt.scatter(resp_alerts, [resp_vals[i] for i in resp_alerts], color='red', label='Resp Anomaly')
    plt.ylabel("Resp Rate (bpm)")
    plt.xlabel("Reading #")
    plt.legend()


    plt.tight_layout()
    plt.show()


# ------------------------------
# 8. Find free port
# ------------------------------
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ------------------------------
# 9. Run server + client with automatic shutdown
# ------------------------------
if __name__ == "__main__":
    port = find_free_port()
    print(f"Starting server on port {port}...")


    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)


    # Run server in a separate thread
    server_thread = threading.Thread(target=server.run)
    server_thread.start()


    # Wait for server to start
    time.sleep(3)


    # Run simulation client
    simulation_client(port, max_iterations=30)


    # Shutdown server automatically
    print("\nSimulation complete. Shutting down server...")
    server.should_exit = True
    server_thread.join()
    print("Server stopped.")
