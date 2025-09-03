# Integrated Arduino and Machine Learning Patient Monitoring Platform
This project presents a low-cost, real-time patient monitoring system that integrates Arduino-based sensor acquisition with machine learning–driven health prediction. The system continuously tracks three vital signs:

-> Heart Rate (BPM) – measured using the AD8232 ECG sensor
-> Respiratory Rate (Breaths/min) – measured using a chest-mounted piezoelectric sensor
-> Body Temperature (°C) – measured using an NTC103 thermistor

The raw analog signals are processed by the Arduino and displayed on an I2C LCD for immediate feedback. Simultaneously, the data is transmitted over serial to a Python ML model, which classifies the patient’s status and generates predictive alerts for timely intervention.

## Workflow
1) Data Acquisition: Sensors (AD8232, piezo, thermistor) capture raw patient signals.
2) Arduino Processing: The Arduino converts raw data into meaningful physical units (BPM, Breaths/min, °C) and updates the LCD.
3) Serial Communication: The Arduino streams JSON-formatted data to a Python environment.
4) ML Model Inference: Python-based ML model classifies the patient’s condition (Normal / Critical) using real-time input.
5) Feedback Loop: The prediction is sent back to the Arduino, which alternates between showing vitals and alerts on the LCD.

## Tech Stack
- Hardware: Arduino Uno, AD8232 ECG module, NTC103 thermistor, piezoelectric sensor, 16x2 I2C LCD
- Software: Arduino IDE (C++), Python (NumPy, Matplotlib, scikit-learn, pyserial, ArduinoJson)
- Machine Learning: Model trained on synthetic datasets to classify patient states and highlight anomalies
