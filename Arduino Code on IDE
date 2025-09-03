#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>


// ----------------------------
// LCD Setup (I2C)
// ----------------------------
LiquidCrystal_I2C lcd(0x27, 16, 2);


// ----------------------------
// Pin Definitions
// ----------------------------
#define HEART_SENSOR A0
#define THERMISTOR A1
#define PIEZO_SENSOR A2
#define BUZZER_PIN 8
#define LO_PLUS 10
#define LO_MINUS 11


// ----------------------------
// Respiratory Rate Variables
// ----------------------------
const int piezoPin = PIEZO_SENSOR;
const int sampleInterval = 2000;  // ms between samples
int lastPiezoValue = 0;
unsigned long lastPeakTime = 0;
float respRate = 0.0;


// Adjusted for high baseline (590–600)
const int PIEZO_BASELINE = 595;  
const int PIEZO_THRESHOLD = 4;  // deviation from baseline for breath detection


// Rolling average for breaths
const int MAX_BREATHS = 5;
unsigned long breathIntervals[MAX_BREATHS];
int breathIndex = 0;
bool breathFilled = false;


// ----------------------------
// Heart Rate Variables
// ----------------------------
int lastHrValue = 0;
unsigned long lastHrPeakTime = 0;
float heartRateBPM = 0.0;


// ----------------------------
// Thermistor Variables
// ----------------------------
const float THERMISTOR_NOMINAL = 10000.0; // 10k ohm at 25°C
const float TEMPERATURE_NOMINAL = 25.0;   // °C
const float B_COEFFICIENT = 3950.0;       // Thermistor beta coefficient
const float SERIES_RESISTOR = 10000.0;    // 10k resistor in voltage divider


// ----------------------------
// Setup
// ----------------------------
void setup() {
  Serial.begin(9600);
  lcd.init();
  lcd.backlight();


  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);
}


// ----------------------------
// Helper: Piezo Peak Detection
// ----------------------------
bool detectPiezoPeak(int currentValue, int &lastValue, bool &risingFlag) {
  if (currentValue > PIEZO_BASELINE + PIEZO_THRESHOLD && !risingFlag && currentValue > lastValue) {
    risingFlag = true;
    lastValue = currentValue;
    return true;
  }
  if (currentValue < PIEZO_BASELINE - PIEZO_THRESHOLD) {
    risingFlag = false;
  }
  lastValue = currentValue;
  return false;
}


// ----------------------------
// Helper: Heart Rate Peak Detection
// ----------------------------
bool detectHrPeak(int currentValue, int &lastValue, bool &risingFlag, int highThreshold, int lowThreshold) {
  if (currentValue > highThreshold && !risingFlag && currentValue > lastValue) {
    risingFlag = true;
    lastValue = currentValue;
    return true;
  }
  if (currentValue < lowThreshold) {
    risingFlag = false;
  }
  lastValue = currentValue;
  return false;
}


// ----------------------------
// Main Loop
// ----------------------------
void loop() {
  static unsigned long lastDisplayTime = 0;
  static bool showVitals = true;  // toggle flag
  unsigned long currentTime = millis();


  // --- 1. Read raw sensors ---
  int rawHeart = analogRead(HEART_SENSOR);
  int rawTherm = analogRead(THERMISTOR);
  int rawPiezo = analogRead(PIEZO_SENSOR);


  // --- 2. Convert Thermistor to Celsius ---
  float voltage = rawTherm * (5.0 / 1023.0);
  float resistance = SERIES_RESISTOR * (5.0 / voltage - 1.0);
  float tempC = 1.0 / (log(resistance / THERMISTOR_NOMINAL) / B_COEFFICIENT + 1.0 / (TEMPERATURE_NOMINAL + 273.15)) - 273.15;


  // --- 3. Detect Piezo Peaks for Respiratory Rate ---
  static bool piezoRising = false;
  if (detectPiezoPeak(rawPiezo, lastPiezoValue, piezoRising)) {
    if (lastPeakTime != 0) {
      unsigned long interval = currentTime - lastPeakTime;
      breathIntervals[breathIndex] = interval;
      breathIndex = (breathIndex + 1) % MAX_BREATHS;
      if (breathIndex == 0) breathFilled = true;


      unsigned long sum = 0;
      int count = breathFilled ? MAX_BREATHS : breathIndex;
      for (int i = 0; i < count; i++) sum += breathIntervals[i];
      float avgInterval = (float)sum / count;
      respRate = 60000.0 / avgInterval;
    }
    lastPeakTime = currentTime;
  }


  // --- 4. Detect Heart Rate Peaks for BPM ---
  static bool hrRising = false;
  if (detectHrPeak(rawHeart, lastHrValue, hrRising, 600, 500)) {
    if (lastHrPeakTime != 0) {
      heartRateBPM = 60000.0 / (currentTime - lastHrPeakTime);
    }
    lastHrPeakTime = currentTime;
  }


  // --- 5. Create JSON ---
  StaticJsonDocument<200> doc;
  doc["temperature"] = tempC;
  doc["heart_rate"] = heartRateBPM;
  doc["resp_rate"] = respRate;


  String jsonData;
  serializeJson(doc, jsonData);


  // --- 6. Send to Serial ---
  Serial.println(jsonData);


  // --- 7. Read ML alert from Serial ---
  String alertMsg = "NORMAL";
  if (Serial.available() > 0) {
    alertMsg = Serial.readStringUntil('\n');
    alertMsg.trim();
  }


  // --- 8. Trigger Buzzer ---
  if (alertMsg != "NORMAL") digitalWrite(BUZZER_PIN, HIGH);
  else digitalWrite(BUZZER_PIN, LOW);


  // --- 9. Alternate LCD display every 2 seconds ---
  if (currentTime - lastDisplayTime >= 2000) {
    lastDisplayTime = currentTime;
    showVitals = !showVitals;  // toggle


    lcd.clear();
    if (showVitals) {
      lcd.setCursor(0, 0);
      lcd.print("Temp:");
      lcd.print(tempC, 1);
      lcd.setCursor(0, 1);
      lcd.print("HR:");
      lcd.print(heartRateBPM, 0);
      lcd.print(" RR:");
      lcd.print(respRate, 0);
    } else {
      lcd.setCursor(0, 0);
      lcd.print("ALERT:");
      lcd.setCursor(0, 1);
      lcd.print(alertMsg);
    }
  }


  delay(100); // small delay for stability
}
