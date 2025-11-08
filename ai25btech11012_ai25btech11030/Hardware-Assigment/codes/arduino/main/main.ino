#include <LiquidCrystal.h>

// LCD pin configuration: RS, EN, D4, D5, D6, D7
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int sensorPin = A0;   // Analog input from PT100 bridge circuit

float voltage = 0.0;
float temperature = 0.0;
float filteredVoltage = 0.0;
float lastTemperature = 0.0;

const float alpha = 0.1;    // smoothing factor (0.0–1.0, smaller = smoother)

void setup() {
  lcd.begin(16, 2);
  analogReference(DEFAULT);  // Use 5V reference (change to INTERNAL if <1.1V signal)
  
  lcd.print("Digital Thermo");
  delay(2000);
  lcd.clear();

  // Optional: Serial monitor for debugging
  Serial.begin(9600);
}

void loop() {
  // --- Step 1: Average multiple samples to reduce ADC noise ---
  int numSamples = 50;
  long total = 0;
  for (int i = 0; i < numSamples; i++) {
    total += analogRead(sensorPin);
    delay(2);
  }
  float rawValue = total / (float)numSamples;

  // --- Step 2: Convert ADC to voltage (assuming 5V reference) ---
  float newVoltage = rawValue * (5.0 / 1023.0);

  // --- Step 3: Apply exponential smoothing for stability ---
  filteredVoltage = alpha * newVoltage + (1 - alpha) * filteredVoltage;
  voltage = filteredVoltage;

  // --- Step 4: Compute temperature using regression curve ---
  temperature = 1181.19 - 1230.93 * voltage + 305.94 * voltage * voltage;

  // --- Step 5: Update LCD only if change is significant (>0.2°C) ---
  if (abs(temperature - lastTemperature) > 0.2) {
    lcd.clear();
    
    lcd.setCursor(0, 0);
    lcd.print("V:");
    lcd.print(voltage, 3);
    lcd.print(" V");
    
    lcd.setCursor(0, 1);
    lcd.print("T:");
    lcd.print(temperature, 2);
    lcd.print(" C");
    
    lastTemperature = temperature;
  }

  // --- Optional: Serial output for testing ---
  Serial.print("Voltage: ");
  Serial.print(voltage, 3);
  Serial.print(" V, Temp: ");
  Serial.print(temperature, 2);
  Serial.println(" C");

  delay(500); // 0.5 second refresh rate
}
