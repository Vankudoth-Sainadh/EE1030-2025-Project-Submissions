
// Code by Harshith Chiruvella and Abhiram Reddy

#include <LiquidCrystal.h>

// Initialize LCD with pin numbers: RS, E, D4, D5, D6, D7
// NOTE: These pins must match your hardware wiring.
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

// ------------------------------------------------------------------
// YOUR CALCULATED COEFFICIENTS (Temperature as a function of Voltage)
// T = m + l*V + k*V^2
// ------------------------------------------------------------------
float m  = 203.025576;  // Intercept (m)
float l = -379.372317; // Linear V coefficient (l)
float k = 117.468670;  // Quadratic V^2 coefficient (k)

// Pin definitions
const int PT100_PIN = A0;

// Constants for PT100 and voltage divider
const float VCC = 5.0;            // Supply voltage for the Arduino

// Calibration parameters (use 0.0 and 1.0 if not needed)
float offset = 0.0;
float sensitivity = 1.0;

void setup() {
  Serial.begin(9600);

  // Initialize LCD, 16 columns and 2 rows
  lcd.begin(16, 2);

  // Display startup message
  lcd.print("PT100 Calib. Start");
  delay(2000);
  lcd.clear();

  analogReference(DEFAULT);  // Use default 5V reference
}

void loop() {
  // Read analog value from voltage divider
  int reading = analogRead(PT100_PIN);

  // Convert reading to voltage
  // 1023.0 is used for the standard 10-bit ADC resolution
  float voltage = (reading * VCC) / 1023.0;

  // ------------------------------------------------------------------
  // APPLY YOUR QUADRATIC CALIBRATION FUNCTION
  // T = m + l*V + k*V^2
  float temperature = m + l * voltage + k * voltage * voltage;
  // ------------------------------------------------------------------

  // Optional: Apply final calibration (sensitivity and offset)
  temperature = (temperature * sensitivity) + offset;

  // Show temperature on LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  // Print temperature with 3 decimal places
  lcd.print(temperature, 3); 
  lcd.print(" C");  

  // Show measured voltage on LCD
  lcd.setCursor(0, 1);
  lcd.print("V: ");
  // Print voltage with 3 decimal places for
  lcd.print(voltage, 3); 
  lcd.print(" V 08&21");

  delay(500); // Wait for 0.5 second before the next reading
}