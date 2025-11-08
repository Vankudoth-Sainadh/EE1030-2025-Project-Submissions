
#include <LiquidCrystal.h>

// Initialize LCD with pin numbers: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

// Coefficients for 4th-degree polynomial (Temperature model)
float a0 = -102666.907358;
float a1 = 3.565468e+05;
float a2 = -4.629939e+05;
float a3 = 2.665158e+05;
float a4 = -5.735873e+04;

// Pin definitions
const int PT100_PIN = A0;

// Constants for PT100 and voltage divider
const float VCC = 5.0; // Supply voltage

// Calibration parameters (adjust after calibration)
float offset = 0.0;
float sensitivity = 1.0;

void setup() {
  Serial.begin(9600);

  // Initialize LCD, 16 columns and 2 rows
  lcd.begin(16, 2);

  // Display startup message
  lcd.print("PT100 Temp");
  delay(2000);
  lcd.clear();

  analogReference(DEFAULT); // Use default 5V reference
}

void loop() {
  // Read analog value from voltage divider
  int adcValue = analogRead(PT100_PIN);

  // Convert ADC value to voltage
  float voltage = (adcValue * VCC) / 1023.0;

  // Biquadratic (4th-degree) temperature calculation
  float temperature = a0 + a1*voltage + a2*voltage*voltage + a3*voltage*voltage*voltage + a4*voltage*voltage*voltage*voltage;

  // Calibrate temperature
  temperature = (temperature * sensitivity) + offset;

  // Show temperature on LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  lcd.print(temperature, 2);
  lcd.print(" C");

  lcd.setCursor(0, 1);
  lcd.print("V: ");
  lcd.print(voltage, 2);
  lcd.print(" V");

  delay(1000);
}