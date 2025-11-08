#include <LiquidCrystal.h>

// LCD pin connections: RS, EN, D4, D5, D6, D7
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int sensorPin = A0;   // PT100 voltage divider connected to A0
float voltage;

void setup() {
  lcd.begin(16, 2);               // Initialize 16x2 LCD
  lcd.print("PT100 Sensor");      // Welcome text
  delay(2000);                    // Wait 2 sec so you can see it
  lcd.clear();                    // Clear screen before showing readings
}

void loop() {
  int rawValue = analogRead(sensorPin);       // Read ADC value (0–1023)
  voltage = (rawValue * 5.0) / 1023.0;        // Convert to voltage (if 5V board)
  // voltage = (rawValue * 3.3) / 1023.0;     // If using 3.3V board like ESP32/Nano 33

  lcd.setCursor(0, 0);
  lcd.print("Voltage: ");
  lcd.print(voltage, 3);                     // Show 2 decimal places
  lcd.print(" V   ");                        // Spaces clear old digits

  lcd.setCursor(0, 1);
  lcd.print("Raw ADC: ");
  lcd.print(rawValue);
  lcd.print("   ");

  delay(1000);                               // Update every 1 second
}