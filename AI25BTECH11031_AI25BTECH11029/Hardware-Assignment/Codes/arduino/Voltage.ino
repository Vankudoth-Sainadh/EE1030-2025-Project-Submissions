#include <LiquidCrystal.h>

// Initialize the LCD with Arduino pins
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int sensorPin = A0; // PT100 + 100Ω midpoint

void setup() {
  lcd.begin(16, 2);        // Initialize LCD (16x2)
  lcd.print("PT100 Voltage:");
}

void loop() {
  // Read analog input (0–1023)
  int adcValue = analogRead(sensorPin);
  
  // Convert to actual voltage
  float voltage = (adcValue * 5.0) / 1023.0;

  // Display voltage on LCD
  lcd.setCursor(0, 1);
  lcd.print(voltage, 4);   // show 4 decimal places
  lcd.print(" V   ");      // clear leftover digits

  delay(1000);             // Update every second
}
