#include "Arduino.h"

#include <LiquidCrystal.h>  // include LCD library
#include "Arduino.h"

// Initialize the LCD (RS, E, D4, D5, D6, D7)
LiquidCrystal lcd(12, 11, 4, 5, 6, 7);

// Coefficients from least squares fit
const double a = 860.735844;
const double b = -1117.636392;
const double c = 369.289134;

void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2); // initialize 16x2 LCD
  lcd.print("PT100 Thermometer");
  delay(2000);
  lcd.clear();
}

void loop() {
  int sensorValue = analogRead(A0);               // read ADC value
  double voltage = (5.0 * sensorValue / 1023.0);  // convert to voltage (0–5V)
  
  // Calculate temperature from quadratic relation
  double temperature = a + b * voltage + c * voltage * voltage;

  // Print values on Serial Monitor
  Serial.print("Voltage (V): ");
  Serial.print(voltage, 3);
  Serial.print("   Temperature (C): ");
  Serial.println(temperature, 2);

  // Display on LCD
  lcd.setCursor(0, 0);
  lcd.print("V: ");
  lcd.print(voltage, 3);
  lcd.print(" V     ");

  lcd.setCursor(0, 1);
  lcd.print("T: ");
  lcd.print(temperature, 2);
  lcd.print(" C     ");

  delay(1000); // update every second
}
