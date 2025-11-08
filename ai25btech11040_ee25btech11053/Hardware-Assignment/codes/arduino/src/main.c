// // Code for displaying temperatures.
#include <LiquidCrystal.h>
#include <math.h>

const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

void setup() {
  // set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
  // Print a message to the LCD.
  lcd.setCursor(0, 0); // Move to the first line
  lcd.print("Temperature:");
  // Serial.begin(9600);
}

void loop() {
  int r = analogRead(A0);
  float voltage = r * (5.0 / 1023.0);
  double a = 1628.651682076511, b = -1917.5811024129323, c = 581.6804791446455, d = 1, e = 1, f = 1; // Coefficients for calibration curve
  double temperature = a + b*voltage + c*pow(voltage, 2);// + d*pow(voltage, 3) + e*pow(voltage, 4) + f*pow(voltage, 5); // Convert voltage to temperature
  lcd.setCursor(0, 1); // Move to the second line
  lcd.print(temperature);
  lcd.print((char)223); // Degree symbol
  lcd.print("C");
}

// Code for reading voltages

// void setup() {
//   // set up the LCD's number of columns and rows:
//   lcd.begin(16, 2);
//   // Print a message to the LCD.
// }

// void loop() {
//   int a = analogRead(A0);
//   float voltage = a * (5.0 / 1023.0);

//   lcd.setCursor(0, 0); // Move to the first line
//   lcd.print("Voltage: ");
//   lcd.print(voltage);
//   lcd.print("V");
// }