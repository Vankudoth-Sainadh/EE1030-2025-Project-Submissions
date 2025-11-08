#include <LiquidCrystal.h>

// LCD pins: RS, EN, D4, D5, D6, D7
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int analogPin = A0;

// Quadratic coefficients (from least-squares fit)
const float a = -18.977056;
const float b = 316.558739;
const float c = -258.655995;

// ADC reference and resolution
const float ADC_REF = 5.0;    // change to 3.3 if you use 3.3V reference
const int ADC_RES = 1023;     // 10-bit ADC for Uno

void setup() {
  lcd.begin(16, 2);          // initialize the LCD
  Serial.begin(9600);
  lcd.print("PT100 Quadratic");
  delay(1500);
  lcd.clear();
}

void loop() {
  int adcValue = analogRead(analogPin);
  // Convert ADC reading to measured voltage at A0
  float voltage = (adcValue * ADC_REF) / ADC_RES -0.1;

  // Compute temperature using quadratic
  // T = a*V^2 + b*V + c
  float temperature = a * voltage * voltage + b * voltage + c;

  // Display on LCD
  lcd.setCursor(0, 0);
  lcd.print("V=");
  lcd.print(voltage, 3);      // 3 decimal places
  lcd.print(" V   ");         // pad to clear leftover chars

  lcd.setCursor(0, 1);
  lcd.print("T=");
  lcd.print(temperature, 1);  // 1 decimal place
  lcd.print((char)223);       // degree symbol
  lcd.print("C   ");          // pad

  // Serial output for logging
  Serial.print("ADC=");
  Serial.print(adcValue);
  Serial.print("  V=");
  Serial.print(voltage, 4);
  Serial.print("  T=");
  Serial.print(temperature, 3);
  Serial.println(" C");

  delay(1000); // update every 1 second
}
