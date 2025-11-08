#include <LiquidCrystal.h>

// LCD pins: RS, EN, D4, D5, D6, D7
LiquidCrystal lcd(7, 6, 5, 4, 3, 2);

const int analogPin = A0;    // Analog input from PT100 amplifier
const float VCC = 5.0;       // Reference voltage (change to 3.3 if using 3.3V board)

// Updated regression coefficients (from latest calibration)
const float a = -172.5707556;
const float b = -40.8781243;
const float c = 91.0056826;

void setup() {
  lcd.begin(16, 2);          // Initialize 16x2 LCD
  lcd.print("PT100 Sensor");
  delay(1500);
  lcd.clear();

  Serial.begin(9600);
  Serial.println("PT100 Temperature Measurement");
  Serial.println("--------------------------------");
  Serial.println("Voltage (V)\tTemp (C)");
}

void loop() {
  // --- Read sensor voltage ---
  int raw = analogRead(analogPin);         
  float Vout = raw * (VCC / 1023.0);       // Convert ADC to voltage (0–5V)

  // --- Calculate temperature from quadratic equation ---
  float T = a + b * Vout + c * Vout * Vout;

  // --- Display voltage and temperature on LCD ---
  lcd.setCursor(0, 0);
  lcd.print("V=");
  lcd.print(Vout, 3);
  lcd.print(" V    ");

  lcd.setCursor(0, 1);
  lcd.print("T=");
  lcd.print(T, 2);
  lcd.print((char)223); // degree symbol
  lcd.print("C   ");

  // --- Print to Serial Monitor ---
  Serial.print(Vout, 3);
  Serial.print("\t\t");
  Serial.print(T, 2);
  Serial.println();

  delay(1000); // Update every 1 second
}