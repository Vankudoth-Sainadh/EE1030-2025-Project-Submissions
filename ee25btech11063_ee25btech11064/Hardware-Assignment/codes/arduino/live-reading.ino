#include "Arduino.h"
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
float a = 430.951;
float b = 1152.413;
float c = 757.306;
void setup() {
  lcd.begin(16, 2);
  lcd.print("Calculating...");
  delay(1000);
  lcd.clear();
}
void loop() {
  int sensorValue = analogRead(A0);
  float V = (5.0 * sensorValue / 1023.0);
  float T = a - (b * V) + (c * V * V);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("V = ");
  lcd.print(V, 3);
  lcd.print(" V");
  lcd.setCursor(0, 1);
  lcd.print("T = ");
  lcd.print(T, 2);
  lcd.print(" ");
  lcd.write(byte(223));
  lcd.print("C");
  delay(1000);
}
