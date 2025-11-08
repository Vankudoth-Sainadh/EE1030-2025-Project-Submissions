#include "Arduino.h"
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const double a = +128.10009657;
const double b = 105.14206697;
const double c = -188.04017689;

void setup() {
  lcd.begin(16, 2);
}

void loop() {
  int sensorValue = analogRead(A0);
  double voltage = (5.0 * sensorValue / 1023.0);
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  double temperature = c + b * voltage + a * voltage * voltage;
  lcd.setCursor(0, 1);
  lcd.print(temperature);
  lcd.print(" C");
  delay(1000);
}