#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  lcd.begin(16, 2);
  lcd.print("Voltage and Temperature");
  delay(1000);
}

void loop() {
  int sensorValue = analogRead(A0);
  float voltage = sensorValue * (5.0 / 1023);
  
  float a0 = -7635.862161878817;
  float a1 = 5168.089243355189;
  float a2 = -863.6756921206706;

  temperature = a0 + (a1 * (voltage)) + (a2 * (voltage) * (voltage));

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("V:");
  lcd.print(voltage, 2);
  lcd.print(" V");

  lcd.setCursor(0, 1);
  lcd.print("T:");
  lcd.print(temperature, 2);
  lcd.print(" Celsius");

  delay(800);
}
