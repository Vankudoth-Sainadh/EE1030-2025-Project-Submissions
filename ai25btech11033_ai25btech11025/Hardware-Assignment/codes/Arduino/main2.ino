#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
const int sensorPin = A0;
const int samplesPerPoint = 100;

void setup() {
  lcd.begin(16, 2);
  lcd.print("PT100 Ready...");
  delay(2000);
  lcd.clear();
}

void loop() {
  float sum = 0;
  for (int i = 0; i < samplesPerPoint; i++) {
    int raw = analogRead(sensorPin);
    float voltage = raw * (5.0 / 1023.0);
    sum += voltage;
    delay(10);
  }

  float avgVoltage = sum / samplesPerPoint;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("PT100 Voltage:");
  lcd.setCursor(0, 1);
  lcd.print(avgVoltage, 3);
  lcd.print(" V");

  delay(3000);
}
