#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5,4,3,2);
const int sensorPin = A0;

void setup() {
  Serial.begin(9600);
  delay(50);
  lcd.begin(16, 2);
}

void loop() {
  int adcValue = analogRead(sensorPin);
  float voltage = (adcValue / 1023.0) * 5.0;
  float T = -1254.58871629 +787.24938821 * voltage -107.35266583 * voltage * voltage; 

  lcd.setCursor(0,0);
  lcd.print("Temp :");
  lcd.print(T, 2);
  lcd.print(" C");
  lcd.setCursor(0, 1);
  lcd.print("Volt :");
  lcd.print(voltage, 2);
  lcd.print(" V");

  Serial.print("Voltage: ");
  Serial.print(voltage, 3);
  Serial.print("V | Temp: ");
  Serial.print(T, 2);
  Serial.println("C");

  delay(1000);
}