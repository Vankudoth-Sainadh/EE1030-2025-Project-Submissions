#include <LiquidCrystal.h>
#include <Arduino.h>
// Pin connections (adjust if needed)
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);  // RS, E, D4, D5, D6, D7

const int sensorPin = A0;

const float c = -409.383968;
const float b = 681.885486;
const float a = -161.730128;


void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);
  lcd.print("PT100 Thermometer");
  delay(1500);
  lcd.clear();
}

void loop() {
 
  float voltage = (analogRead(A0) / 1023.0) * 5.0;
  float temperature = a * voltage * voltage + b * voltage + c;

  lcd.setCursor(0, 0);
  lcd.print("V: ");
  lcd.print(voltage, 3);
  lcd.print(" V   ");

  lcd.setCursor(0, 1);
  lcd.print("T: ");
  lcd.print(temperature, 2);
  lcd.print((char)223);
  lcd.print("C   ");

  
 
  Serial.print("\tV = ");
  Serial.print(voltage, 3);
  Serial.print(" V\tT = ");
  Serial.print(temperature, 2);
  Serial.println(" °C");

  delay(1000);
}
