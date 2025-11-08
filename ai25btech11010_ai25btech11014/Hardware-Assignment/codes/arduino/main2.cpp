
#include <Arduino.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const float a0 = 2216.927002;
const float a1 = -1660.122419;
const float a2 = 309.13658;

constexpr int PT100_PIN = A2;

constexpr float VCC = 5.0f; 

float offset = 0.0f;       
float sensitivity = 1.0f; 

float readVoltage(int pin) {
  int adcValue = analogRead(pin);
  return (adcValue * VCC) / 1023.0f;
}

float computeTemperature(float voltage) {
  float temp = a0 + a1 * voltage + a2 * voltage * voltage;
  return (temp * sensitivity) + offset;
}

void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);
  lcd.print("PT100 Temp Init");
  delay(1000);
  lcd.clear();

  analogReference(DEFAULT); 
}

void loop() {
  float voltage = readVoltage(PT100_PIN);
  float temperature = computeTemperature(voltage);

  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  lcd.print(temperature, 2);
  lcd.print(" C");

  lcd.setCursor(0, 1);
  lcd.print("V: ");
  lcd.print(voltage, 3);
  lcd.print(" V");


  Serial.print("Voltage: ");
  Serial.print(voltage, 3);
  Serial.print(" V | Temp: ");
  Serial.print(temperature, 2);
  Serial.println(" C");

  delay(1000); 
}
