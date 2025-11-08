#include <Arduino.h>
#include <LiquidCrystal.h>
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
void setup() {
  lcd.begin(16, 2);
  lcd.setCursor(0,0);
  lcd.print("Digital");
  lcd.setCursor(0,1);
  lcd.print("Thermometer");
  delay(3000);
}

void loop() {
  const int numSamples = 15;
  long total = 0;

  for (int i = 0; i < numSamples; i++) {
    total += analogRead(A0);
    delay(100);  
  }
  float avgsensor = total/ (float)numSamples;
  float voltage = 5.00 * avgsensor / 1023.00;
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temperature : ");
  lcd.setCursor(0,1);
  float temp=666.35685-699.79083*voltage+172.3304997*voltage*voltage;
  lcd.print(temp, 1);
  delay(2000);  
}
