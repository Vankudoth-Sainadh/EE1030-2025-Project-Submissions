#include "Arduino.h"
#include <LiquidCrystal.h>

 
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

void setup() { 
   lcd.begin(16, 2);
   lcd.clear();
  Serial.begin(9600);
  lcd.print("Digital");
  lcd.setCursor(0,1);
  lcd.print("Thermometer");
  delay(1500);
  lcd.clear();
  lcd.print("Temperature:");
}


void loop() { 
  int sensorValue = analogRead(A0);
  double voltage = (5.0*sensorValue/1023.0);
   double   temp = 100.73826784  - 548.01601557 * voltage + 607.7593742 * pow(voltage, 2); 

  Serial.println(voltage,2);
  delay(500);
  lcd.setCursor(0,1);
  lcd.print(temp,2);
  lcd.write(0xDF);  
  lcd.print("C");
  delay(1000);
  lcd.setCursor(0, 1);  
  for (int i = 0; i < 16; i++) {  
    lcd.print(" ");  
  }
}
