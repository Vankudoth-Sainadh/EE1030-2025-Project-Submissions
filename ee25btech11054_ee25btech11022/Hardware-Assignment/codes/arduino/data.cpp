#include <LiquidCrystal.h>

 
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

void setup() { 
   lcd.begin(16, 2);
   lcd.clear();
  Serial.begin(9600);
   
}


void loop() { 
  int sensorValue = analogRead(A0);
  double voltage = (5.0*sensorValue/1023.0);
   double temp=301.1599*pow(voltage,2)-972.1064*(voltage)+775.1808;

  Serial.println(voltage,4);
lcd.print(temp,2);
  delay(500);
  lcd.clear();
  delay(500);
}
