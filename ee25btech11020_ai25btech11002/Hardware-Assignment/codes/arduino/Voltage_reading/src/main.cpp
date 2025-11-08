#include "Arduino.h"
#include <LiquidCrystal.h>
int analogPin=A0;
LiquidCrystal lcd(7,6,5,4,3,2);
void setup() {
	  lcd.begin(16,2);
	    Serial.begin(9600);
}

void loop() {
	   double r=analogRead(analogPin);
	   double voltage=(5.0*r/1023.0);
	   Serial.println(voltage,6);
	   delay(1000);
	   lcd.clear();
	   lcd.setCursor(0,0);
	   lcd.print(voltage,3);
	   }
