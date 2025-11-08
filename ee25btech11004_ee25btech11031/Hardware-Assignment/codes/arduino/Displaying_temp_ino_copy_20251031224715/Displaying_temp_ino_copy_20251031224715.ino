#include "Arduino.h"
#include <LiquidCrystal.h>

// initialize the library by associating any needed LCD interface pin
// with the arduino pin number it is connected to
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

const int analogInPin = A5;

void setup() {
  // set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
  // Print a message to the LCD.
  
}


void loop() 
{
  
 int sensorValue = analogRead(analogInPin);
  double voltage = (5.0*sensorValue/1023);
  lcd.setCursor(0, 0);
  lcd.print("Voltage:"); lcd.print(voltage);
	double temperature = -2228.38488041 + 1382.40759879*voltage -199.744928*voltage*voltage;
	lcd.setCursor(0,1);
  lcd.print("Temp:");  lcd.print(temperature);
  delay(2000); // interval of fetching data from sensor
}