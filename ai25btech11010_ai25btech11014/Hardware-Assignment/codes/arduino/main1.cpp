
#include <LiquidCrystal.h>


LiquidCrystal lcd(12, 11, 5, 4, 3, 2);


const int analogPin = A2;


const float VREF = 5.0;

void setup() {
  lcd.begin(16, 2);             
  lcd.print("Voltage at A2:");
  delay(1000);
}

void loop() {
  int adcValue = analogRead(analogPin);      
  float voltage = (adcValue * VREF) / 1023.0; 

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("A2 Voltage (V):");

  lcd.setCursor(0, 1);
  lcd.print(voltage, 4);  

  delay(500); 
}

