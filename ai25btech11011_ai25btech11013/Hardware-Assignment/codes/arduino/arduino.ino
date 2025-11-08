#include <LiquidCrystal.h>
#include <math.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
const double c = 243.5372;
const double b = -1579.0764;
const double a =2561.6935 ;
const int sensorPin = A0;

void setup() {
  lcd.begin(16, 2);
  Serial.begin(9600); // <-- ADD THIS LINE
}

void loop() {
  int sensorVal = analogRead(sensorPin);
  double voltage = (sensorVal / 1023.0) * 5.0;

  Serial.println(voltage); // <-- ADD THIS LINE

  double temperature = a + b*(voltage) + c*(voltage*voltage);

 

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Voltage: ");
  lcd.print(voltage,4);
  lcd.setCursor(0, 1);
  lcd.print("Temp: ");
  lcd.print(temperature,2);
  lcd.print(" C");

  delay(1000);
}
