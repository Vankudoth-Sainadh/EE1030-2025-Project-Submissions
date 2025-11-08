#include <Arduino.h>
#include <LiquidCrystal.h>
LiquidCrystal lcd(2, 3, 8, 9, 10, 11);

int lastUpdate = 0;

void setup() {
  lcd.begin(16, 2);
  lcd.print("Temperature:");
  Serial.begin(9600);
  // analogReference(INTERNAL);
}

void loop() {
  int time = millis();
  int sensorValue = analogRead(A0);
double V = (5.0*sensorValue/1023.0);
double temp=(65.976142 + 10.543790 * ((V - 2.768667) / 0.073200) + 10.482903 * ((V*V - 7.670873) / 0.405495));
  if ((time  - lastUpdate) >= 100)
  {
    lcd.setCursor(0, 1);
    //lcd.print("Temperature:");
    lcd.print(temp);
   // lcd.setCursor(0,1);
   // lcd.print(V);
    Serial.print(temp);
    Serial.print(",");
    Serial.println(V);
    lastUpdate = time;
  }
}
