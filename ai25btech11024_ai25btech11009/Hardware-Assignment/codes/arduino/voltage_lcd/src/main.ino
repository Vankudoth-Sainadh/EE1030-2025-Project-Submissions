#include <LiquidCrystal.h>
#include <math.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  analogReference(DEFAULT);
  lcd.begin(16, 2);
  lcd.print("Voltage Meter");
  delay(1500);
  lcd.clear();
}

void loop() {
  int raw = analogRead(A0);
  float volts = raw * (5.0 / 1023.0);

  double a0 = -3730.0634686797857;
  double a1 = 3618.449107721448;
  double a2 = -857.2127421665937;

  double T = 0;

  T = a0 + a1*volts + a2*volts*volts;

  lcd.setCursor(0, 0);
  lcd.print("Voltage: ");
  lcd.print(volts, 4);
  lcd.print("V  ");

  lcd.setCursor(0,1);
  lcd.print("T: ");
  lcd.print(T,2);
  lcd.print("C ");

  delay(1000);
}
