#include<Arduino.h>
#include<LiquidCrystal.h>
#include<math.h>
int analogPin=A1;
LiquidCrystal lcd(7,6,5,4,3,2);
void setup(){
lcd.begin(16,2);
Serial.begin(9600);
}

void loop() {
  double V=(5.0*analogRead(analogPin))/1023.0;
  double a=-1.85446529e-05;
double b=0.00624739849;
double c= 1.49861405;
double T=(-b+sqrt(b*b-4*a*(c-V)))/2*a;
  Serial.println(V,6);
  lcd.clear();
  lcd.setCursor(0,1);
  lcd.print(T,3);
  delay(1000);
  }
