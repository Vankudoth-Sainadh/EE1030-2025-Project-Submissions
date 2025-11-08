#include <LiquidCrystal.h>

LiquidCrystal lcd(7, 8, 9, 10, 11, 12);

float aw  = 156.71273;
float bw =  197.683959;
float cw = -11.1347464 ;

const int pinVout = A0;
const float VREF = 5.0;
const int pinVout2 = A1;

void setup() {
  lcd.begin(16, 2);
  analogReference(DEFAULT);
  lcd.clear();
  lcd.print("PT100 Voltage");
  delay(1500);
}

void loop() {
  float v1 = readVoltage(pinVout);
  float v2 = readVoltage(pinVout2);

  float diff = v1 -v2 ;
  float tempw = (aw*diff*diff)+(bw*diff)+cw ;
  lcd.clear();
  lcd.print("Temp : ");
  lcd.setCursor(8, 0);
  lcd.print(tempw, 3);
  lcd.print("C");
  lcd.setCursor(0, 1);
  lcd.print(diff, 4);
  delay(500);
}

float readVoltage(int pin) {
  const int samples = 50;
  long sum = 0;
  
  for(int i = 0; i < samples; i++) {
    sum += analogRead(pin);
    delay(2);
  }

  float avgADC = float(sum) / samples;
  return avgADC * (VREF / 1023.0);
}
