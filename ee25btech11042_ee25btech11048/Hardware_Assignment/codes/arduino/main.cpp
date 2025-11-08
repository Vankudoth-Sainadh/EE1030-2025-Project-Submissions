#include <LiquidCrystal.h> 
const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

unsigned long previousMillis = 0;      
const long interval = 500;         


const double VOLTAGE_CONVERSION = 5.0 / 1023.0;


int lastSensorValue = -1; 




void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);
  lcd.print("Voltage: "); 
}

void loop() {

  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    int sensorValue = analogRead(A1);
    if (sensorValue != lastSensorValue) {
      lastSensorValue = sensorValue;
      double voltage = sensorValue * VOLTAGE_CONVERSION;
      lcd.setCursor(9, 0);
      lcd.print(voltage, 3); 
      lcd.print("  "); 
      Serial.println(voltage);
    }
  }
}
