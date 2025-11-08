#include <LiquidCrystal.h>

// LCD wiring: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int testPin = A1;       // Analog input pin (A1)
const float VREF = 5.0;       // Reference voltage

void setup() {
  Serial.begin(9600);         // Bluetooth serial/USB serial
  lcd.begin(16, 2);           // Initialize LCD (16 columns x 2 rows)
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("A0 Voltage Test");
  delay(1500);                // Show test message for 1.5 seconds
}

void loop() {
  int adcRaw = analogRead(testPin);               // Read ADC value from A1
  float v = (adcRaw * VREF) / 1023.0;       // Convert to voltage
  float a0 = 794.88620258,a1 = -384.63936743,a2 = 26.03817558; 
  float temp = a0 + a1*v + a2*v*v;

  // Print to serial monitor (or Bluetooth terminal)
  Serial.print("Voltage: ");
  Serial.print(v, 4);
  Serial.println(" V");
  Serial.print("Temp: ");
  Serial.print(temp, 4);
  Serial.println(" *C");

  // Print to LCD
  lcd.clear();
  lcd.setCursor(0, 1);
  lcd.print("Temp:");
  lcd.print(temp,4);

  lcd.setCursor(0, 0);
  lcd.print("V:");
  lcd.print(v, 4);
  lcd.print("V");

  delay(1000);                                   // Update every second
}
