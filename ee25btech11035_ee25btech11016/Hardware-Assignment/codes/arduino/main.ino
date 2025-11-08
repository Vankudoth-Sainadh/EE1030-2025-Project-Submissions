#include <LiquidCrystal.h>
#include <math.h>

// This code is for getting the voltage input through arduino and thus substituting that value in the T(V) equation to get the value of temperature.
// This value of temperature is then printed on to the LCD.

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

// The co-efficients of the T(V) expression such that T(V) = a + bV + cV^2
const double c = 110.04876635;
const double b = -239.19052507;
const double a = 29.08430195;

void setup() {
  lcd.begin(16, 2);
  Serial.begin(9600);
}

void loop() {
  // Get the voltage value through the arduino and store it in the variable named "voltage"
  int sensorVal = analogRead(A0);
  double voltage = (sensorVal / 1023.0) * 5.0;

  Serial.println(voltage); // prints voltage value to serial monitor

  double temperature = a + b*(voltage) + c*(voltage*voltage);

  // Code for printing the temperature value onto the LCD

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temp: ");
  lcd.setCursor(0, 1);
  lcd.print(temperature);
  lcd.print(" C");

  delay(1000); // Wait 1 second before printing next value
}