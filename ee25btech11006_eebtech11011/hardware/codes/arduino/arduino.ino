#include <Arduino.h>
/* 
  Quadratic calibration + Vcc measurement + LCD display
  (Updated userOffset to correct measured bias)
*/

#include <LiquidCrystal.h>

// LCD pins (change to match your wiring)
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int sensorPin = A0;   // analog pin where sensor/voltage is connected
const int samples = 10;     // number of samples to average

// Quadratic coefficients from your dataset
const double Aq = 1632.33387192;
const double Bq = -3742.80067555;
const double Cq = 2188.70207452;

// USER CALIBRATION: set these to correct any constant/scale error
const double userOffset = -1.0;   // <-- updated: actual (39) - lcd (49.2) = -10.2
const double userScale  = 1.0;     // leave 1.0 unless slope correction required

// readVcc() returns Vcc in millivolts
long readVcc() {
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  delay(2);
  ADCSRA |= _BV(ADSC);
  while (bit_is_set(ADCSRA, ADSC));
  uint16_t result = ADC;
  long vccmV = (1100L * 1023L) / result;
  return vccmV;
}

void setup() {
  lcd.begin(16, 2);
  lcd.print("Calibrating...");
  delay(800);
  lcd.clear();
}

void loop() {
  long vcc_mV = readVcc();
  double Vcc = vcc_mV / 1000.0;

  long acc = 0;
  for (int i = 0; i < samples; ++i) {
    acc += analogRead(sensorPin);
    delay(10);
  }
  double raw = acc / (double)samples;

  double voltage = (raw / 1023.0) * Vcc; // volts

  // Quadratic mapping
  double temperature = Aq * voltage * voltage + Bq * voltage + Cq;

  // Apply user calibration
  temperature = temperature * userScale + userOffset;

  lcd.setCursor(0, 0);
  lcd.print("V: ");
  lcd.print(voltage, 3);
  lcd.print("V ");

  lcd.setCursor(0, 1);
  lcd.print("T: ");
  lcd.print(temperature, 2);
  lcd.print((char)223);
  lcd.print("C   ");

  delay(800);
}