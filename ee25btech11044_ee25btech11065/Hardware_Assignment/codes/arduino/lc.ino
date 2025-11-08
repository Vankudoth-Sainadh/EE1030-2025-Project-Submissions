#include <Arduino.h>

#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

const int sensorPin = A0;   
const int samples = 8;     

const double Aq = 418.8699;
const double Bq = -827.0696;
const double Cq = 457.6605;

const double userOffset = -1.0;   
const double userScale  = 1.0;     
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

  double voltage = (raw / 1023.0) * Vcc; 

  double temperature = Aq * voltage * voltage + Bq * voltage + Cq;

  temperature = temperature * userScale + userOffset;

  lcd.setCursor(0, 0);
  lcd.print("V: ");
  lcd.print(voltage, 3);
  lcd.print("V ");

  lcd.setCursor(0, 1);
  lcd.print("T: ");
  lcd.print(temperature, 1);
  lcd.print((char)223);
  lcd.print("C   ");

  delay(800);
}