#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2); // RS, E, D4, D5, D6, D7
Adafruit_ADS1115 ads;

const float a = 0.000017826;
const float b = 0.0014987;
const float c = 0.1570;

const int AVG_SAMPLES = 10;
float moving_avg[AVG_SAMPLES] = {0.0};
int idx = 0;

void setup() {
  Serial.begin(9600);
  ads.begin();
  ads.setGain(GAIN_EIGHT);
  lcd.begin(16, 2);
  lcd.clear();
  lcd.print("PT100 System");
}

void loop() {
  int16_t adc_raw = ads.readADC_Differential_0_1();
  float v_out = adc_raw * 0.015625 / 1000.0;

  if(idx >= AVG_SAMPLES){
    idx = 0;
  }
  moving_avg[idx++] = v_out;
  float avg = 0;
  for(int i = 0; i<AVG_SAMPLES; i++){
    avg += moving_avg[i];
  }
  avg/=AVG_SAMPLES;

  float discriminant = b*b - 4*a*(c-avg);
  float temp = 0;
  if(discriminant > 0){
      temp = (-b+sqrt(discriminant))/(2*a);
  }
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Temperature: ");
  lcd.setCursor(0, 1);
  lcd.print(temp, 2);
  lcd.print((char)223);
  lcd.print("C");

  delay(1000);
}
