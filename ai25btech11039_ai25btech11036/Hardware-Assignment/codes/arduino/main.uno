const int PT100_PIN = A0;
const float VCC = 5.0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  int adcValue = analogRead(PT100_PIN);
  float voltage = (adcValue * VCC) / 1023.0;
  Serial.print("Voltage: ");
  Serial.print(voltage, 3);
  Serial.println(" V");
  delay(1000);
}
