#include "Arduino.h"
void setup(){
      serial.begin(9600);
      }
      void loop(){
      int sensorvalue=analogRead(A1);
      double Voltage=(5.0*sensorValue/1023.0);
      serial.println(Voltage);
      delay(1000);
      }
