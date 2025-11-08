#include <LiquidCrystal.h>
LiquidCrystal lcd(12,11,5,4,3,2);
int status = 1 ;
byte customChar[8] = {
	0b00000,
	0b00111,
	0b00101,
	0b00111,
	0b00000,
	0b00000,
	0b00000,
	0b00000
};
void setup() {
    lcd.begin(16,2);
    lcd.clear();
    Serial.begin(9600);
    pinMode(13,OUTPUT);
    pinMode(7,INPUT_PULLUP);
    digitalWrite(13,HIGH);
    lcd.createChar(0, customChar);
}

void loop() {
    float volt , a0,a1,a2 , temp , tempf ; 
    
    volt = (5.0*analogRead(A1)/1023);
    
    a0 = -367.48044857;
    a1 = 37.67235012;
    a2 = 54.77146835;
    
    int val = digitalRead(7);
    if(val == 0)
        if(status == 0 )
          status = 1 ;             
        else if(status == 1)
          status++;
        else
          status = 0 ;        

    temp = a0+a1*volt+a2*volt*volt;
    tempf = temp *1.8 + 32.0 ; 
    switch(status) {
      	case 1 :
         lcd.setCursor(0,0);
         lcd.print("Temperature ");
         lcd.write((byte)0);
      	 lcd.print("C");
         lcd.setCursor(0,1);
         lcd.print(temp);
         Serial.println(temp);
  		 break;
  		case 2:
         lcd.setCursor(0,0);
         lcd.print("Temperature ");
         lcd.write((byte)0);
      	 lcd.print("F");
         lcd.setCursor(0,1);
         lcd.print(tempf);
         Serial.println(tempf);
  		 break;
  		default :
        lcd.setCursor(0,0);
        lcd.print("Voltage Reading");
        lcd.setCursor(0,1);
        lcd.print(volt);
        Serial.println(volt);
    }
    delay(200);
	lcd.clear();
}