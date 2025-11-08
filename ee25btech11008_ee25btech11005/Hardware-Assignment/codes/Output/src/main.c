#include <avr/io.h>
#include <util/delay.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RS 4
#define EN 3

#define D4 5
#define D5 4
#define D6 3
#define D7 2

double read_voltage() {
    ADCSRA |= 1 << ADSC;
    while (ADCSRA & (1 << ADSC));
    int frac = ADCL;
    frac += (ADCH << 8);
    double ans = 5*(frac/1023.0);
    return ans;
}

void initialising_ADC() {
    ADMUX |= 1 << REFS0;
    ADCSRA |= 1 << ADPS0;
    ADCSRA |= 1 << ADPS1;
    ADCSRA |= 1 << ADPS2;
    ADCSRA |= 1 << ADEN;
}

float read_EEPROM(uint8_t addr) {
    float data;
    uint8_t* ptr = (uint8_t*) &data;
    for (int i=0; i<4; i++) {
        EEAR = addr+i;
        EECR |= 1 << EERE;
        ptr[i] = EEDR;
    }
    return data;
}

float find_temperature(float a, float b, float c, float v) {
    float temp = (v-a)/b;
    temp = round(temp*100)/100;
    return temp;
}

void send_nibble(uint8_t nibble) {
    if (nibble & 1) {
        PORTD |= 1 << D4;
    } else {
        PORTD &= ~(1 << D4);
    }

    if (nibble & 2) {
        PORTD |= 1 << D5;
    } else {
        PORTD &= ~(1 << D5);
    }

    if (nibble & 4) {
        PORTD |= 1 << D6;
    } else {
        PORTD &= ~(1 << D6);
    }

    if (nibble & 8) {
        PORTD |= 1 << D7;
    } else {
        PORTD &= ~(1 << D7);
    }

    PORTB |= 1 << EN;
    _delay_us(100);
    PORTB &= ~(1 << EN);
    _delay_us(100);
}

void send_cmd(uint8_t cmd) {
    PORTB &=  ~(1 << RS);
    send_nibble(cmd >> 4);
    send_nibble(cmd);
    _delay_ms(2);
}

void send_data(uint8_t data) {
    PORTB |= 1 << RS;
    send_nibble(data >> 4);
    send_nibble(data);
    _delay_ms(2);
}

void initialising_LCD() {
    DDRB |= (1 << RS) | (1 << EN);
    DDRD |= (1 << D4) | (1 << D5) | (1 << D6) | (1 << D7);

    _delay_ms(20);
    send_nibble(0x03);
    _delay_ms(5);
    send_nibble(0x03);
    _delay_us(200);
    send_nibble(0x03);
    _delay_us(200);
    send_nibble(0x02);

    send_cmd(0x28);
    send_cmd(0x0C);
    send_cmd(0x01);
    send_cmd(0x06);
}

void set_loc(int x, int y) {
    uint8_t addr = (uint8_t)(64*x+y);
    send_cmd(addr | 128);
}

void print_to_LCD(char *str) {
    int l = strlen(str);
    for (int i=0; i<l; i++) {
        send_data(str[i]);
    }
}

char* num_to_str(double num, int k) {
    long temp = round(num*pow(10, k));
    long n1 = temp;
    int l = 1;
    while (n1>0) {
        n1 = n1/10;
        l++;
    }
    char* str = (char*) malloc((l+1)*sizeof(char));
    str[l] = '\0';
    for (int i=0; i<k; i++) {
        str[l-i-1] = (char) ((temp%10)+48);
        temp = temp/10;
    }
    str[l-k-1] = '.';
    for (int i=l-k-2; i>-1; i--) {
        str[i] = (char) ((temp%10)+48);
        temp = temp/10;
    }
    return str;
}

int main() {
    initialising_ADC();
    initialising_LCD();
    float a = read_EEPROM(0);
    float b = read_EEPROM(4);
    float c = read_EEPROM(8);
    set_loc(0, 0);
    print_to_LCD("VOLT: ");
    set_loc(1, 0);
    print_to_LCD("TEMP: ");
    while (1) {
        _delay_ms(1000);
        double volt = read_voltage();
        char* strV = num_to_str(volt, 5);
        double temp = find_temperature(a, b, c, volt);
        char* strT = num_to_str(temp, 2);
        set_loc(0, 6);
        print_to_LCD("            ");
        set_loc(0, 6);
        print_to_LCD(strV);
        print_to_LCD("V");
        free(strV);
        set_loc(1, 6);
        print_to_LCD("            ");
        set_loc(1, 6);
        print_to_LCD(strT);
        send_data(223);
        print_to_LCD("C");
        free(strT);
    }
    return 0;
}