#include <avr/io.h>
#include <stdint.h>
#include <util/delay.h>
#include <math.h>

#define BAUD 9600L

void initialising_ADC() {
    ADMUX |= 1 << REFS0;
    ADCSRA |= 1 << ADPS0;
    ADCSRA |= 1 << ADPS1;
    ADCSRA |= 1 << ADPS2;
    ADCSRA |= 1 << ADEN;
}

void initialising_UART() {
    uint16_t ubbr = (uint16_t)(F_CPU/(16L*BAUD))-1;
    UBRR0L = (uint8_t)(ubbr);
    UBRR0H = (uint8_t)(ubbr >> 8);
    UCSR0B = (1 << TXEN0);
    UCSR0C = (1 << UCSZ00) | (1 << UCSZ01);
}

void send_byte(uint8_t byte) {
    while (!(UCSR0A & (1 << UDRE0)));
    UDR0 = byte;
}

double read_voltage() {
    ADCSRA |= 1 << ADSC;
    while (ADCSRA & (1 << ADSC));
    int frac = ADCL;
    frac += (ADCH << 8);
    double ans = 5*(frac/1023.0);
    return ans;
}

void print_to_serial(double ans) {
    ans = ans*10000;
    ans = round(ans);
    ans = ans/10000;
    for (int i=0; i<6; i++) {
        uint8_t byte = (int)ans%10;
        send_byte('0' + byte);
        if (i==0) send_byte('.');
        ans *= 10;
    }
    send_byte('\n');
    _delay_ms(1000);
} 

int main() {
    initialising_ADC();
    initialising_UART();
    while (1) {
        double ans = read_voltage();
        print_to_serial(ans);
    }
    return 0;
}