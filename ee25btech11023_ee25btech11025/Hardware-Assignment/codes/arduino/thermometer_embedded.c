#ifndef F_CPU
#define F_CPU 16000000UL
#endif

#include <avr/io.h>
#include <util/delay.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LCD_RS_PIN   PB4
#define LCD_EN_PIN   PB3
#define LCD_D4_PIN   PD5
#define LCD_D5_PIN   PD4
#define LCD_D6_PIN   PD3
#define LCD_D7_PIN   PD2

void lcd_send(uint8_t value, uint8_t is_data) {
    if (is_data) PORTB |= (1 << LCD_RS_PIN);
    else PORTB &= ~(1 << LCD_RS_PIN);

    uint8_t pins[] = {LCD_D4_PIN, LCD_D5_PIN, LCD_D6_PIN, LCD_D7_PIN};

    for (uint8_t i = 0; i < 4; i++) {
        if ((value >> (i + 4)) & 1) PORTD |= (1 << pins[i]);
        else PORTD &= ~(1 << pins[i]);
    }
    PORTB |= (1 << LCD_EN_PIN); _delay_us(1); PORTB &= ~(1 << LCD_EN_PIN); _delay_us(200);

    for (uint8_t i = 0; i < 4; i++) {
        if ((value >> i) & 1) PORTD |= (1 << pins[i]);
        else PORTD &= ~(1 << pins[i]);
    }
    PORTB |= (1 << LCD_EN_PIN); _delay_us(1); PORTB &= ~(1 << LCD_EN_PIN); _delay_us(200);
}

void lcd_print(const char *str) {
    while (*str) {
        lcd_send(*str++, 1);
    }
}

int main(void) {
    DDRB |= (1 << LCD_RS_PIN) | (1 << LCD_EN_PIN);
    DDRD |= (1 << LCD_D4_PIN) | (1 << LCD_D5_PIN) | (1 << LCD_D6_PIN) | (1 << LCD_D7_PIN);
    ADMUX = (1 << REFS0);
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);

    _delay_ms(50);
    lcd_send(0x33, 0);
    lcd_send(0x32, 0);
    lcd_send(0x28, 0);
    lcd_send(0x0C, 0);
    lcd_send(0x06, 0);
    lcd_send(0x01, 0);
    _delay_ms(2);

    lcd_print("Digital");
    lcd_send(0xC0, 0);
    lcd_print("Thermometer");
    _delay_ms(1500);

    lcd_send(0x01, 0);
    _delay_ms(2);
    lcd_print("Temperature:");

    while (1) {
        ADCSRA |= (1 << ADSC);
        while (ADCSRA & (1 << ADSC));
        double temp = -876.97*pow(((5.0*ADC)/1023.0),3) + 6107.11*pow(((5.0*ADC)/1023.0),2) - 13897.54*((5.0*ADC)/1023.0) + 10390.95;

        char buffer[16];
        long int_part = (long)temp;
        long frac_part = labs((long)((temp - int_part) * 100));
        sprintf(buffer, "%ld.%02ld%cC", int_part, frac_part, 0xDF);

        lcd_send(0xC0, 0);
        lcd_print(buffer);
        _delay_ms(1000);
        lcd_send(0xC0, 0);
        lcd_print("                ");
        _delay_ms(500);
    }
}
