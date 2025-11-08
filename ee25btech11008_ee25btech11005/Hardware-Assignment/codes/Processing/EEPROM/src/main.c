#include <avr/io.h>
#include <stdint.h>
#include <math.h>

void write(uint16_t addr, uint8_t data) {
    while (EECR & (1 << EEPE));
    EEAR = addr;                     
    EEDR = data;                     
    EECR |= (1 << EEMPE);             
    EECR |= (1 << EEPE);   
}            
   
void write_float(uint16_t addr, float data) {
    uint8_t* ptr = (uint8_t*) (&data);
    for (int i =0; i<4; i++) {
        write(addr+i, ptr[i]);
    }
}

int main() {
    float a = 2.60773402;
    float b = 4.12641341*pow(10, -3);
    float c = -3.31632888*pow(10, -6);
    write_float(0, a);
    write_float(4, b);
    write_float(8, c);
}



