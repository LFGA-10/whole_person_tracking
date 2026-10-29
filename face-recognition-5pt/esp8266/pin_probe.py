# pin_probe.py - find which ESP8266 pins are wired to the ULN2003 inputs.
#
# Run:  mpremote run esp8266/pin_probe.py      (or paste into Thonny and press Run)
#
# Each pin is set HIGH for 3 seconds, one at a time, and its name is printed.
# Watch the 4 LEDs on the green ULN2003 board (A=IN1, B=IN2, C=IN3, D=IN4):
# write down which LED lights up while each pin name is printed.
# The motor may twitch a little - that is fine.

import time
from machine import Pin

PINS = (
    ("D0", 16), ("D1", 5), ("D2", 4), ("D3", 0), ("D4", 2),
    ("D5", 14), ("D6", 12), ("D7", 13), ("D8", 15),
)

while True:
    for name, gpio in PINS:
        p = Pin(gpio, Pin.OUT, value=0)
        print("HIGH on", name, "(GPIO%d) - which LED (A/B/C/D) is lit?" % gpio)
        p.value(1)
        time.sleep(3)
        p.value(0)
        time.sleep_ms(300)
    print("---- round done, repeating ----")
