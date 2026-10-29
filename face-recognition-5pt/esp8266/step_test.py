# step_test.py - spin the 28BYJ-48 back and forth, no WiFi/MQTT involved.
#
# Run:  mpremote run esp8266/step_test.py     (or paste into Thonny and press Run)
#
# Expected: the needle turns smoothly about a quarter turn one way, pauses,
# then turns back. If it only buzzes/jitters, the IN1..IN4 pin order is wrong;
# if nothing moves, check the G and 5V wires and that IN3/IN4 are connected.

import time
from machine import Pin

PIN_NUMBERS = (5, 4, 14, 12)   # IN1, IN2, IN3, IN4  (D1, D2, D5, D6) - keep equal to stepper_main.py
STEPS = 1024                   # half-steps = quarter turn of the shaft
DELAY_US = 1500

SEQ = (
    (1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0),
    (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 1),
)
pins = [Pin(n, Pin.OUT, value=0) for n in PIN_NUMBERS]
phase = 0


def move(steps):
    global phase
    d = 1 if steps > 0 else -1
    for _ in range(abs(steps)):
        phase = (phase + d) % 8
        for p, v in zip(pins, SEQ[phase]):
            p.value(v)
        time.sleep_us(DELAY_US)
    for p in pins:
        p.value(0)


while True:
    print("forward")
    move(STEPS)
    time.sleep(1)
    print("back")
    move(-STEPS)
    time.sleep(1)
