# stepper_main.py - ESP8266 (MicroPython) needle controller for a 28BYJ-48 + ULN2003.
#
# Flash/copy to the board AS main.py (this does not touch src/main.py, the old servo firmware).
#
# Wiring (ULN2003 board):
#   IN1 <- D1 (GPIO5)    IN2 <- D2 (GPIO4)
#   IN3 <- D5 (GPIO14)   IN4 <- D6 (GPIO12)
#   "-" <- G             "+" <- 5V
#
# MQTT:
#   subscribes  face_tracking          commands: ANGLE:<0-180>  STOP  HOME
#                                                JOG:<+/-steps>  (calibration, see below)
#   publishes   face_tracking/status   JSON {"status":..., "angle":...}
#
# Calibration: the 28BYJ-48 has no position sensor. At power-up the firmware ASSUMES the needle
# is at START_ANGLE (centre of the dial). Place it there by hand before powering on, or send
# JOG:+100 / JOG:-100 to nudge the needle to the centre (the believed angle is left unchanged).

import time
import network
import machine
import ubinascii
import ujson as json
from machine import Pin
from umqtt.simple import MQTTClient

# ---------------- CONFIG ----------------
WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASS = "YOUR_WIFI_PASSWORD"

MQTT_HOST = "192.168.169.47"   # the PC running Mosquitto (its WiFi IP)
MQTT_PORT = 1884

TOPIC_CMD = b"face_tracking"
TOPIC_STATUS = b"face_tracking/status"

PIN_NUMBERS = (5, 4, 14, 12)   # IN1, IN2, IN3, IN4  (D1, D2, D5, D6)

STEPS_PER_REV = 4096           # 28BYJ-48, half-step mode
SWEEP_DEG = 180                # needle sweep covered by ANGLE:0..180
START_ANGLE = 90               # assumed needle angle at power-up
STEP_DELAY_US = 1500           # lower = faster; below ~1000 the motor stalls
REVERSE = False                # set True if the needle moves the wrong way

# ---------------- STEPPER ----------------
SEQ = (
    (1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0),
    (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 1),
)
pins = [Pin(n, Pin.OUT, value=0) for n in PIN_NUMBERS]

STEPS_FULL_SWEEP = STEPS_PER_REV * SWEEP_DEG // 360
phase = 0
current_step = STEPS_FULL_SWEEP * START_ANGLE // 180
target_step = current_step


def angle_to_step(angle):
    angle = max(0, min(180, int(angle)))
    return STEPS_FULL_SWEEP * angle // 180


def step_to_angle(step):
    return step * 180 // STEPS_FULL_SWEEP


def coils_off():
    for p in pins:
        p.value(0)


def one_step(direction):
    """direction +1 / -1 -> energise next coil pattern."""
    global phase
    if REVERSE:
        direction = -direction
    phase = (phase + direction) % 8
    pattern = SEQ[phase]
    for p, v in zip(pins, pattern):
        p.value(v)


# ---------------- WIFI / MQTT ----------------
client = None


def publish_status(status):
    if client is None:
        return
    try:
        client.publish(TOPIC_STATUS, json.dumps({"status": status, "angle": step_to_angle(current_step)}))
    except Exception as e:
        print("status publish failed:", e)


def wifi_connect():
    """Join WiFi, retrying forever (the link on this network is intermittent)."""
    sta = network.WLAN(network.STA_IF)
    sta.active(True)
    sta.config(pm=network.WLAN.PM_NONE)   # no modem sleep: steadier link for MQTT
    while not sta.isconnected():
        print("Connecting to WiFi:", WIFI_SSID)
        try:
            sta.disconnect()
        except Exception:
            pass
        sta.connect(WIFI_SSID, WIFI_PASS)
        start = time.ticks_ms()
        while not sta.isconnected() and time.ticks_diff(time.ticks_ms(), start) < 20000:
            time.sleep(0.3)
        if not sta.isconnected():
            print("WiFi attempt failed, retrying (status %s)" % sta.status())
            time.sleep(2)
    print("WiFi connected:", sta.ifconfig())


def on_message(topic, msg):
    global target_step, current_step
    try:
        cmd = msg.decode().strip().upper()
        print("cmd:", cmd)
        if cmd.startswith("ANGLE:"):
            target_step = angle_to_step(int(float(cmd.split(":", 1)[1])))
        elif cmd == "STOP":
            target_step = current_step
            coils_off()
            publish_status("STOPPED")
        elif cmd == "HOME":
            target_step = angle_to_step(START_ANGLE)
        elif cmd.startswith("JOG:"):
            n = int(cmd.split(":", 1)[1])
            d = 1 if n > 0 else -1
            for _ in range(abs(n)):
                one_step(d)
                time.sleep_us(STEP_DELAY_US)
            coils_off()   # position belief intentionally unchanged (calibration)
        else:
            print("unknown command:", cmd)
    except Exception as e:
        print("command error:", e)


def mqtt_connect():
    cid = b"needle_" + ubinascii.hexlify(machine.unique_id())
    c = MQTTClient(cid, MQTT_HOST, port=MQTT_PORT, keepalive=60)
    c.set_callback(on_message)
    c.connect()
    c.subscribe(TOPIC_CMD)
    print("MQTT connected, subscribed to", TOPIC_CMD)
    return c


# ---------------- MAIN ----------------
print("NEEDLE CONTROLLER (28BYJ-48)")
wifi_connect()
while client is None:
    try:
        client = mqtt_connect()
    except Exception as e:
        print("MQTT connect failed, retrying:", e)
        time.sleep(3)
publish_status("READY")


def reconnect():
    """Bring WiFi and MQTT back; loops until both are up."""
    global client
    coils_off()
    while True:
        try:
            wifi_connect()
            client = mqtt_connect()
            publish_status("RECONNECTED")
            return
        except Exception as e:
            print("reconnect failed:", e)
            time.sleep(3)


# umqtt.simple never pings by itself; without this the broker drops us after ~1.5x keepalive.
PING_EVERY_MS = 10000
last_ping = time.ticks_ms()
counter = 0
while True:
    try:
        if time.ticks_diff(time.ticks_ms(), last_ping) > PING_EVERY_MS:
            if not network.WLAN(network.STA_IF).isconnected():
                raise OSError("WiFi link lost")
            client.ping()
            last_ping = time.ticks_ms()

        if current_step != target_step:
            d = 1 if target_step > current_step else -1
            one_step(d)
            current_step += d
            time.sleep_us(STEP_DELAY_US)
            counter += 1
            if counter >= 20:          # keep MQTT responsive while stepping
                counter = 0
                client.check_msg()
            if current_step == target_step:
                coils_off()
                publish_status("ANGLE_REACHED")
        else:
            client.check_msg()
            time.sleep_ms(20)
    except Exception as e:
        print("loop error:", e)
        reconnect()
        last_ping = time.ticks_ms()
