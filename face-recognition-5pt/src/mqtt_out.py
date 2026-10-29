# src/mqtt_out.py
"""
Publishes face-tracking output to the MQTT broker so the ESP8266 can move the needle.

    py -m src.mqtt_out            # test: sweeps the needle 90 -> 0 -> 180 -> 90

Topic: face_tracking   (payloads: ANGLE:<0-180>, STOP, HOME)
"""

from __future__ import annotations

import os
import time

import paho.mqtt.client as mqtt

BROKER_HOST = os.environ.get("FACE_MQTT_HOST", "localhost")
BROKER_PORT = int(os.environ.get("FACE_MQTT_PORT", "1884"))
TOPIC = "face_tracking"
TOPIC_STATUS = "face_tracking/status"


class NeedlePublisher:
    def __init__(
        self,
        host: str = BROKER_HOST,
        port: int = BROKER_PORT,
        min_interval: float = 0.1,   # max ~10 publishes/s
        min_delta: int = 2,          # ignore angle changes smaller than this (deg)
        invert: bool = False,
        mode: str = "center",
        dead_zone: float = 0.07,     # normalised; same as the tracker's dead zone
        gain: float = 12.0,          # degrees per tick at full-scale error (center mode)
    ):
        self.min_interval = min_interval
        self.min_delta = min_delta
        self.invert = invert
        self.mode = mode
        self.dead_zone = dead_zone
        self.gain = gain
        self.angle = 90.0
        self.last_angle: int | None = None
        self.last_time = 0.0
        self.stopped = False

        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_message = lambda c, u, m: print("[ESP]", m.payload.decode(errors="replace"))
        self.client.on_connect = lambda c, u, f, rc, p: c.subscribe(TOPIC_STATUS)
        self.client.connect(host, port, keepalive=30)
        self.client.loop_start()
        print(f"[MQTT] connected to {host}:{port}, publishing on '{TOPIC}'")

    def send_error_x(self, error_x: float) -> None:
        """error_x: -1 (face at left edge) .. +1 (right edge).

        mode "center": closed loop - nudge the angle toward the face each tick and hold when the
        face is inside the dead zone, so the face ends up centred in the image.
        mode "absolute": map the face position straight to a needle angle (needle points at face).
        """
        if self.invert:
            error_x = -error_x
        if self.mode == "absolute":
            self.send_angle(int(round(90 + max(-1.0, min(1.0, error_x)) * 90)))
            return

        now = time.time()
        if now - self.last_time < self.min_interval:
            return
        if abs(error_x) <= self.dead_zone:
            return  # centred: hold position, send nothing
        self.angle = max(0.0, min(180.0, self.angle + self.gain * error_x))
        self.send_angle(int(round(self.angle)), force=True)

    def send_angle(self, angle: int, force: bool = False) -> None:
        now = time.time()
        if now - self.last_time < self.min_interval:
            return
        if not force and self.last_angle is not None and abs(angle - self.last_angle) < self.min_delta:
            return
        self.client.publish(TOPIC, f"ANGLE:{angle}")
        self.last_angle, self.last_time, self.stopped = angle, now, False

    def send_stop(self) -> None:
        """Face lost: hold the needle where it is (sent once until motion resumes)."""
        if not self.stopped:
            self.client.publish(TOPIC, "STOP")
            self.stopped = True
            self.last_angle = None

    def close(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    pub = NeedlePublisher(min_interval=0, min_delta=0)
    for a in (90, 0, 180, 90):
        print("->", a)
        pub.send_angle(a)
        time.sleep(6)
    pub.close()
