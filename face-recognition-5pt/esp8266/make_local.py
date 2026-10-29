# Builds esp8266/main_local.py (git-ignored) = stepper_main.py + real WiFi credentials + broker address.
# Usage: py esp8266/make_local.py <ssid> <password> [broker_ip]
import sys
ssid, pw = sys.argv[1], sys.argv[2]
s = open("esp8266/stepper_main.py", encoding="utf-8").read()
s = s.replace('"YOUR_WIFI_SSID"', repr(ssid)).replace('"YOUR_WIFI_PASSWORD"', repr(pw))
if len(sys.argv) > 3:
    s = s.replace('MQTT_HOST = "192.168.169.47"', 'MQTT_HOST = %r' % sys.argv[3])
open("esp8266/main_local.py", "w", encoding="utf-8").write(s)
print("wrote esp8266/main_local.py")
