#!/usr/bin/env python3
"""Stats screen for the Yahboom CUBE Nano case on the Orin (128x32 SSD1306 OLED).

Rotates through three pages:
  1 System   CPU load + time, RAM, disk, IP
  2 Jetson   GPU load + power mode, CPU/GPU temps, Tj temp + fan
  3 Lab      MQTT broker, Ollama, voice services, time since last MQTT message

Also: turns on the case fan and RGB bar, shifts the picture by a pixel every
few minutes and dims (or switches off) the screen overnight to avoid OLED
burn-in, and flashes a warning if the Orin runs hot.

  python3 orin/oled_stats.py              run (Ctrl+C to stop)
  python3 orin/oled_stats.py --debug      print what it finds
  python3 orin/oled_stats.py clear        blank the screen, RGB and fan off, exit
  python3 orin/oled_stats.py --preview out.png   render each page to PNGs, no hardware

Needs:   pip3 install smbus Adafruit_SSD1306 Pillow
Better:  sudo pip3 install jetson-stats   (GPU load, power mode, fan; reboot after)
For the MQTT "last message" line: pip3 install paho-mqtt, and give the broker
login in MQTT_USER / MQTT_PASS (the systemd unit reads /etc/default/oled-stats).
"""
import argparse
import glob
import os
import shutil
import signal
import socket
import sys
import threading
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 128, 32
LINE_CHARS = 21           # what fits across the screen in the built-in font
OLED_ADDR = 0x3C          # the screen
CASE_ADDR = 0x0E          # the case controller (fan + RGB bar)
REG_RGB_EFFECT = 0x04     # light effect 1-5
REG_RGB_OFF = 0x07        # write 0 to turn the RGB bar off
REG_FAN = 0x08            # 1 = fan on, 0 = fan off
CANDIDATE_BUSES = (7, 1)  # Orin Nano header I2C is usually bus 7; Yahboom's default was 1
BURN_IN_OFFSETS = [(0, 0), (1, 0), (1, 1), (0, 1)]

DEBUG = False


def log(*msg):
    if DEBUG:
        print(*msg, flush=True)


# ---------------------------------------------------------------- system stats

class CpuMeter:
    """CPU load from /proc/stat, measured between successive calls."""

    def __init__(self):
        self.last = self._read()

    @staticmethod
    def _read():
        with open("/proc/stat") as f:
            fields = [int(x) for x in f.readline().split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
        return sum(fields[:8]), idle

    def percent(self):
        total, idle = self._read()
        d_total, d_idle = total - self.last[0], idle - self.last[1]
        self.last = (total, idle)
        return round(100 * (d_total - d_idle) / d_total) if d_total > 0 else 0


def ram():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])  # kB
    total = info["MemTotal"]
    avail = info.get("MemAvailable", info.get("MemFree", 0))
    return round(100 * (total - avail) / total), total / 1048576  # %, GB


def disk():
    du = shutil.disk_usage("/")
    return round(100 * du.used / du.total), du.total / 1e9  # %, GB


def local_ip():
    # Connecting a UDP socket sends nothing; it just asks which address the
    # OS would use to reach the LAN.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.0.1", 9))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


# ---------------------------------------------------------------- Jetson stats

def _read_number(path):
    try:
        with open(path) as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _first_existing(*patterns):
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


class JetsonStats:
    """GPU load, temperatures, fan and power mode.

    Uses jtop (jetson-stats) when it is installed and its service is running,
    because it knows where each JetPack version keeps these. Falls back to
    reading /sys directly, which covers temperatures and usually the fan.
    """

    def __init__(self):
        self.jtop = None
        try:
            from jtop import jtop
            self.jtop = jtop()
            self.jtop.start()
            log("jetson stats: using jtop")
        except Exception as exc:
            self.jtop = None
            log("jetson stats: jtop not available, reading /sys directly:", exc)
        self.zones = {}
        for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                with open(os.path.join(zone, "type")) as f:
                    name = f.read().strip().lower()
            except OSError:
                continue
            self.zones[name] = os.path.join(zone, "temp")
        log("thermal zones:", sorted(self.zones))
        # Found once at start-up (searching /sys every second is slow).
        self.gpu_load_path = _first_existing(
            "/sys/devices/platform/bus@0/17000000.gpu/load",   # Orin, JetPack 6
            "/sys/devices/platform/17000000.ga10b/load",       # Orin, JetPack 5
            "/sys/devices/platform/17000000.gpu/load",
            "/sys/devices/gpu.0/load",                         # older Jetsons
            "/sys/devices/platform/*gpu*/load",
            "/sys/devices/platform/bus@0/*gpu*/load")
        self.fan_path = _first_existing(
            "/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1",
            "/sys/class/hwmon/hwmon*/pwm1")
        log("gpu load file:", self.gpu_load_path, "| fan file:", self.fan_path)

    def close(self):
        if self.jtop is not None:
            try:
                self.jtop.close()
            except Exception:
                pass

    def _jtop_stats(self):
        if self.jtop is None:
            return {}
        try:
            # jtop refreshes .stats in its own thread; just read the latest copy.
            return dict(self.jtop.stats)
        except Exception as exc:
            log("jtop read failed:", exc)
        return {}

    def _zone_temp(self, *names):
        for name in names:
            for zone, path in self.zones.items():
                if zone.startswith(name):
                    v = _read_number(path)
                    if v is not None:
                        return v / 1000.0
        return None

    def read(self):
        js = self._jtop_stats()

        def jtop_value(match):
            for key, value in js.items():
                if match(key.lower()) and isinstance(value, (int, float)):
                    return float(value)
            return None

        out = {
            "gpu": jtop_value(lambda k: k == "gpu"),
            "cpu_temp": jtop_value(lambda k: k.startswith("temp") and "cpu" in k),
            "gpu_temp": jtop_value(lambda k: k.startswith("temp") and "gpu" in k),
            "tj_temp": jtop_value(lambda k: k.startswith("temp") and "tj" in k),
            "fan": jtop_value(lambda k: k.startswith("fan")),
            "power_mode": None,
        }
        mode = js.get("nvp model")
        if isinstance(mode, str):
            out["power_mode"] = mode

        # /sys fallbacks for whatever jtop didn't give us.
        if out["cpu_temp"] is None:
            out["cpu_temp"] = self._zone_temp("cpu")
        if out["gpu_temp"] is None:
            out["gpu_temp"] = self._zone_temp("gpu")
        if out["tj_temp"] is None:
            out["tj_temp"] = self._zone_temp("tj")
        if out["gpu"] is None and self.gpu_load_path:
            v = _read_number(self.gpu_load_path)
            if v is not None:
                out["gpu"] = v / 10.0  # reported in tenths of a percent
        if out["fan"] is None and self.fan_path:
            v = _read_number(self.fan_path)
            if v is not None:
                out["fan"] = v * 100 / 255
        return out


def short_power_mode(mode):
    """'15W' / '25W' / 'MAXN' from names like 'MAXN_SUPER' or '15W'."""
    if not mode:
        return "--"
    mode = mode.upper()
    if "MAXN" in mode:
        return "MAXN"
    return mode.split("_")[0][:5]


# ---------------------------------------------------------------- lab services

def port_open(port, host="127.0.0.1", timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ollama_up():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


VOICE_PORTS = {"whisper": 10300, "piper": 10200, "wakeword": 10400}


class LabServices:
    """Checks the Orin's lab services in a background thread so the screen never stalls."""

    def __init__(self, mqtt_host="127.0.0.1", mqtt_port=1883, mqtt_user=None, mqtt_pass=None,
                 topic="lab/#", interval=10):
        self.interval = interval
        self.mqtt_port = mqtt_port
        self.status = {"mqtt": None, "ollama": None, "voice": None}
        self.last_message = None  # time.time() of the last MQTT message seen
        self.listening = False
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self._client = self._start_mqtt(mqtt_host, mqtt_port, mqtt_user, mqtt_pass, topic)

    def _start_mqtt(self, host, port, user, password, topic):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log("paho-mqtt not installed; MQTT is checked by port only")
            return None
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="oled-stats")
        except AttributeError:  # paho 1.x
            client = mqtt.Client(client_id="oled-stats")
        if user:
            client.username_pw_set(user, password)

        def on_connect(c, userdata, flags, rc, *args):
            ok = (rc == 0) if isinstance(rc, int) else not rc.is_failure
            self.listening = ok
            log("MQTT connected" if ok else f"MQTT login refused ({rc})")
            if ok:
                c.subscribe(topic)

        def on_disconnect(c, *args):
            self.listening = False

        def on_message(c, userdata, msg):
            self.last_message = time.time()

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        try:
            client.connect_async(host, port, keepalive=30)
            client.loop_start()
        except Exception as exc:
            log("MQTT connect failed:", exc)
        return client

    def _loop(self):
        while not self._stop.is_set():
            voice_up = sum(port_open(p) for p in VOICE_PORTS.values())
            self.status = {
                "mqtt": port_open(self.mqtt_port),
                "ollama": ollama_up(),
                "voice": voice_up,
            }
            log("services:", self.status)
            self._stop.wait(self.interval)

    def close(self):
        self._stop.set()
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass


def ago(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


# ---------------------------------------------------------------- pages

def _t(v):
    return "--" if v is None else f"{v:.0f}C"


def _pct(v):
    return "--" if v is None else f"{v:.0f}%"


def _ok(v):
    return "--" if v is None else ("ok" if v else "DOWN")


def page_system(s):
    ram_pct, ram_gb = s["ram"]
    disk_pct, disk_gb = s["disk"]
    return [
        f"CPU {s['cpu']:3d}%".ljust(LINE_CHARS - 8) + s["clock"],
        f"RAM {ram_pct:2d}% of {ram_gb:.1f}GB",
        f"DSK {disk_pct:2d}% of {disk_gb:.0f}GB",
        f"IP {s['ip'] or 'no network'}",
    ]


def page_jetson(s):
    j = s["jetson"]
    return [
        "JETSON",
        f"GPU {_pct(j['gpu'])}".ljust(11) + short_power_mode(j["power_mode"]),
        f"CPU {_t(j['cpu_temp'])}".ljust(11) + f"GPU {_t(j['gpu_temp'])}",
        f"Tj  {_t(j['tj_temp'])}".ljust(11) + f"Fan {_pct(j['fan'])}",
    ]


def page_lab(s):
    svc = s["services"]
    voice = svc["voice"]
    voice_txt = "--" if voice is None else ("ok" if voice == len(VOICE_PORTS) else f"{voice}/{len(VOICE_PORTS)}")
    if s["last_message"] is not None:
        last = f"last msg {ago(s['now'] - s['last_message'])} ago"
    elif s["mqtt_listening"]:
        last = "no msgs yet"
    else:
        last = "msgs: not watching"
    return [
        "LAB SERVICES",
        f"MQTT {_ok(svc['mqtt'])}".ljust(10) + f"Ollama {_ok(svc['ollama'])}",
        f"Voice {voice_txt}",
        last,
    ]


def page_alert(s):
    j = s["jetson"]
    return [
        "!! RUNNING HOT !!",
        f"Tj  {_t(j['tj_temp'])}",
        f"CPU {_t(j['cpu_temp'])}  GPU {_t(j['gpu_temp'])}",
        f"Fan {_pct(j['fan'])}",
    ]


PAGES = [page_system, page_jetson, page_lab]


def load_font():
    # The small fixed bitmap font, so 4 lines fit. Newer Pillow's
    # load_default() returns a bigger TrueType font instead.
    if hasattr(ImageFont, "load_default_imagefont"):
        return ImageFont.load_default_imagefont()
    return ImageFont.load_default()


def render(lines, font, offset=(0, 0), invert=False):
    image = Image.new("1", (WIDTH, HEIGHT), 1 if invert else 0)
    draw = ImageDraw.Draw(image)
    dx, dy = offset
    for i, text in enumerate(lines[:4]):
        draw.text((dx, i * 8 - 2 + dy), text[:LINE_CHARS], font=font, fill=0 if invert else 1)
    return image


# ---------------------------------------------------------------- hardware

def find_bus(addr, forced=None):
    if forced is not None:
        return forced
    try:
        import smbus
    except ImportError:
        return None
    for b in CANDIDATE_BUSES:
        try:
            bus = smbus.SMBus(b)
        except OSError:
            continue
        try:
            bus.read_byte(addr)
            return b
        except OSError:
            pass
        finally:
            bus.close()
    return None


class Case:
    """Fan and RGB bar on the case controller. Does nothing if it isn't found."""

    def __init__(self, bus_num):
        self.bus = None
        try:
            import smbus
            if bus_num is not None:
                self.bus = smbus.SMBus(bus_num)
        except (ImportError, OSError) as exc:
            log("case controller not available:", exc)

    def _write(self, reg, value):
        if self.bus is None:
            return
        try:
            self.bus.write_byte_data(CASE_ADDR, reg, value)
            time.sleep(0.05)
        except OSError as exc:
            log(f"case write reg 0x{reg:02X} failed:", exc)

    def fan(self, on):
        self._write(REG_FAN, 1 if on else 0)

    def rgb_effect(self, effect):
        self._write(REG_RGB_EFFECT, effect)

    def rgb_off(self):
        self._write(REG_RGB_OFF, 0)


class Screen:
    SET_CONTRAST, DISPLAY_OFF, DISPLAY_ON = 0x81, 0xAE, 0xAF

    def __init__(self, bus_num):
        self.bus_num = bus_num
        self.oled = None
        self.state = None  # "on", "dim" or "off"

    def open(self):
        try:
            import Adafruit_SSD1306
            # gpio=1 stops the library probing for a reset pin (there isn't one).
            oled = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=self.bus_num, gpio=1)
            oled.begin()
            oled.clear()
            oled.display()
            self.oled = oled
            self.state = None
            log(f"OLED found on I2C bus {self.bus_num}")
            return True
        except Exception as exc:
            log(f"OLED not found on bus {self.bus_num}:", exc)
            self.oled = None
            return False

    def set_state(self, state):
        """'on' = full brightness, 'dim' = lowest brightness, 'off' = panel off."""
        if state == self.state or self.oled is None:
            return
        if state == "off":
            self.oled.command(self.DISPLAY_OFF)
        else:
            self.oled.command(self.DISPLAY_ON)
            self.oled.command(self.SET_CONTRAST)
            self.oled.command(0x01 if state == "dim" else 0xCF)
        self.state = state

    def show(self, image):
        self.oled.image(image)
        self.oled.display()

    def blank(self):
        if self.oled is None and not self.open():
            return
        try:
            self.oled.clear()
            self.oled.display()
        except Exception:
            pass


# ---------------------------------------------------------------- main

def in_night(hour, start, end):
    if start == end:
        return False
    return start <= hour < end if start < end else (hour >= start or hour < end)


def gather(cpu, jetson, services):
    now = time.time()
    return {
        "now": now,
        "clock": time.strftime("%H:%M:%S"),
        "cpu": cpu.percent(),
        "ram": ram(),
        "disk": disk(),
        "ip": local_ip(),
        "jetson": jetson.read(),
        "services": services.status,
        "last_message": services.last_message,
        "mqtt_listening": services.listening,
    }


def preview(path):
    """Render every page with sample numbers to PNGs (scaled up 4x), no hardware needed."""
    sample = {
        "now": 1000.0, "clock": "17:42:05", "cpu": 12, "ram": (35, 7.4), "disk": (41, 233.0),
        "ip": "192.168.0.50",
        "jetson": {"gpu": 34.0, "power_mode": "MAXN_SUPER", "cpu_temp": 52.1, "gpu_temp": 49.4,
                   "tj_temp": 54.0, "fan": 40.0},
        "services": {"mqtt": True, "ollama": True, "voice": 3},
        "last_message": 997.0, "mqtt_listening": True,
    }
    font = load_font()
    base, ext = os.path.splitext(path)
    for i, page in enumerate(PAGES + [page_alert], 1):
        lines = page(sample)
        for line in lines:
            assert len(line) <= LINE_CHARS, f"line too long for the screen: {line!r}"
        img = render(lines, font, invert=(page is page_alert)).convert("L")
        out = f"{base}-{i}{ext or '.png'}"
        img.resize((WIDTH * 4, HEIGHT * 4), Image.NEAREST).save(out)
        print("saved", out)


def main():
    global DEBUG
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", choices=["run", "clear"], default="run")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--preview", metavar="FILE.png", help="save each page as an image and exit")
    p.add_argument("--oled-bus", type=int, help="I2C bus of the screen (default: auto)")
    p.add_argument("--case-bus", type=int, help="I2C bus of the fan/RGB controller (default: auto)")
    p.add_argument("--effect", type=int, default=3, choices=range(1, 6), help="RGB light effect 1-5")
    p.add_argument("--page-seconds", type=float, default=5, help="time on each page")
    p.add_argument("--alert-temp", type=float, default=85, help="flash a warning at this Tj temperature (C)")
    p.add_argument("--night-start", type=int, default=23, help="hour the screen dims (0-23)")
    p.add_argument("--night-end", type=int, default=7, help="hour it comes back (set equal to start to disable)")
    p.add_argument("--night-mode", choices=["dim", "off"], default="dim")
    p.add_argument("--mqtt-host", default="127.0.0.1")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--mqtt-topic", default="lab/#")
    a = p.parse_args()
    DEBUG = a.debug

    if a.preview:
        preview(a.preview)
        return

    oled_bus = find_bus(OLED_ADDR, a.oled_bus)
    case_bus = find_bus(CASE_ADDR, a.case_bus)
    log(f"screen bus: {oled_bus}, case controller bus: {case_bus}")
    screen = Screen(oled_bus if oled_bus is not None else CANDIDATE_BUSES[0])
    case = Case(case_bus)

    if a.mode == "clear":
        screen.blank()
        case.rgb_off()
        case.fan(False)
        print("screen cleared, RGB and fan off")
        return

    def on_sigterm(signum, frame):  # systemd stop -> same as Ctrl+C
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)

    case.fan(True)
    case.rgb_effect(a.effect)
    font = load_font()
    cpu = CpuMeter()
    jetson = JetsonStats()
    services = LabServices(a.mqtt_host, a.mqtt_port, os.environ.get("MQTT_USER"),
                           os.environ.get("MQTT_PASS"), a.mqtt_topic)
    start = time.time()
    flash = False

    try:
        while True:
            if screen.oled is None and not screen.open():
                time.sleep(2)  # screen unplugged or not found yet; try again
                continue
            try:
                s = gather(cpu, jetson, services)
                tj = s["jetson"]["tj_temp"]
                hot = tj is not None and tj >= a.alert_temp
                elapsed = s["now"] - start
                offset = BURN_IN_OFFSETS[int(elapsed // 180) % len(BURN_IN_OFFSETS)]
                if hot:
                    flash = not flash
                    screen.set_state("on")  # always show a heat warning, even at night
                    image = render(page_alert(s), font, offset, invert=flash)
                else:
                    night = in_night(time.localtime().tm_hour, a.night_start, a.night_end)
                    screen.set_state(a.night_mode if night else "on")
                    page = PAGES[int(elapsed // a.page_seconds) % len(PAGES)]
                    image = render(page(s), font, offset)
                screen.show(image)
            except OSError as exc:  # I2C error, e.g. cable knocked loose
                log("screen write failed, will reconnect:", exc)
                screen.oled = None
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        screen.blank()
        case.rgb_off()  # the fan is left running on purpose, for cooling
        services.close()
        jetson.close()
        print("stopped")


if __name__ == "__main__":
    sys.exit(main())
