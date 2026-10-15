"""OLED stats screen: pages fit the 21-character screen, helpers behave."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "orin"))
import oled_stats as o  # noqa: E402


def sample(**jetson):
    j = {"gpu": 100.0, "power_mode": "MAXN_SUPER", "cpu_temp": 101.4, "gpu_temp": 99.9,
         "tj_temp": 102.0, "fan": 100.0}
    j.update(jetson)
    return {"now": 100000.0, "clock": "23:59:59", "cpu": 100, "ram": (100, 15.6), "disk": (100, 1999.0),
            "ip": "192.168.100.200", "jetson": j,
            "services": {"mqtt": False, "ollama": False, "voice": 2},
            "last_message": 0.0, "mqtt_listening": True}


def test_pages_fit_screen_worst_case():
    for s in (sample(), sample(gpu=None, power_mode=None, cpu_temp=None, gpu_temp=None, tj_temp=None, fan=None)):
        for page in o.PAGES + [o.page_alert]:
            lines = page(s)
            assert len(lines) <= 4
            for line in lines:
                assert len(line) <= o.LINE_CHARS, line


def test_power_mode_names():
    assert o.short_power_mode("MAXN_SUPER") == "MAXN"
    assert o.short_power_mode("15W") == "15W"
    assert o.short_power_mode("25W_xyz") == "25W"
    assert o.short_power_mode(None) == "--"


def test_night_window_wraps_midnight():
    assert o.in_night(23, 23, 7) and o.in_night(3, 23, 7)
    assert not o.in_night(7, 23, 7) and not o.in_night(12, 23, 7)
    assert o.in_night(14, 13, 15) and not o.in_night(15, 13, 15)
    assert not o.in_night(5, 0, 0)


def test_ago():
    assert o.ago(5) == "5s" and o.ago(125) == "2m" and o.ago(7300) == "2h" and o.ago(200000) == "2d"
