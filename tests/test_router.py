"""Intent router tests: rules, model replies (faked), checks, actions and the log."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "orin"))
import router as r  # noqa: E402


def make_router(model_reply=None, fail=False):
    d = tempfile.mkdtemp()
    s = r.settings({"log_file": os.path.join(d, "router.jsonl")})
    sent, calls = [], []

    def fake_post(url, body, timeout):
        calls.append(body)
        if fail:
            raise OSError("connection refused")
        return {"message": {"content": json.dumps(model_reply)}}

    router = r.Router(s, publish=lambda t, p: sent.append((t, p)), post=fake_post)
    return router, sent, calls, s


def read_log(s):
    with open(s["log_file"]) as f:
        return [json.loads(line) for line in f]


def test_rules_catch_common_commands_without_the_model():
    router, sent, calls, _ = make_router()
    cases = {
        "Jarvis, turn on the exhaust fan.": {"intent": "set_device", "device": "exhaust", "state": "on"},
        "lights off please": {"intent": "set_device", "device": "lights", "state": "off"},
        "switch the fume extractor off": {"intent": "set_device", "device": "exhaust", "state": "off"},
        "clear the bench": {"intent": "clear_bench"},
        "show me a breadboard grid": {"intent": "show_grid", "cols": 30, "rows": 10},
        "show grid 20 by 5": {"intent": "show_grid", "cols": 20, "rows": 5},
        "write solder here": {"intent": "show_text", "text": "solder here"},
    }
    for text, expected in cases.items():
        result = router.handle(text, "test")
        assert result["path"] == "rule", text
        assert result["decision"] == expected, (text, result["decision"])
    assert calls == []  # the model was never asked


def test_model_reply_is_used_and_published():
    router, sent, calls, s = make_router({"intent": "set_device", "device": "exhaust", "state": "on"})
    result = router.handle("get rid of these fumes", "voice")
    assert result["path"] == "model"
    assert ("lab/device/exhaust/set", {"state": "on"}) in sent
    assert sent[-1][0] == "lab/router/reply" and sent[-1][1]["text"] == "Exhaust on."
    body = calls[0]
    assert body["format"]["properties"]["intent"]["enum"] == r.INTENTS
    assert body["options"]["temperature"] == 0


def test_invented_device_goes_to_the_brain():
    router, sent, _, _ = make_router({"intent": "set_device", "device": "laser", "state": "on"})
    result = router.handle("fire up the laser", "voice")
    assert result["decision"] == {"intent": "ask_brain", "text": "fire up the laser"}
    assert sent[0][0] == "lab/brain/request" and sent[0][1]["id"] == result["id"]


def test_model_down_falls_back_to_the_brain():
    router, sent, _, s = make_router(fail=True)
    result = router.handle("what's the pinout of an ESP32", "voice")
    assert result["path"] == "model_error"
    assert result["decision"]["intent"] == "ask_brain"
    assert sent[0][0] == "lab/brain/request"


def test_grid_and_text_scenes():
    router, sent, _, s = make_router({"intent": "show_grid", "cols": 500, "rows": "x"})
    router.handle("I need a huge grid", "voice")
    topic, scene = sent[0]
    assert topic == "lab/display/bench"
    item = scene["items"][0]
    assert item["type"] == "grid" and abs(item["w"] - 63 * 2.54) < 1e-9 and abs(item["h"] - 9 * 2.54) < 1e-9

    router2, sent2, _, _ = make_router()
    router2.handle("write hello", "test")
    item = sent2[0][1]["items"][0]
    assert item["type"] == "text" and item["text"] == "hello" and item["x"] == 600 and item["y"] == 300


def test_dry_run_sends_nothing_but_logs():
    router, sent, _, s = make_router()
    router.handle("clear", "cli", dry_run=True)
    assert sent == []
    rec = read_log(s)[0]
    assert rec["type"] == "decision" and rec["dry_run"] and rec["decision"] == {"intent": "clear_bench"}


def test_feedback_and_brain_reply():
    router, sent, _, s = make_router()
    router.on_command("lab/router/command", "lights on")        # plain-text payload
    rid = read_log(s)[0]["id"]
    router.on_feedback("lab/router/feedback", {"id": rid, "correct": False, "expected": "bench lights only"})
    router.on_brain_reply("lab/brain/reply", {"id": "abc", "text": "Pin 1 is 3V3.",
                                              "scene": {"items": [{"type": "text", "x": 1, "y": 1, "text": "3V3"}]}})
    kinds = [rec["type"] for rec in read_log(s)]
    assert kinds == ["decision", "feedback", "brain_reply"]
    assert ("lab/display/bench", {"items": [{"type": "text", "x": 1, "y": 1, "text": "3V3"}]}) in sent
    assert sent[-1] == ("lab/router/reply", {"id": "abc", "text": "Pin 1 is 3V3.", "intent": "ask_brain"})


def test_settings_override():
    s = r.settings({"devices": {"heater": []}, "ollama": {"model": "llama3.2:3b"}})
    assert s["devices"] == {"heater": []}
    assert s["ollama"]["model"] == "llama3.2:3b" and s["ollama"]["url"].endswith(":11434")
