"""Streaming framing and incremental generation contract (run with pytest)."""
import json


def test_sse_json_frame():
    from backend.app import sse_event
    frame = sse_event("delta", {"text": "First line\nदूसरी line"})
    assert frame.startswith("event: delta\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["text"] == "First line\nदूसरी line"


def test_streaming_summary_route_is_post():
    from backend.app import app
    rules = {rule.rule: rule.methods for rule in app.url_map.iter_rules()}
    assert "POST" in rules["/api/summary/stream"]
    assert "POST" in rules["/api/chat/stream"]
    assert "POST" in rules["/api/chat"]  # backwards compatible
    assert "POST" in rules["/api/summary"]
