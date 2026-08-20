"""GPU-free tests for the Phase 1 taxonomy classifier (CI-safe)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))

from classify import bootstrap_ci, classify, extract_tool_calls  # noqa: E402

ENTRY = {
    "id": "simple_python_0",
    "ground_truth": [{"get_weather": {"city": ["Tokyo"]}}],
}


def test_correct_json_body():
    raw = '<tool_call>\nget_weather({"city": "Tokyo"})\n</tool_call>'
    assert classify(ENTRY, raw, "simple_python", 192, 50) == "correct"


def test_correct_legacy_tag():
    raw = "<get_weather city=Tokyo>"
    assert classify(ENTRY, raw, "simple_python", 192, 50) == "correct"


def test_wrong_args():
    raw = '<tool_call>\nget_weather({"temp": "25"})\n</tool_call>'
    assert classify(ENTRY, raw, "simple_python", 192, 50) == "wrong_args"


def test_missed_required():
    raw = "I do not have weather data."
    assert classify(ENTRY, raw, "simple_python", 192, 10) == "missed_required"


def test_wrong_tool():
    raw = '<tool_call>\nget_time({"city": "Tokyo"})\n</tool_call>'
    assert classify(ENTRY, raw, "simple_python", 192, 50) == "wrong_tool"


def test_truncation():
    raw = '<tool_call>\nget_weather({"city": "To'
    assert classify(ENTRY, raw, "simple_python", 192, 192) == "truncation"


def test_unnecessary():
    no_expected = {"id": "relevance_0", "ground_truth": None}
    raw = '<tool_call>\nget_weather({"city": "Tokyo"})\n</tool_call>'
    assert classify(no_expected, raw, "relevance", 192, 50) == "unnecessary"


def test_extract_parse_json():
    raw = '<tool_call>\nget_weather({"city": "Tokyo"})\n</tool_call>'
    assert extract_tool_calls(raw) == [{"get_weather": {"city": "Tokyo"}}]


def test_json_object_form_correct():
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    assert classify(ENTRY, raw, "simple_python", 192, 50) == "correct"


def test_extract_json_object_form():
    raw = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
    assert extract_tool_calls(raw) == [{"get_weather": {"city": "Tokyo"}}]


def test_bootstrap_ci_bounds():
    labels = ["correct"] * 30 + ["wrong_args"] * 10 + ["missed_required"] * 10
    ci = bootstrap_ci(labels, ["correct", "wrong_args", "missed_required", "malformed"])
    assert ci["correct"]["mean"] == 0.6
    assert 0.0 <= ci["correct"]["ci_lo"] <= ci["correct"]["ci_hi"] <= 1.0
