import pytest

from core.json_parser import JsonParseError, parse_llm_json


def test_parse_plain_json():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_fenced_json_once():
    raw = '```json\n{"a": 1}\n```'
    assert parse_llm_json(raw) == {"a": 1}


def test_parse_invalid_json_fails():
    with pytest.raises(JsonParseError):
        parse_llm_json("{not json")


def test_parse_does_not_repair_partial_json():
    with pytest.raises(JsonParseError):
        parse_llm_json('{"a": 1,')
