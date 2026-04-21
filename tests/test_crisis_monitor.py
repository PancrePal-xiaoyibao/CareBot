from care_chat.agents import _parse_crisis_assessment


def test_parse_crisis_assessment_valid_json() -> None:
    raw = '{"risk_level": "high", "reason": "用户提到了自杀想法"}'
    result = _parse_crisis_assessment(raw)
    assert result.risk_level == "high"
    assert "自杀" in result.reason


def test_parse_crisis_assessment_with_markdown_fences() -> None:
    raw = '```json\n{"risk_level": "moderate", "reason": "绝望感"}\n```'
    result = _parse_crisis_assessment(raw)
    assert result.risk_level == "moderate"


def test_parse_crisis_assessment_invalid_json_defaults_to_none() -> None:
    raw = "This is not JSON at all"
    result = _parse_crisis_assessment(raw)
    assert result.risk_level == "none"


def test_parse_crisis_assessment_none_risk() -> None:
    raw = '{"risk_level": "none", "reason": "正常对话"}'
    result = _parse_crisis_assessment(raw)
    assert result.risk_level == "none"
