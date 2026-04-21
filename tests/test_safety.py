from care_chat.safety import (
    SafetyCategory,
    build_guardrail_response,
    build_output_block_response,
    detect_local_crisis,
)


def test_detect_local_crisis_for_self_harm() -> None:
    alert = detect_local_crisis("我不想活了，刚刚一直在想伤害自己。")

    assert alert is not None
    assert alert.category == SafetyCategory.SELF_HARM


def test_detect_local_crisis_for_medical_emergency() -> None:
    alert = detect_local_crisis("我现在胸痛而且完全喘不过气。")

    assert alert is not None
    assert alert.category == SafetyCategory.MEDICAL_EMERGENCY


def test_detect_local_crisis_returns_none_for_general_distress() -> None:
    assert detect_local_crisis("我今天很焦虑，也很害怕下一次化疗。") is None


def test_guardrail_response_mentions_988_for_self_harm() -> None:
    response = build_guardrail_response("self_harm", language="zh-CN")

    assert "988" in response


def test_output_block_response_stays_within_safe_boundary() -> None:
    response = build_output_block_response(language="zh-CN")

    assert "医生" in response
