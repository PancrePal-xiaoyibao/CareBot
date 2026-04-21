from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class SafetyCategory(StrEnum):
    MEDICAL_EMERGENCY = "medical_emergency"
    SELF_HARM = "self_harm"


@dataclass(frozen=True, slots=True)
class SafetyAlert:
    category: SafetyCategory
    matched_terms: tuple[str, ...]


SELF_HARM_PATTERNS = (
    r"想自杀",
    r"不想活了",
    r"结束生命",
    r"伤害自己",
    r"自我伤害",
    r"kill myself",
    r"end my life",
    r"hurt myself",
    r"suicid",
)

MEDICAL_EMERGENCY_PATTERNS = (
    r"呼吸困难",
    r"喘不过气",
    r"无法呼吸",
    r"胸痛",
    r"剧烈胸痛",
    r"昏迷",
    r"叫不醒",
    r"抽搐",
    r"意识不清",
    r"大出血",
    r"大量出血",
    r"can't breathe",
    r"cannot breathe",
    r"severe shortness of breath",
    r"chest pain",
    r"passed out",
    r"unresponsive",
    r"seizure",
    r"bleeding heavily",
)


def _collect_matches(message: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    lowered = message.lower()
    matches = []
    for pattern in patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            matches.append(pattern)
    return tuple(matches)


def detect_local_crisis(message: str) -> SafetyAlert | None:
    self_harm_matches = _collect_matches(message, SELF_HARM_PATTERNS)
    if self_harm_matches:
        return SafetyAlert(
            category=SafetyCategory.SELF_HARM,
            matched_terms=self_harm_matches,
        )

    medical_matches = _collect_matches(message, MEDICAL_EMERGENCY_PATTERNS)
    if medical_matches:
        return SafetyAlert(
            category=SafetyCategory.MEDICAL_EMERGENCY,
            matched_terms=medical_matches,
        )

    return None


def build_guardrail_response(
    category: str,
    language: str = "zh-CN",
    reason: str | None = None,
) -> str:
    normalized = language.lower()
    if normalized.startswith("zh"):
        if category == SafetyCategory.SELF_HARM.value:
            return (
                "我现在更担心你的即时安全。请立刻联系身边可信任的人陪你，不要一个人待着。"
                "如果你在美国，请现在拨打或短信联系 988；如果你有立刻伤害自己的风险，请直接拨打 911 "
                "或去最近的急诊。若你不在美国，请马上联系当地急救电话或危机干预热线。"
            )

        return (
            "这听起来可能涉及紧急情况，我不能替代线下医疗判断。"
            "如果你现在有呼吸困难、胸痛、抽搐、叫不醒、大量出血或意识改变，请立即拨打急救电话"
            "或前往最近的急诊；如果你在美国，请拨打 911。若你正在接受肿瘤治疗，也请尽快联系你的肿瘤科团队。"
        )

    if category == SafetyCategory.SELF_HARM.value:
        return (
            "I am concerned about your immediate safety. Please contact a trusted person now and do not stay alone. "
            "If you are in the United States, call or text 988 right now. If you might act on these thoughts, "
            "call 911 or go to the nearest emergency department."
        )

    return (
        "This may be an emergency, and I cannot assess it safely online. If you have trouble breathing, chest pain, "
        "a seizure, heavy bleeding, confusion, or someone cannot wake you, call emergency services now. "
        "If you are in the United States, call 911, and contact your oncology team as soon as possible."
    )


def build_output_block_response(language: str = "zh-CN") -> str:
    if language.lower().startswith("zh"):
        return (
            "我先收紧一下回答边界：这类问题需要医生或肿瘤科团队给出个体化判断。"
            "如果你愿意，我可以帮你整理要问医生的问题，或者帮你把症状按时间线梳理出来。"
        )

    return (
        "I need to stay inside safer boundaries here. This is better answered by your clinician or oncology team. "
        "If helpful, I can help you organize questions for your doctor or summarize symptoms in a timeline."
    )
