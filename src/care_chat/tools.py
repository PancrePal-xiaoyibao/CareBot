from __future__ import annotations

from agents import function_tool


@function_tool
def grounding_exercise(feeling: str) -> str:
    """Return a short grounding exercise for distress, panic, or overwhelm."""
    return (
        f"当前情绪焦点: {feeling}\n"
        "建议先做一个 90 秒的落地练习：\n"
        "1. 双脚踩地，慢慢吸气 4 秒，呼气 6 秒，连续做 5 轮。\n"
        "2. 说出你眼前能看到的 5 样东西、能摸到的 4 样东西。\n"
        "3. 把注意力放回一个最小动作，比如喝一口水、把肩膀放松。\n"
        "4. 如果身边有人，发一句很短的信息: 我现在有点撑不住，能陪我一下吗？\n"
        "5. 如果情绪持续升级，联系你的照护者、心理支持资源或医疗团队。"
    )


@function_tool
def doctor_question_builder(topic: str, current_symptoms: str | None = None) -> str:
    """Build a concise list of questions the user can ask their care team."""
    symptom_line = current_symptoms or "暂无补充症状"
    return (
        f"就诊主题: {topic}\n"
        f"当前症状: {symptom_line}\n"
        "可以优先问医生这些问题:\n"
        "1. 现在最需要关注的风险信号是什么？出现什么情况要当天联系你们？\n"
        "2. 这些症状更可能与治疗副作用、感染风险，还是原发疾病相关？\n"
        "3. 今天之后我应该记录哪些指标，比如体温、疼痛评分、进食饮水、排便或睡眠？\n"
        "4. 目前有哪些安全的缓解方式是我可以先做的？哪些做法不建议自己尝试？\n"
        "5. 如果今晚或周末加重，我应该联系谁，走什么流程？"
    )


@function_tool
def symptom_journal_template(focus: str) -> str:
    """Return a structured symptom log template for the next clinician contact."""
    return (
        f"症状记录模板: {focus}\n"
        "- 开始时间:\n"
        "- 持续多久:\n"
        "- 严重程度 0-10:\n"
        "- 伴随症状:\n"
        "- 今天体温/进食/饮水/睡眠:\n"
        "- 我已经做过什么:\n"
        "- 哪些情况会变重或变轻:\n"
        "- 我最想让医生回答的问题:"
    )


@function_tool
def urgent_support_playbook(
    symptoms: str,
    on_active_treatment: bool = True,
    temperature_c: float | None = None,
) -> str:
    """Return a conservative escalation checklist for concerning symptoms."""
    treatment_line = "正在接受治疗" if on_active_treatment else "当前不确定是否正在治疗"
    temperature_line = f"{temperature_c}C" if temperature_c is not None else "未提供"
    return (
        f"症状摘要: {symptoms}\n"
        f"治疗状态: {treatment_line}\n"
        f"体温: {temperature_line}\n"
        "保守处理建议:\n"
        "1. 如果有呼吸困难、胸痛、抽搐、叫不醒、大量出血或明显意识改变，立即去急诊或呼叫急救。\n"
        "2. 如果在化疗、免疫治疗或放疗期间出现发热、寒战、反复呕吐、无法进水、症状快速加重，今天就联系肿瘤科团队。\n"
        "3. 在等待线下帮助时，准备药物清单、最近一次治疗时间、体温和症状开始时间。"
    )


@function_tool
def caregiver_coordination_plan(primary_need: str, next_24_hours: str | None = None) -> str:
    """Create a caregiver-facing coordination checklist for the next day or shift."""
    timing = next_24_hours or "接下来 24 小时"
    return (
        f"照护重点: {primary_need}\n"
        f"时间范围: {timing}\n"
        "建议照护分工:\n"
        "1. 先确认最关键的一件事: 今天最需要盯的是症状变化、就诊准备，还是生活照料。\n"
        "2. 指定一个主要联系人负责和患者、医院沟通，避免多人同时传话。\n"
        "3. 记录今天需要观察的 3 项信息: 体温/症状变化、进食饮水、精神状态或睡眠。\n"
        "4. 提前整理药物、治疗时间、医保/就诊材料和要问医生的问题。\n"
        "5. 如果照护者自己已经明显透支，安排一个可替班的人和一个最小休息窗口。"
    )


@function_tool
def community_help_request(need: str, timeframe: str | None = None) -> str:
    """Draft a short practical help request for friends, family, or volunteers."""
    window = timeframe or "这几天"
    return (
        f"需要支持的事项: {need}\n"
        f"时间范围: {window}\n"
        "可直接发送的求助消息:\n"
        f"大家好，这边最近因为治疗相关安排，{window} 需要一些实际帮助，"
        f"主要是 {need}。如果你这段时间方便支持一次，请直接回复我可提供的时间和方式。"
        "为了减少来回沟通，我会统一整理安排。谢谢你们。"
    )


@function_tool
def volunteer_support_boundaries(task: str, observed_concern: str | None = None) -> str:
    """Return a safe support checklist for a community volunteer."""
    concern = observed_concern or "暂无额外异常"
    return (
        f"志愿服务任务: {task}\n"
        f"观察到的情况: {concern}\n"
        "安全边界提醒:\n"
        "1. 你的职责优先是陪伴、接送、跑腿、信息转达和生活支持，不是做医疗判断。\n"
        "2. 不替患者决定是否用药，也不自行处理导管、伤口、注射或治疗设备。\n"
        "3. 如果你看到高热、呼吸困难、明显意识变化、反复呕吐或快速恶化，立刻联系家属或医疗团队。\n"
        "4. 只共享获得允许的必要信息，避免在群里扩散病情细节。\n"
        "5. 结束服务前，留下一条简短交接: 你做了什么、看到了什么、谁接手。"
    )
