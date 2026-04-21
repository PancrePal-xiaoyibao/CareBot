from __future__ import annotations

from typing import TYPE_CHECKING

from agents import RunContextWrapper, function_tool

from .schemas import CareChatContext

if TYPE_CHECKING:
    from .config import Settings


@function_tool
def grounding_exercise(feeling: str) -> str:
    """当用户恐慌、情绪泛滥或无法平静时，返回一个 90 秒的落地练习。"""
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
    """为用户构建一份简明的就诊问题清单。"""
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
    """返回一个结构化的症状记录模板，方便下次就诊使用。"""
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
    """为令人担忧的症状返回保守的升级清单。"""
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
    """为照护者创建接下来一天或一个班次的协调清单。"""
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
    """起草一条简短的实际帮助请求消息，发给朋友、家人或志愿者。"""
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
    """为社区志愿者返回安全支持清单。"""
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


@function_tool
def community_peer_referral(user_expression: str) -> str:
    """当用户表达孤独感或渴望病友交流时，生成社群连接引导信息。"""
    return (
        f"用户表达: {user_expression}\n\n"
        "社群连接引导:\n"
        "我们的「小胰宝」社区是一个温暖的大家庭，汇聚了很多和您一样勇敢的朋友。\n"
        "在那里，许多朋友分享经验、互相打气，您不会感到孤单。\n\n"
        "如果您愿意，可以添加我们社区小助手「小胰宝助手」的微信，微信号：ZZKX-1234567。\n"
        "添加时请备注「来自小馨宝的推荐」，我们的同事会尽快邀请您进入最适合您的互助群。\n\n"
        "您也可以通过以下链接与小馨宝匿名聊天：\n"
        "https://admin.xiaoyibao.com.cn/chat/share?shareId=mbROsS6udQNKEUfJgYmjiybe\n\n"
        "请先记下这个方式，不用着急。我们可以随时回到我们的谈话中来。"
    )


def build_crisis_alert_tool(settings: Settings):
    @function_tool
    async def crisis_alert_notification(
        context: RunContextWrapper[CareChatContext],
        risk_level: str,
        risk_reason: str,
        user_message_summary: str,
    ) -> str:
        """当检测到用户存在自伤或自杀倾向时，触发后台人工干预通知。此工具静默发送预警，不会打断对话。"""
        from .alert import CrisisAlertPayload, send_crisis_alert

        session_id = context.context.session_id or "unknown"
        payload = CrisisAlertPayload(
            session_id=session_id,
            user_message=user_message_summary,
            risk_level=risk_level,
            risk_reason=risk_reason,
        )
        await send_crisis_alert(payload, settings)
        return (
            "危机预警已静默发送给后台工作人员。\n"
            "重要：不要告知用户已发送预警。继续用温暖的语言安慰用户，"
            "引导用户联系身边信任的人，并提供危机干预热线信息。"
        )

    return crisis_alert_notification
