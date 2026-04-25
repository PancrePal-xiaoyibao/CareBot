"""QA 对照测试：CareBot vs 社区 FastGPT

发送相同的测试问题给两端，对比输出长度和质量。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# 配置(key用完就删除，还需要类似测试自己后台生成一个新key填入即可)
# ---------------------------------------------------------------------------

FASTGPT_BASE_URL = "https://admin.xiaoyibao.com.cn/api/v1"
FASTGPT_API_KEY = "openapi-nDKNkOXeYK8siby0XyO3bRD1KlCEefsEqKn9hSud5eX51tYkuWJiMKE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TEST_QUESTIONS = [
    {
        "id": "Q1",
        "role_hint": "patient",
        "message": "你好，我是刚确诊的乳腺癌患者，最近心情很不好",
    },
    {
        "id": "Q2",
        "role_hint": "patient",
        "message": "最近化疗后恶心吃不下东西，感觉自己快撑不住了",
    },
    {
        "id": "Q3",
        "role_hint": "caregiver",
        "message": "我妈妈得了胰腺癌，我作为她的照护者感觉压力好大，经常失眠",
    },
    {
        "id": "Q4",
        "role_hint": "patient",
        "message": "治疗费用太高了，家里经济压力大，我不知道该怎么办",
    },
    {
        "id": "Q5",
        "role_hint": "patient",
        "message": "我想找病友聊聊，感觉很孤单",
    },
]


# ---------------------------------------------------------------------------
# FastGPT 调用
# ---------------------------------------------------------------------------

async def call_fastgpt(message: str, chat_id: str) -> dict:
    """调用社区 FastGPT，返回非流式响应。"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{FASTGPT_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {FASTGPT_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "stream": False,
                "detail": False,
                "chatId": chat_id,
                "messages": [{"role": "user", "content": message}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = ""
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            content += msg.get("content", "")
        return {
            "content": content,
            "char_count": len(content),
            "usage": data.get("usage", {}),
            "raw_choices": data.get("choices", []),
        }


# ---------------------------------------------------------------------------
# CareBot 调用
# ---------------------------------------------------------------------------

async def call_carebot(message: str, role_hint: str, session_id: str) -> dict:
    """调用本地 CareBot，返回完整回复。"""
    from care_chat.config import load_settings
    from care_chat.schemas import CareRoleHint
    from care_chat.service import CareChatService

    settings = load_settings()
    db_path = PROJECT_ROOT / ".local" / f"qa_test_{session_id}.sqlite3"

    service = CareChatService(
        settings=settings,
        session_id=session_id,
        db_path=db_path,
    )
    try:
        await service._ensure_agent_ready()
        start = time.time()
        chunks: list[str] = []
        async for chunk in service.stream_reply(
            message,
            include_reasoning=False,
            role_hint=role_hint,  # type: ignore[arg-type]
        ):
            chunks.append(chunk.text)
        elapsed = time.time() - start
        content = "".join(chunks).strip()
        return {
            "content": content,
            "char_count": len(content),
            "elapsed_seconds": round(elapsed, 2),
        }
    finally:
        await service.aclose()


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  QA 对照测试报告：CareBot vs 社区 FastGPT")
    print("=" * 80)

    for r in results:
        qid = r["id"]
        msg = r["message"][:40] + "..." if len(r["message"]) > 40 else r["message"]
        carebot = r.get("carebot", {})
        fastgpt = r.get("fastgpt", {})
        cb_chars = carebot.get("char_count", "N/A")
        fg_chars = fastgpt.get("char_count", "N/A")
        cb_time = carebot.get("elapsed_seconds", "N/A")

        print(f"\n{'─' * 70}")
        print(f"  [{qid}] {msg}")
        print(f"{'─' * 70}")
        print(f"  CareBot  | {cb_chars} 字 | {cb_time}s")
        print(f"  FastGPT  | {fg_chars} 字")
        print(f"{'─' * 70}")

        if carebot.get("content"):
            print(f"\n  [CareBot 回复]")
            content = carebot["content"]
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(f"  {preview}")

        if fastgpt.get("content"):
            print(f"\n  [FastGPT 回复]")
            content = fastgpt["content"]
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(f"  {preview}")

    print(f"\n{'=' * 80}")
    print("  汇总统计")
    print("=" * 80)

    cb_counts = [r["carebot"]["char_count"] for r in results if r.get("carebot", {}).get("char_count")]
    fg_counts = [r["fastgpt"]["char_count"] for r in results if r.get("fastgpt", {}).get("char_count")]

    if cb_counts:
        print(f"  CareBot 平均字数: {sum(cb_counts) / len(cb_counts):.0f}")
        print(f"  CareBot 最短/最长: {min(cb_counts)} / {max(cb_counts)}")
    if fg_counts:
        print(f"  FastGPT 平均字数: {sum(fg_counts) / len(fg_counts):.0f}")
        print(f"  FastGPT 最短/最长: {min(fg_counts)} / {max(fg_counts)}")

    if cb_counts and fg_counts:
        ratio = (sum(cb_counts) / len(cb_counts)) / (sum(fg_counts) / len(fg_counts))
        print(f"  CareBot / FastGPT 字数比: {ratio:.2f}")

    print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def run_all() -> None:
    results: list[dict] = []

    for q in TEST_QUESTIONS:
        qid = q["id"]
        msg = q["message"]
        role = q["role_hint"]
        print(f"\n>>> 测试 {qid}: {msg[:50]}...")

        result_entry: dict = {"id": qid, "message": msg}

        # CareBot
        print(f"    调用 CareBot...")
        try:
            cb_result = await call_carebot(msg, role, f"qa-{qid}")
            result_entry["carebot"] = cb_result
            print(f"    CareBot 完成: {cb_result['char_count']} 字, {cb_result['elapsed_seconds']}s")
        except Exception as exc:
            result_entry["carebot"] = {"error": str(exc)}
            print(f"    CareBot 失败: {exc}")

        # FastGPT
        print(f"    调用 FastGPT...")
        try:
            fg_result = await call_fastgpt(msg, chat_id=f"qa-compare-{qid}")
            result_entry["fastgpt"] = fg_result
            print(f"    FastGPT 完成: {fg_result['char_count']} 字")
        except Exception as exc:
            result_entry["fastgpt"] = {"error": str(exc)}
            print(f"    FastGPT 失败: {exc}")

        results.append(result_entry)

    # 保存原始 JSON（先于打印，避免报告打印崩溃丢失数据）
    report_path = PROJECT_ROOT / ".local" / "qa_compare_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原始报告已保存到: {report_path}")

    print_report(results)


if __name__ == "__main__":
    asyncio.run(run_all())
