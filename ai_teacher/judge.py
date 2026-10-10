"""L1 视觉模型判定：只处理 L0 拿不准的样本。

成本控制的核心手段：
1. 屏幕+摄像头合成一张图 -> 图片 token 只算一份
2. detail="low" -> 服务端缩到 512x512 后再推理，约 159 tokens/张
3. 提示词极简，不列举规则、不做解释
4. **关闭思考模式** -> 实测开思考要 613 tokens，关掉14，省 40 倍
5. **单次预算上限** -> 见 VisionJudge.max_calls，超过就拒绝发请求

最后一条是硬护栏：调参数时反复试很容易在不知不觉中发出上百次请求
（实测一次调参就花了 118 次、¥0.07）。有了它，即使忘了手动控制也不会失控。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

from .capture import to_data_url
from .config import ApiSettings

# 调试脚本应当设置这个环境变量来豁免预算上限。
# 设为 "1" 表示「我明确知道自己在花钱，主动放开限制」。
BYPASS_ENV = "AI_TEACHER_UNLIMITED_CALLS"

# 极简提示词：不解释规则、不列举例外，只给输出格式。
# 目标描述由 USER_TEMPLATE 内联，整体长度受控。
#
# **必须用双引号**。开启 JSON Output 后模型会模仿提示词里的写法，
# 原来写单引号 {'s':0} 会让模型也回单引号，而 json.loads 解析不了。
# 而且 prompt 越短越省 token——实测带完整样例的版本多花 43 tokens。
PROMPT = (
    "图1=屏幕，图2=摄像头(可能无)。判断用户此刻是否在认真学习。\n"
    '仅输出 json，格式：{"s": 0或1, "c": "≤8字"}'
)

USER_TEMPLATE = "目标：{goal}"


@dataclass(slots=True)
class ModelVerdict:
    """模型判定结果。"""

    studying: bool
    comment: str
    ok: bool = True
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""


def _extract_json(text: str) -> dict | None:
    """从模型输出里抠出 JSON 对象，容忍多余文字和代码块。"""

    text = (text or "").strip()
    if not text:
        return None
    # 去掉```json 包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 直接尝试
    for candidate in (text, *_brace_candidates(text)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _brace_candidates(text: str) -> list[str]:
    """截取所有形如 {...} 的片段，按长度降序尝试。"""

    out: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
    out.sort(key=len, reverse=True)
    return out


class VisionJudge:
    """DeepSeek视觉判定客户端。

    **单次会话的调用预算**：`max_calls` 是硬护栏，超过后 `judge()` 直接返回
    budget_exceeded 而不再发请求。默认 600次（按60秒间隔约 10 小时）。
    正常使用永远碰不到，但调参数时反复试很容易超——实测一次调参就发了 118 次。

    设置环境变量 `AI_TEACHER_UNLIMITED_CALLS=1` 可豁免，仅用于明确知情的调试。
    """

    # 默认预算：正常跑一天（60~300s 自适应间隔）远小于这个数
    DEFAULT_MAX_CALLS = 600

    def __init__(
        self,
        api: ApiSettings,
        client: httpx.Client | None = None,
        *,
        max_calls: int | None = None,
    ) -> None:
        self.api = api
        # 允许注入 client 便于测试拦截；默认按配置构造
        self._client = client or httpx.Client(
            base_url=api.base_url,
            timeout=api.timeout_s,
            headers={
                "Authorization": f"Bearer {api.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._owns_client = client is None
        # total_calls 统计「逻辑调用」（每 tick 一次），用于报告与护栏
        self.total_calls = 0
        # request_count 统计「真实 HTTP 请求」，含重试，用于防失控
        self.request_count = 0
        self.max_calls = max_calls if max_calls is not None else self.DEFAULT_MAX_CALLS
        self._unlimited = os.environ.get(BYPASS_ENV) == "1"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # 最近一次调用的 token，供日志逐条记录
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def judge(self, jpeg: bytes, goal: str) -> ModelVerdict:
        """判定一次。失败时返回 ok=False，由上层降级处理。"""

        if not self.api.api_key:
            return ModelVerdict(
                studying=True, comment="未配置 API key，跳过判定", ok=False, error="no_api_key"
            )

        # ---- 预算护栏：按「逻辑调用次数」算。超过上限就不再发请求，
        # 避免调参数时反复试导致失控（实测一次调参就发了 118 次）。
        if not self._unlimited and self.total_calls >= self.max_calls:
            return ModelVerdict(
                studying=True,
                comment="已达调用预算上限，停止调用模型",
                ok=False,
                error="budget_exceeded",
            )
        self.total_calls += 1

        payload = {
            "model": self.api.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{PROMPT}\n{USER_TEMPLATE.format(goal=goal)}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": to_data_url(jpeg),
                                "detail": self.api.image_detail,
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.api.max_output_tokens,
            "temperature": 0,
            "stream": False,
        }

        # ---- 按provider 追加可选参数
        #
        # **DeepSeek 专属**：思考模式必须关。flash 默认开思考，思考链 token
        # 计入 completion_tokens —— 实测 max_tokens=48 时 48 个 token 全被
        # 思考吃掉，正文一字未出，finish_reason='length'，表现为 bad_json。
        # 其他 OpenAI 兼容服务不认识这个字段，硬发会导致 400。
        if self.api.send_thinking:
            payload["thinking"] = {"type": "disabled"}

        # JSON Output（OpenAI 标准字段）。
        # 官方要求 prompt 含 "json" 字样并给格式样例，PROMPT 已满足。
        if self.api.json_output:
            payload["response_format"] = {"type": "json_object"}

        last_err = ""
        for attempt in range(self.api.max_retries + 1):
            try:
                # **每次真的发出请求就计数**，包含失败与重试。
                # 只在成功时计数会让防失控统计漏掉失败的调用——而失败
                # 恰恰是最容易反复触发、也最该被拦住的情况。
                self.request_count += 1
                resp = self._client.post("/chat/completions", json=payload)
                if resp.status_code == 429:
                    # 限流：退避后重试
                    import time

                    time.sleep(2.0 * (attempt + 1))
                    last_err = "429 rate limited"
                    continue
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage") or {}
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                self.last_prompt_tokens = pt
                self.last_completion_tokens = ct
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct

                raw = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                obj = _extract_json(raw)
                if obj is None:
                    return ModelVerdict(
                        studying=True,
                        comment="解析失败",
                        ok=False,
                        error="bad_json",
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        raw=raw[:200],
                    )
                studying = bool(int(obj.get("s", 0)))
                comment = str(obj.get("c", ""))[:16]
                return ModelVerdict(
                    studying=studying,
                    comment=comment,
                    ok=True,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    raw=raw[:200],
                )
            except Exception as exc:  # 网络/解析类异常统一走重试
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.api.max_retries:
                    import time

                    time.sleep(1.5 * (attempt + 1))

        return ModelVerdict(studying=True, comment="调用失败，暂不打扰", ok=False, error=last_err)

    def usage_summary(self) -> dict[str, int]:
        return {
            "calls": self.total_calls,
            "requests": self.request_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "max_calls": self.max_calls,
        }

    def close(self) -> None:
        # 只关自己建的 client，注入的由调用方负责
        if self._owns_client:
            self._client.close()
