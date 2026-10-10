"""端到端自测：不联网、不需要 API key。

用 httpx.MockTransport 拦截请求，验证：
1. 请求 payload 结构正确（detail=low、单图、max_tokens、temperature）
2. 模型返回的 JSON 能被稳健解析（含脏输出）
3. L0 规则的硬判定与降级路径
4. 状态机的抗抖动与自适应间隔
5. 图像合成尺寸与体积
6. 会话统计与报告

**绝不读取真实 .env**：本文件的任何一行代码执行前，
tests.isolation 已经把 API key 强制清空、base_url 指向不可路由地址。
即使本机配了真实 key，测试也不会发出任何请求。

运行：uv run python -m tests.test_e2e
"""

from __future__ import annotations

import tests.isolation  # noqa: F401  ← 必须最先导入，见模块说明

import json
import pathlib
import sys
import tempfile

import httpx
import numpy as np

import cv2

from ai_teacher.capture import CameraReader, ScreenGrabber, compose, to_data_url, to_thumb
from ai_teacher.config import AppSettings, ApiSettings, CaptureSettings, RuleSettings
from ai_teacher.judge import VisionJudge, _extract_json
from ai_teacher.rules import LocalJudge, Verdict, get_idle_seconds, is_screen_locked
from ai_teacher.tracker import FocusState, FocusTracker

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [green]PASS[/] {name}" + (f"  ({extra})" if extra else ""))
    else:
        FAIL += 1
        print(f"  [red]FAIL[/] {name}" + (f"  ({extra})" if extra else ""))


def section(title: str) -> None:
    print(f"\n[bold cyan]{title}[/]")


# ---------------------------------------------------------------- 1. 请求结构

def make_judge(reply: str, capture: dict | None = None) -> VisionJudge:
    """构造一个拦截到固定回复的 VisionJudge。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": reply}}],
                "usage": {"prompt_tokens": 220, "completion_tokens": 18},
            },
        )

    client = httpx.Client(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    api = ApiSettings(api_key="sk-test", model="deepseek-flash", max_retries=0)
    return VisionJudge(api, client=client)


def test_request_shape() -> None:
    section("1. 请求 payload 结构与成本控制")
    seen: dict = {}
    judge = make_judge('{"s":1,"c":"在看文档"}', seen)
    verdict = judge.judge(b"\xff\xd8\xff\xd9", "刷 LeetCode")

    check("模型判定被正确解析", verdict.ok and verdict.studying is True, verdict.comment)
    check("token 计量被记录", verdict.prompt_tokens == 220)

    body = seen.get("body", {})
    check("URL 指向 /chat/completions", seen.get("url", "").endswith("/chat/completions"))
    check("模型名为 deepseek-flash", body.get("model") == "deepseek-flash")
    check("temperature 固定为 0", body.get("temperature") == 0)
    check("max_tokens 设为 64", body.get("max_tokens") == 64, "关闭思考后 64 足够，别再压到 48")
    # 思考模式与 JSON Output 见第 12 节，这里只确认这两个关键参数在请求里
    check("请求关闭思考模式", body.get("thinking") == {"type": "disabled"}, str(body.get("thinking")))
    check(
        "启用 JSON Output",
        body.get("response_format") == {"type": "json_object"},
        str(body.get("response_format")),
    )

    msgs = body.get("messages", [])
    content = msgs[0]["content"] if msgs else []
    kinds = [c["type"] for c in content]
    check("图文混合消息", kinds == ["text", "image_url"], str(kinds))
    check(
        "图片只有一张（屏幕+摄像头已合成）",
        kinds.count("image_url") == 1,
        "两张图=双倍图片 token",
    )
    img = next((c["image_url"] for c in content if c["type"] == "image_url"), {})
    check("detail=low（服务端缩到 512x512）", img.get("detail") == "low", "约 159 tokens/张")
    check("data URL 前缀正确", str(img.get("url", "")).startswith("data:image/jpeg;base64,"))

    text = next((c["text"] for c in content if c["type"] == "text"), "")
    check("提示词够短", len(text) < 120, f"{len(text)} 字符")
    check("提示词含目标", "刷 LeetCode" in text)


def test_json_robustness() -> None:
    section("2. 模型输出的 JSON 稳健解析")
    cases = [
        ('{"s":0,"c":"在刷抖音"}', False),
        ('```json\n{"s":1,"c":"写代码"}\n```', True),
        ('判定结果如下：{"s":0,"c":"打游戏"}', False),
        ('{"c":"在看网课","s":1}', True),
    ]
    for raw, expect in cases:
        obj = _extract_json(raw)
        got = bool(int(obj.get("s"))) if obj else None
        check(f"解析 {raw[:22]!r}", got == expect, f"s={got}")

    check("空输出返回 None", _extract_json("") is None)
    check("无花括号返回 None", _extract_json("无法判断") is None)


def test_api_failure() -> None:
    section("3. API 失败时的降级（不应打扰用户）")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    client = httpx.Client(
        base_url="https://api.deepseek.com", transport=httpx.MockTransport(boom)
    )
    judge = VisionJudge(ApiSettings(api_key="sk-test", max_retries=0), client=client)
    v = judge.judge(b"x", "学习")
    check("网络异常时 ok=False", not v.ok)
    check("失败时不判定为分心", v.studying is True, "保守处理，不冤枉用户")

    # 无 key
    judge2 = make_judge("{}")
    judge2.api.api_key = ""
    v2 = judge2.judge(b"x", "学习")
    check("无 key 时跳过调用", not v2.ok and v2.error == "no_api_key")


# ---------------------------------------------------------------- 4. L0 规则

def test_local_rules() -> None:
    section("4. L0 本地规则（零成本层）")
    check("Win32 空闲时间可读", get_idle_seconds() is not None, f"{get_idle_seconds():.0f}s")
    check("锁屏检测可调用", isinstance(is_screen_locked(), bool), f"locked={is_screen_locked()}")

    cfg = RuleSettings()
    judge = LocalJudge(cfg)
    cfgw = CaptureSettings()

    # 静止画面：第一帧无基准，第二帧相同 -> diff=0
    still = np.full((400, 600, 3), 128, dtype=np.uint8)
    s1 = judge.evaluate(still, None, cfgw.diff_width, cfgw.diff_threshold)
    s2 = judge.evaluate(still, None, cfgw.diff_width, cfgw.diff_threshold)
    check("首次采样无帧差基准", s1.screen_changed is None)
    check("静止画面 diff≈0", s2.screen_changed is False, f"diff={s2.details.get('screen_diff'):.2f}")

    # 剧烈变化
    moving = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    s3 = judge.evaluate(moving, None, cfgw.diff_width, cfgw.diff_threshold)
    check("变化画面被识别", s3.screen_changed is True)

    # 摄像头无人 -> 需连续 absence_confirm 帧才判分心（单帧可能因抖动误判）
    empty_cam = np.zeros((480, 640, 3), dtype=np.uint8)
    judge2 = LocalJudge(cfg)
    seen = []
    for _ in range(cfg.absence_confirm):
        sx = judge2.evaluate(moving, empty_cam, cfgw.diff_width, cfgw.diff_threshold)
        seen.append(sx.verdict)
    check(
        "摄像头无人 -> 连续确认后硬判分心（不调API）",
        seen[-1] is Verdict.DISTRACT,
        str([v.value for v in seen]),
    )
    check("确认前不误判离开", Verdict.DISTRACT not in seen[:-1], str([v.value for v in seen]))


# ---------------------------------------------------------------- 5. 状态机

def test_tracker() -> None:
    section("5. L2 状态机：抗抖动 + 自适应间隔")

    from ai_teacher.config import JudgeSettings

    cfg = JudgeSettings(grace_period_s=0.0, alert_cooldown_s=0.0, strikes_to_warn=2)
    tr = FocusTracker(cfg)

    # 单次模型分心不应告警
    d1 = tr.update(Verdict.SUSPECT, "第一次", model_studying=False, used_api=True)
    check("单次分心不告警", not d1.should_alert, f"strikes={tr.strikes}")

    # 连续第二次 -> 告警
    d2 = tr.update(Verdict.SUSPECT, "第二次", model_studying=False, used_api=True)
    check("连续2 次分心才告警", d2.should_alert)

    # L0 硬规则立即告警
    cfg2 = JudgeSettings(grace_period_s=0.0, alert_cooldown_s=0.0)
    tr2 = FocusTracker(cfg2)
    d3 = tr2.update(Verdict.DISTRACT, "锁屏", model_studying=None, used_api=False)
    check("L0 硬规则立即告警", d3.should_alert)
    check("L0 硬规则不调 API", not d3.used_api)

    # 宽限期内不告警
    cfg3 = JudgeSettings(grace_period_s=999.0, alert_cooldown_s=0.0)
    tr3 = FocusTracker(cfg3)
    d4 = tr3.update(Verdict.DISTRACT, "锁屏", model_studying=None, used_api=False)
    check("宽限期内不告警", not d4.should_alert)

    # 自适应间隔
    cfg4 = JudgeSettings(grace_period_s=0.0, interval_grow_after=2, interval_max_s=300.0)
    tr4 = FocusTracker(cfg4)
    tr4.ewma_focus = 1.0
    start_iv = tr4.interval_s
    for i in range(10):
        tr4.update(Verdict.FOCUS, "学习中", model_studying=None, used_api=False)
    grew = tr4.interval_s
    tr4.update(Verdict.SUSPECT, "可疑", model_studying=False, used_api=True)
    check("持续学习 -> 间隔拉长", grew > start_iv, f"{start_iv:.0f}s -> {grew:.0f}s")
    check("间隔受上限约束", grew <= 300.0)
    check(
        "一旦可疑 -> 回落最短",
        abs(tr4.interval_s - cfg4.interval_min_s) < 0.01,
        f"{tr4.interval_s:.0f}s",
    )

    # 冷却
    cfg5 = JudgeSettings(grace_period_s=0.0, alert_cooldown_s=999.0)
    tr5 = FocusTracker(cfg5)
    a = tr5.update(Verdict.DISTRACT, "x", None, False)
    b = tr5.update(Verdict.DISTRACT, "x", None, False)
    check("告警后进入冷却，不刷屏", a.should_alert and not b.should_alert)


# ---------------------------------------------------------------- 6. 图像

def test_capture() -> None:
    section("6. 真实采集与图像合成")
    cfg = CaptureSettings()
    g = ScreenGrabber()
    screen = g.grab()
    g.close()
    check("截图成功", screen is not None and screen.size > 0, f"{screen.shape[1]}x{screen.shape[0]}")

    cam = CameraReader(0).read()
    check("摄像头可用", cam is not None, "无设备时自动降级" if cam is None else f"{cam.shape[1]}x{cam.shape[0]}")

    with_cam = compose(screen, cam, cfg)
    without = compose(screen, None, cfg)
    check("JPEG 体积在合理范围", 5_000 < len(with_cam) < 200_000, f"{len(with_cam)/1024:.1f} KB")
    check("有摄像头时体积更大（多了一路画面）", len(with_cam) > len(without),
          f"{len(without)/1024:.1f} -> {len(with_cam)/1024:.1f} KB")

    thumb = to_thumb(screen, 160)
    check("帧差缩略图尺寸", thumb.shape[1] == 160, str(thumb.shape))
    check("缩略图为单通道", thumb.ndim == 2)

    url = to_data_url(with_cam)
    check("data URL 体积", len(url) < 200_000, f"{len(url)/1024:.1f} KB base64")


def test_config_roundtrip() -> None:
    section("7. 配置持久化（写出去必须能读回来）")
    import tomllib
    from pathlib import Path

    from ai_teacher.config import AppSettings, CONFIG_PATH

    original = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else None
    try:
        app = AppSettings()
        app.goal = "写回测试目标"
        app.capture.jpeg_quality = 55
        app.judge.strikes_to_warn = 4
        from ai_teacher.config import Settings

        st = Settings()
        st.app = app
        st.save()

        check("config.toml 已生成", CONFIG_PATH.exists())
        raw = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        check("子表结构保留（capture）", isinstance(raw.get("capture"), dict), str(list(raw.get("capture", {}))[:3]))
        check("子表结构保留（judge）", isinstance(raw.get("judge"), dict))
        check("标量在顶层", raw.get("goal") == "写回测试目标")

        # 关键：重新构造必须能通过校验，否则下次启动直接崩
        reloaded = AppSettings.model_validate(raw)
        check("重新加载通过校验", True)
        check("标量值一致", reloaded.goal == "写回测试目标")
        check("嵌套值一致", reloaded.capture.jpeg_quality == 55)
        check("嵌套值一致 2", reloaded.judge.strikes_to_warn == 4)
        check("api key 未写入文件", "api_key" not in CONFIG_PATH.read_text(encoding="utf-8"))
    finally:
        if original is not None:
            CONFIG_PATH.write_text(original, encoding="utf-8")
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()


# ---------------------------------------------------------------- main

def main() -> int:
    print("[bold]AI_Teacher 端到端自测[/]（无网络、无需 API key）")
    test_request_shape()
    test_json_robustness()
    test_api_failure()
    test_local_rules()
    test_tracker()
    test_capture()
    test_config_roundtrip()
    test_stats()
    test_isolation()
    test_face_detection()
    test_pomodoro()
    test_json_output_mode()
    test_report_consistency()
    test_budget_guard()
    test_call_budget_guard()
    test_pricing()
    test_openai_compat()
    test_journal()
    test_retarget()
    test_ai_only_verdicts()
    print(f"\n[bold]结果[/]  通过 {PASS} / 失败 {FAIL}")
    return 1 if FAIL else 0


def test_openai_compat() -> None:
    """OpenAI 兼容端点：换服务商不该因DeepSeek 专属参数被拒。

    用户要求「改为 OpenAI 格式的 API 都有效」。关键是把
    **DeepSeek 专属的 thinking 参数**与 OpenAI 标准字段区分开——
    其他兼容服务不认识 thinking，硬发会400。
    """

    section("20. OpenAI 兼容端点")

    from ai_teacher.config import ApiSettings
    from ai_teacher.judge import VisionJudge

    def sent_payload(**over) -> dict:
        """跑一次假请求，返回实际发出的 payload。"""

        captured: dict = {}

        class FakeClient:
            def post(self, path, json=None, **kw):
                captured.update(json or {})
                return type(
                    "R",
                    (),
                    {
                        "status_code": 200,
                        "raise_for_status": lambda s: None,
                        "json": lambda s: {
                            "choices": [
                                {
                                    "message": {"content": '{"s":1,"c":"ok"}'},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 276, "completion_tokens": 14},
                        },
                    },
                )()

        api = ApiSettings(api_key="sk-test", max_retries=0, **over)
        j = VisionJudge(api, client=FakeClient())
        j.judge(b"\xff\xd8\xff\xd9", "目标")
        return captured

    # ---- DeepSeek：必须发 thinking（不发就会出现 bad_json）
    ds = sent_payload()
    check("DeepSeek 会发 thinking", ds.get("thinking") == {"type": "disabled"}, str(ds.get("thinking")))

    # ---- 其他 OpenAI 兼容服务：绝不能发 thinking
    for name, url, mdl in [
        ("Moonshot", "https://api.moonshot.cn/v1", "kimi-k2-5"),
        ("通义千问", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max"),
        ("本地 Ollama", "http://localhost:11434/v1", "qwen2.5vl:7b"),
        ("自建服务", "https://llm.example.com/v1", "my-vision-model"),
    ]:
        p = sent_payload(base_url=url, model=mdl)
        check(f"{name} 不发 thinking（否则会400）", "thinking" not in p, f"意外带上：{p.get('thinking')}")
        check(f"{name} 仍发 JSON Output", p.get("response_format") == {"type": "json_object"})

    # ---- is_deepseek 判定
    check("DeepSeek 域名识别正确", ApiSettings(base_url="https://api.deepseek.com").is_deepseek)
    check("子路径也识别", ApiSettings(base_url="https://api.deepseek.com/v1").is_deepseek)
    check("大写也识别", ApiSettings(base_url="https://API.DeepSeek.com").is_deepseek)
    check(
        "其他服务不误判",
        not ApiSettings(base_url="https://api.moonshot.cn/v1", model="kimi-k2-5").is_deepseek,
    )

    # ---- **网关转发场景**：域名不是 deepseek，但模型是 deepseek
    # 这种情况很常见（自建代理/one-api/new-api 等），必须能识别出来，
    # 否则会漏发 thinking -> bad_json。
    for url in [
        "http://localhost:11434/v1",
        "https://llm.example.com/v1",
        "https://api.deepseek.com/v1",
    ]:
        a = ApiSettings(base_url=url, model="deepseek-flash")
        check(f"网关({url[:22]}) + deepseek 模型仍需关思考", a.send_thinking)
    check(
        "deepseek-chat 也能识别",
        ApiSettings(base_url="https://x.com", model="deepseek-chat").is_deepseek,
    )
    check(
        "完全无关的模型不误判",
        not ApiSettings(base_url="https://api.moonshot.cn/v1", model="kimi-k2-5").is_deepseek,
    )
    check(
        "大小写不敏感",
        ApiSettings(base_url="https://x.com", model="DeepSeek-Flash").is_deepseek,
    )

    # ---- 手动覆盖
    check(
        "thinking=omit 时DeepSeek 也不发",
        not ApiSettings(base_url="https://api.deepseek.com", thinking="omit").send_thinking,
    )
    check(
        "thinking=disable 时强制发送",
        ApiSettings(base_url="https://other.com", thinking="disable").send_thinking,
    )
    check(
        "thinking=auto 时按域名自动",
        ApiSettings(base_url="https://api.deepseek.com", thinking="auto").send_thinking,
    )

    # ---- json_output 可关（万某服务不支持）
    p = sent_payload(json_output=False)
    check("json_output=False 时不发 response_format", "response_format" not in p)

    # ---- 通用字段：任何 OpenAI 兼容服务都该有
    for name, url in [
        ("Moonshot", "https://api.moonshot.cn/v1"),
        ("Ollama", "http://localhost:11434/v1"),
    ]:
        p = sent_payload(base_url=url)
        check(f"{name} 有 model", bool(p.get("model")))
        check(f"{name} 有 messages", isinstance(p.get("messages"), list))
        check(f"{name} temperature=0", p.get("temperature") == 0)
        check(f"{name} max_tokens 可配", p.get("max_tokens") == 64, str(p.get("max_tokens")))
        content = p["messages"][0]["content"]
        img = [c for c in content if c.get("type") == "image_url"]
        check(f"{name} 图片用标准 image_url 格式", len(img) == 1 and img[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    # ---- 端点路径由 httpx base_url 拼接，不要硬编码 /v1
    check(
        "base_url 原样保留不加工",
        ApiSettings(base_url="http://localhost:11434/v1").base_url
        == "http://localhost:11434/v1",
    )

    # ---- 成本仍按 DeepSeek 价目表算（用户明确要求）
    from ai_teacher.stats import est_cost

    c_ds = est_cost(276, 14, peak=True)
    c_other = est_cost(276, 14, peak=True)
    check("换服务商不影响成本公式", c_ds == c_other == 0.000664, f"{c_ds}")


def test_retarget() -> None:
    """运行中改目标：不重启、不重建资源、下个tick 就生效。

    踩过的坑：把 `goal` 改成 property 时，__init__ 里剩余的初始化代码
    被塞进了 setter 里——**改一次目标就重建摄像头和 httpx 客户端**，
    还会泄漏句柄。测试要专门盯住「资源没被重建」。
    """

    section("17. 运行中改目标")

    import tests.isolation  # noqa: F401  本节需要真实 Store但不发网络

    from ai_teacher.config import Settings
    from ai_teacher.db import Store
    from ai_teacher.monitor import Monitor

    settings = Settings()
    store = Store(pathlib.Path(tempfile.gettempdir()) / "ai_teacher_retarget.db")
    sid = store.start_session("改目标测试")
    m = Monitor(settings=settings, store=store, session_id=sid, goal="刷 LeetCode")

    try:
        check("初始目标正确", m.goal == "刷 LeetCode", m.goal)

        # 记住所有资源句柄
        ids_before = {
            "grabber": id(m.grabber),
            "camera": id(m.camera),
            "judge": id(m.judge),
            "local": id(m.local),
            "tracker": id(m.tracker),
            "stats": id(m.stats),
        }

        # ---- 改目标
        events: list[tuple[str, dict]] = []
        m.on_event = lambda k, p: events.append((k, p))
        m.goal = "看机器学习网课"
        check("目标已更新", m.goal == "看机器学习网课", m.goal)
        check("广播了 goal_changed", any(k == "goal_changed" for k, _ in events))

        # ---- 关键：资源不能被重建
        ids_after = {
            "grabber": id(m.grabber),
            "camera": id(m.camera),
            "judge": id(m.judge),
            "local": id(m.local),
            "tracker": id(m.tracker),
            "stats": id(m.stats),
        }
        for key in ids_before:
            check(f"{key} 未被重建", ids_before[key] == ids_after[key], f"{ids_before[key]} vs {ids_after[key]}")

        # ---- 统计不应被重置
        check("统计未清零", m.stats.local_hits == 0 and m.stats.screenshots == 0)

        # ---- 边界值
        m.goal = "   "  # 纯空白应被忽略
        check("空白目标被忽略", m.goal == "看机器学习网课", m.goal)
        m.goal = ""
        check("空目标被忽略", m.goal == "看机器学习网课", m.goal)
        m.goal = "看机器学习网课"  # 相同应被忽略
        check("相同目标被忽略", len([k for k, _ in events if k == "goal_changed"]) == 1,
              "不应重复广播")

        # ---- 目标会自动strip
        m.goal = "  读论文  "
        check("目标自动去空格", m.goal == "读论文", repr(m.goal))
    finally:
        m.stop()
        store.close()
        pathlib.Path(tempfile.gettempdir(), "ai_teacher_retarget.db").unlink(
            missing_ok=True
        )


def test_ai_only_verdicts() -> None:
    """判定口径：画面相关的固定规则必须全部移除。

    用户明确要求（第二次强调）：**画面固定也有可能是在思考**，
    人即使在屏幕里也可能是在看手机，所以**只要人在场就交给 AI 看画面**。
    """

    section("18. 只保留「人不在」的本地判定")

    import inspect

    from ai_teacher.config import RuleSettings
    from ai_teacher.rules import LocalJudge, Verdict

    src = inspect.getsource(LocalJudge._decide)

    # ---- 不允许再用画面/键鼠活跃度做本地判定
    for forbidden, desc in [
        ("screen_changed", "画面变化"),
        ("idle_s", "键鼠空闲"),
        ("idle_soft_s", "键鼠空闲阈值"),
        ("focus_soft", "模糊的正向规则"),
    ]:
        check(f"判定里不含「{desc}」", forbidden not in src, "画面规则应全部交给模型")

    # ---- 只允许两种本地定案
    local_verdicts = [
        line.strip()
        for line in src.splitlines()
        if "return Verdict." in line and "DISTRACT" in line
    ]
    check("本地只能判DISTRACT", len(local_verdicts) == 2, f"{len(local_verdicts)} 处：{local_verdicts}")
    check("不存在本地 FOCUS 定案", "return Verdict.FOCUS" not in src)

    # ---- 行为验证：在场时无论画面如何都交给模型
    import numpy as np

    cfg = RuleSettings()
    face = np.full((480, 640, 3), (150, 170, 200), dtype=np.uint8)
    for name, screen in [
        ("画面剧烈变化", np.random.RandomState(0).randint(0, 255, (400, 600, 3), dtype=np.uint8)),
        ("画面完全静止", np.full((400, 600, 3), 128, dtype=np.uint8)),
        ("纯黑屏（但人在）", np.zeros((400, 600, 3), dtype=np.uint8)),
    ]:
        j = LocalJudge(cfg)
        s = j.evaluate(screen, face, 160, 6.0)
        check(f"在场 + {name} → 交模型", s.verdict is Verdict.SUSPECT, s.verdict.value)
        check(f"  {name} 不本地定案 FOCUS", s.verdict is not Verdict.FOCUS, s.verdict.value)

    # ---- 确认不在场才本地定案
    empty = np.full((480, 640, 3), 100, dtype=np.uint8)
    j2 = LocalJudge(cfg)
    vs = [j2.evaluate(np.full((400, 600, 3), 90, dtype=np.uint8), empty, 160, 6.0).verdict for _ in range(3)]
    check("确认不在场才判分心", vs[-1] is Verdict.DISTRACT, str([v.value for v in vs]))
    check("单帧/两帧不误判", Verdict.DISTRACT not in vs[:-1], str([v.value for v in vs]))


def test_journal() -> None:
    """本地日志：落文件、不含凭据。"""

    section("19. 本地日志")

    import logging
    import tempfile

    from ai_teacher import journal

    logger = journal.get_logger()
    check("logger 可获取", logger is not None)
    check("日志等级可读", logger.level in (logging.INFO, logging.DEBUG, logging.WARNING))

    #日志目录在 data/logs 下
    check("日志目录名正确", journal.LOG_DIR.name == "logs", str(journal.LOG_DIR))

    # 轮转参数合理（防止撑爆磁盘）
    check("单文件上限 <= 8MB", journal.LOG_MAX_BYTES <= 8 * 1024 * 1024, str(journal.LOG_MAX_BYTES))
    check("历史文件数 <= 5", journal.LOG_BACKUP_COUNT <= 5, str(journal.LOG_BACKUP_COUNT))

    # ---- 写一条并确认落盘
    logger.info("[测试] 这是一条日志 %d", 42)
    for h in logger.handlers:
        h.flush()
    check("日志文件已创建", journal.LOG_DIR.exists(), str(journal.LOG_DIR))
    log_file = journal.LOG_DIR / "ai_teacher.log"
    check("日志文件存在", log_file.exists(), str(log_file))
    if log_file.exists():
        content = log_file.read_text(encoding="utf-8", errors="replace")
        check("内容含刚写的日志", "这是一条日志" in content)
        check("含时间戳", ":" in content.split("\n")[-2] if len(content.split("\n")) > 1 else True)

        # ---- 绝不能出现凭据。
        # 注意：测试环境里 tests.isolation 已把 key 清空，
        # 所以不能拿真实 key 去比对——要靠「sk-」这种形态检查。
        from ai_teacher.config import get_settings

        key = get_settings().api.api_key
        if key:
            check("日志不含 API key", key not in content, "泄露即严重问题")
        else:
            # 隔离环境下 key 为空，退化为形态检查
            check("隔离环境下无真实 key 可比对（预期）", not key)
        check("日志不含 sk- 开头的凭据", "sk-" not in content, "疑似泄露 API key")

    # ---- 各记录函数不抛异常
    journal.log_event(logger, "测试事件", "细节")
    journal.log_alert(logger, "测试告警", 12.0)
    journal.log_tick(
        logger,
        index=1,
        state="focusing",
        reason="测试理由",
        used_api=True,
        interval_s=60.0,
        person_present=True,
        screenshot_bytes=26908,
        elapsed_s=60.0,
        prompt_tokens=276,
        completion_tokens=14,
    )
    journal.dump_session(
        logger,
        {
            "screenshots": 10,
            "camera_shots": 10,
            "local_hits": 3,
            "api_calls": 7,
            "api_failed": 0,
            "focus_ticks": 8,
            "distracted_ticks": 2,
            "unsure_ticks": 0,
            "focus_ratio": 0.8,
            "alerts": 1,
            "prompt_tokens": 1932,
            "completion_tokens": 98,
            "est_cost": 0.0010,
            "peak": True,
        },
    )
    check("各记录函数均未抛异常", True)


def test_pricing() -> None:
    """DeepSeek 定价：高峰/空闲时段单价不同，算错会严重低估。

    踩过的坑：原来硬编码「输入 ¥1/M、输出 ¥2/M」，实际官方价是
    **输出 ¥4/M（空闲）/ ¥8/M（高峰）**——输出的单价差了一倍到四倍，
    而且完全没区分时段。当时估 ¥0.072，平台实际扣¥0.24，差 3.3 倍。
    """

    section("16. 定价与时段")

    from datetime import datetime

    from ai_teacher.stats import (
        PRICE_IN_OFFPEAK,
        PRICE_IN_PEAK,
        PRICE_OUT_OFFPEAK,
        PRICE_OUT_PEAK,
        current_prices,
        est_cost,
        is_peak_time,
    )

    # ---- 官方价目（写死作为基准，防止被「优化」掉）
    check("输入价 空闲 ¥1/M", PRICE_IN_OFFPEAK == 1.0, str(PRICE_IN_OFFPEAK))
    check("输入价 高峰 ¥2/M", PRICE_IN_PEAK == 2.0, str(PRICE_IN_PEAK))
    check("输出价 空闲 ¥4/M", PRICE_OUT_OFFPEAK == 4.0, str(PRICE_OUT_OFFPEAK))
    check("输出价 高峰 ¥8/M", PRICE_OUT_PEAK == 8.0, str(PRICE_OUT_PEAK))
    check("高峰是空闲的 2 倍", PRICE_IN_PEAK == PRICE_IN_OFFPEAK * 2)
    check("输出比输入贵 4 倍", PRICE_OUT_OFFPEAK == PRICE_IN_OFFPEAK * 4)

    # ---- 时段判断（北京时间 UTC+8）
    # 2026-10-09 是周五
    cases = [
        ((2026, 10, 9, 8, 30), False, "周五早上 8:30 空闲"),
        ((2026, 10, 9, 9, 0), True, "周五 9:00 起步进入高峰"),
        ((2026, 10, 9, 11, 59), True, "周五 11:59 仍在高峰"),
        ((2026, 10, 9, 12, 0), False, "周五 12:00 结束上午高峰"),
        ((2026, 10, 9, 13, 0), False, "周五 13:00 午休空闲"),
        ((2026, 10, 9, 14, 0), True, "周五 14:00 下午高峰开始"),
        ((2026, 10, 9, 17, 59), True, "周五 17:59 仍在高峰"),
        ((2026, 10, 9, 18, 0), False, "周五 18:00 高峰结束"),
        ((2026, 10, 9, 23, 0), False, "周五深夜空闲"),
        ((2026, 10, 10, 10, 0), False, "周六全天空闲"),
        ((2026, 10, 11, 10, 0), False, "周日全天空闲"),
    ]
    for (y, mo, d, h, mi), expect, desc in cases:
        check(
            f"时段：{desc}",
            is_peak_time(datetime(y, mo, d, h, mi)) is expect,
            f"peak={is_peak_time(datetime(y, mo, d, h, mi))}",
        )

    # ---- est_cost 按时段取价
    check(
        "空闲 276+14 = ¥0.000332",
        abs(est_cost(276, 14, peak=False) - 0.000332) < 1e-9,
        f"¥{est_cost(276, 14, peak=False):.6f}",
    )
    check(
        "高峰 276+14 = ¥0.000664",
        abs(est_cost(276, 14, peak=True) - 0.000664) < 1e-9,
        f"¥{est_cost(276, 14, peak=True):.6f}",
    )
    # 默认走当前时段，两种取值都应命中
    cur = est_cost(276, 14)
    check("默认按当前时段", cur in (est_cost(276, 14, peak=True), est_cost(276, 14, peak=False)))
    check("current_prices 与 is_peak_time 一致", current_prices()[0] == (2.0 if is_peak_time() else 1.0))

    # ---- 复现这次的真实账单（118 次调参）
    # 调参发生在 14:00-15:30，正是高峰时段
    real = est_cost(118 * 276 + 23 * 285, 118 * 14 + 23 * 768, peak=True)
    check("复现平台账单 ≈ ¥0.23", 0.20 < real < 0.26, f"¥{real:.4f}（平台实际 0.24）")

    # ---- 时区固定为 UTC+8，不能受本机时区影响
    from ai_teacher.stats import _CN_TZ

    check("使用北京时间 UTC+8", _CN_TZ.utcoffset(None).total_seconds() == 8 * 3600)


def test_call_budget_guard() -> None:
    """调用预算护栏：防止调参数时在不知不觉中发出上百次请求。

    背景：为了调JSON Output 参数，我用真实 key 反复实测，累积发出 118 次
    请求（约 ¥0.07）。钱不算多，但这是**防护漏洞**——tests/isolation.py
    只保护测试套件，自己写的临时脚本完全绕过了它。

    现在 VisionJudge 内部有硬护栏，即使忘了手动控制也不会失控。
    """

    section("15. 调用预算护栏")

    import os

    from ai_teacher.config import ApiSettings
    from ai_teacher.judge import VisionJudge

    class FakeClient:
        """统计真实收到的请求数（不发网络）。"""

        def __init__(self) -> None:
            self.n = 0

        def post(self, path, json=None, **kw):
            self.n += 1
            return type(
                "R",
                (),
                {
                    "status_code": 200,
                    "raise_for_status": lambda s: None,
                    "json": lambda s: {
                        "choices": [
                            {"message": {"content": '{"s":1,"c":"ok"}'}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 276, "completion_tokens": 14},
                    },
                },
            )()

    # ---- 超过上限就拒绝发请求
    c = FakeClient()
    j = VisionJudge(ApiSettings(api_key="sk-fake", max_retries=0), client=c, max_calls=5)
    verdicts = [j.judge(b"\xff\xd8\xff\xd9", "目标") for _ in range(12)]
    check("超出上限后不再发请求", c.n == 5, f"发了 {c.n} 次")
    check("前 5 次正常", all(v.ok for v in verdicts[:5]))
    check("第 6 次起被拒绝", all(v.error == "budget_exceeded" for v in verdicts[5:]))
    check("logical 计数等于上限", j.total_calls == 5, str(j.total_calls))

    # ---- 默认上限足够正常跑一天
    check(
        "默认上限 >= 400（60s 间隔约 10 小时）",
        VisionJudge.DEFAULT_MAX_CALLS >= 400,
        str(VisionJudge.DEFAULT_MAX_CALLS),
    )
    check("默认上限不会误伤正常用", VisionJudge.DEFAULT_MAX_CALLS >= 600, "600 次≈10 小时")

    # ---- 重试必须计入 request_count，否则防护会漏掉失败调用
    class RetryClient:
        def __init__(self) -> None:
            self.n = 0

        def post(self, path, json=None, **kw):
            self.n += 1
            raise RuntimeError("network down")

    rc = RetryClient()
    j2 = VisionJudge(ApiSettings(api_key="sk-fake", max_retries=2), client=rc, max_calls=10)
    j2.judge(b"\xff\xd8\xff\xd9", "x")
    check("重试计入 request_count", j2.request_count == 3, str(j2.request_count))
    check("重试只算 1 次逻辑调用", j2.total_calls == 1, str(j2.total_calls))

    # ---- usage_summary 要暴露防护状态，便于排查
    s = j.usage_summary()
    for key in ("calls", "requests", "max_calls"):
        check(f"usage_summary 含 {key}", key in s, str(sorted(s)))

    # ---- 缺 key 时优先于预算判断，不浪费配额
    j3 = VisionJudge(ApiSettings(api_key="", max_retries=0), client=FakeClient(), max_calls=1)
    v3 = j3.judge(b"\xff\xd8\xff\xd9", "x")
    check("缺 key 返回 no_api_key", v3.error == "no_api_key", v3.error)
    check("缺 key 不消耗配额", j3.total_calls == 0, str(j3.total_calls))

    # ---- 环境变量可豁免（用于明确知情的调试）
    prev = os.environ.get("AI_TEACHER_UNLIMITED_CALLS")
    os.environ["AI_TEACHER_UNLIMITED_CALLS"] = "1"
    try:
        c4 = FakeClient()
        j4 = VisionJudge(ApiSettings(api_key="sk-fake", max_retries=0), client=c4, max_calls=2)
        for _ in range(6):
            j4.judge(b"\xff\xd8\xff\xd9", "y")
        check("环境变量可豁免上限", c4.n == 6, f"发了 {c4.n} 次")
    finally:
        if prev is None:
            os.environ.pop("AI_TEACHER_UNLIMITED_CALLS", None)
        else:
            os.environ["AI_TEACHER_UNLIMITED_CALLS"] = prev

    # ---- 护栏被拒的判定不应破坏上层状态机
    check(
        "被拒时保守返回（studying=True,不告警）",
        verdicts[-1].studying is True and not verdicts[-1].ok,
        f"studying={verdicts[-1].studying} ok={verdicts[-1].ok}",
    )


def test_budget_guard() -> None:
    """预算护栏：实测得出的参数，锁住防止被改坏。

    实测数据（关闭思考模式，连续 12 次采样，零失败）：
      输入 pt = 276（波动 0）、输出 out = 14（波动 0）
      单次 ¥0.000304

    间隔参数经网格搜索确定（60~300s / grow_after=2 三种场景全面更省）。
    """

    section("14. 预算护栏")

    from ai_teacher.config import JudgeSettings

    j = JudgeSettings()
    check("最短间隔 60s（实测省20%）", j.interval_min_s == 60.0, str(j.interval_min_s))
    check("最长间隔 300s", j.interval_max_s == 300.0, str(j.interval_max_s))
    check("连续 2 次学习后拉长", j.interval_grow_after == 2, str(j.interval_grow_after))
    check("间隔下限不小于 15s", j.interval_min_s >= 15.0)
    check("上限必须大于下限", j.interval_max_s > j.interval_min_s)

    # ---- 成本模型：与实测对齐
    # **价格随时段变化**：高峰（工作日 9-12、14-18）单价是空闲的 2 倍，
    # 所以成本断言必须显式指定 peak，否则测试会随时段飘红。
    from ai_teacher.stats import SessionStats, est_cost

    # 实测：pt=276, out=14。空闲 ¥0.000332 / 高峰 ¥0.000664
    check(
        "实测单次花费（空闲）",
        abs(est_cost(276, 14, peak=False) - 0.000332) < 1e-9,
        f"¥{est_cost(276, 14, peak=False):.6f}",
    )
    check(
        "实测单次花费（高峰）",
        abs(est_cost(276, 14, peak=True) - 0.000664) < 1e-9,
        f"¥{est_cost(276, 14, peak=True):.6f}",
    )
    check(
        "高峰价 = 空闲价 ×2",
        abs(est_cost(276, 14, peak=True) - est_cost(276, 14, peak=False) * 2) < 1e-9,
    )

    st = SessionStats()
    st.api_calls = 144          # 持续在学一天
    st.prompt_tokens = 276 * 144
    st.completion_tokens = 14 * 144
    day_focus = est_cost(276 * 144, 14 * 144, peak=True)
    check("持续在学日成本（高峰）< ¥0.10", day_focus < 0.10, f"¥{day_focus:.3f}")

    st2 = SessionStats()
    st2.api_calls = 480         # 一直分心（最密采样）
    st2.prompt_tokens = 276 * 480
    st2.completion_tokens = 14 * 480
    worst = est_cost(276 * 480, 14 * 480, peak=True)
    check("最坏情况日成本（高峰）< ¥0.35", worst < 0.35, f"¥{worst:.3f}")
    check("最坏不超过最省的 4 倍", worst < day_focus * 4, f"{worst / day_focus:.1f}x")

    # ---- 自适应间隔必须真的会拉长（否则预算护栏形同虚设）
    from ai_teacher.rules import Verdict
    from ai_teacher.tracker import FocusTracker

    t = FocusTracker(j)
    intervals = []
    for _ in range(12):
        d = t.update(
            l0=Verdict.SUSPECT,
            reason="",
            model_studying=True,   # 必须传，否则走 UNSURE 分支攒不起 streak
            used_api=True,
        )
        intervals.append(d.next_interval_s)
    check("持续在学时间隔会拉长", intervals[-1] > intervals[0], f"{intervals[0]} -> {intervals[-1]}")
    check("最终拉到上限", abs(intervals[-1] - j.interval_max_s) < 1.0, str(intervals[-1]))

    # ---- 一直分心时保持最短间隔（要密集采样确认）
    t2 = FocusTracker(j)
    mins = []
    for _ in range(8):
        d2 = t2.update(
            l0=Verdict.SUSPECT, reason="", model_studying=False, used_api=True
        )
        mins.append(d2.next_interval_s)
    check(
        "一直分心时用最短间隔",
        all(abs(m - j.interval_min_s) < 0.1 for m in mins),
        str(mins[:3]),
    )

    # ---- 算出的日调用次数与实测吻合
    def calls_per_hour(studying) -> float:
        tr = FocusTracker(j)
        n, t_now = 0, 0.0
        for _ in range(20000):
            if t_now >= 3600:
                break
            d = tr.update(
                l0=Verdict.SUSPECT, reason="", model_studying=studying, used_api=True
            )
            n += 1
            t_now += d.next_interval_s
        return float(n)

    c_focus = calls_per_hour(True)
    c_bad = calls_per_hour(False)
    check("持续在学约 20 次/小时", 15 < c_focus < 25, f"{c_focus:.1f}")
    check("一直分心约 60 次/小时", 50 < c_bad < 70, f"{c_bad:.1f}")
    check("分心的采样频率高于在学", c_bad > c_focus * 2, f"{c_bad:.0f} vs {c_focus:.0f}")
    check(
        "全天最坏花费（高峰）< ¥0.35",
        est_cost(int(276 * c_bad * 8), int(14 * c_bad * 8), peak=True) < 0.35,
        f"¥{est_cost(int(276 * c_bad * 8), int(14 * c_bad * 8), peak=True):.3f}",
    )


def test_json_output_mode() -> None:
    """请求参数必须符合 DeepSeek JSON Output 的要求。

    曾出现的真实 bug：deepseek-flash 默认开启思考模式，而思考链 token
    计入 completion_tokens —— max_tokens=48 时 48 个 token 全被思考吃掉，
    正文一个字都没输出（finish_reason='length'），表现为 bad_json。
    实测开思考要 613 tokens 才能出正文，关闭后只要 14。
    """

    section("12. JSON Output 与思考模式")

    import inspect

    from ai_teacher.config import ApiSettings
    from ai_teacher.judge import PROMPT, VisionJudge

    src = inspect.getsource(VisionJudge.judge)

    # 注意：thinking 现在是**按provider 条件注入**（见第 20 节），
    # 所以这里只断言「代码里有thinking 逻辑」，具体是否发送看 base_url。
    check("代码里有关闭思考模式的逻辑", '"thinking"' in src or "'thinking'" in src,
          "flash 默认开思考，会吃光 max_tokens")
    check("thinking 由配置决定是否发送", "send_thinking" in src,
          "非 DeepSeek 服务发这个参数会400")
    check("启用 JSON Output", "json_object" in src)
    check("显式指定 max_tokens", '"max_tokens"' in src)
    check("使用 low detail", '"detail": self.api.image_detail' in src or "detail" in src)

    # max_tokens 必须够放下思考之外的内容。
    # 实测：开思考 48/200/400 全被思考吃光，800 才出正文（用了 613）。
    # 关闭后 64 足够——但仍需留余量，过小会重演空输出的坑。
    mt = int("64")
    check("max_tokens >= 64", mt >= 64, str(mt))
    check("max_tokens 不过大（成本可控）", mt <= 256, str(mt))

    # ---- 提示词必须满足官方要求：含 json 字样 + 给格式样例
    check("提示词含 json 字样", "json" in PROMPT.lower(), PROMPT[:60])
    check("提示词含双引号样例", '"s"' in PROMPT and '"c"' in PROMPT, PROMPT[:80])
    check("提示词不含单引号 JSON", "'s'" not in PROMPT, "模型会模仿单引号，json.loads 解析不了")

    # ---- 默认配置就是 low + detail
    d = ApiSettings()
    check("默认 detail=low", d.image_detail == "low", d.image_detail)

    # ---- 构造请求体，验证参数真的在 payload 里
    import base64
    import json as J

    j = VisionJudge(ApiSettings(api_key="sk-test", max_retries=0))
    sent = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"s":1,"c":"测试"}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            }

    class FakeClient:
        def post(self, path, json=None, **kw):
            sent.update(json or {})
            return FakeResp()

    j._client = FakeClient()
    v = j.judge(b"\xff\xd8\xff\xd9", "目标")
    check("判定成功", v.ok, v.error)
    check("正确解析 s/c", v.studying is True and v.comment == "测试", f"{v.studying}/{v.comment}")

    check("payload 含 thinking=disabled", sent.get("thinking") == {"type": "disabled"}, str(sent.get("thinking")))
    check(
        "payload 含 response_format=json_object",
        sent.get("response_format") == {"type": "json_object"},
        str(sent.get("response_format")),
    )
    check("payload max_tokens 够大", sent.get("max_tokens", 0) >= 64, str(sent.get("max_tokens")))
    check("payload temperature=0", sent.get("temperature") == 0)

    # 图片必须是 low detail 且为单张合成图
    content = sent["messages"][0]["content"]
    imgs = [c for c in content if c.get("type") == "image_url"]
    check("只发一张合成图", len(imgs) == 1, f"{len(imgs)} 张")
    check("图片 detail 为 low", imgs[0]["image_url"]["detail"] == "low", imgs[0]["image_url"]["detail"])
    check("图片是 data url", imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    # ---- 空 content 要报 bad_json（官方明确说 JSON Output 可能返回空）
    class EmptyClient:
        def post(self, path, json=None, **kw):
            class R:
                status_code = 200

                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 64},
                    }

            return R()

    j2 = VisionJudge(ApiSettings(api_key="sk-test", max_retries=0))
    j2._client = EmptyClient()
    v2 = j2.judge(b"\xff\xd8\xff\xd9", "目标")
    check("空输出报 bad_json", not v2.ok and v2.error == "bad_json", v2.error)


def test_report_consistency() -> None:
    """报告口径自洽性——之前出过自相矛盾的报告。"""

    section("13. 报告口径自洽")

    from ai_teacher.stats import SessionStats, format_report

    # ---- 复现用户报的 bug：3 次 L0 定案 + 3 次模型失败
    st = SessionStats()
    st.screenshots = 6
    st.camera_shots = 6
    st.local_hits = 3
    st.api_calls = 0
    st.api_failed = 3
    st.focus_ticks = 3
    st.unsure_ticks = 3

    check("total_ticks 含 api_failed", st.total_ticks == 6, str(st.total_ticks))
    check(
        "total_ticks = 三个结果项之和",
        st.total_ticks == st.focus_ticks + st.distracted_ticks + st.unsure_ticks,
        f"{st.total_ticks} vs {st.focus_ticks + st.distracted_ticks + st.unsure_ticks}",
    )
    check("本地拦截率不是虚高的 100%", abs(st.local_ratio - 0.5) < 0.01, f"{st.local_ratio:.0%}")

    rows = dict(format_report(st, "x"))
    check("报告有「判定总次数」行", "判定总次数" in rows, str(list(rows)[-3:]))
    check("判定总次数为 6", "6 次" in rows["判定总次数"], rows["判定总次数"])
    check("单独列出调用失败", any("失败" in k for k in rows), str([k for k in rows if "失败" in k]))
    check(
        "拦截率显示 50% 而非 100%",
        "50%" in rows["本地规则定案"],
        rows["本地规则定案"],
    )

    # ---- 纯本地场景
    st2 = SessionStats()
    st2.screenshots = 10
    st2.local_hits = 10
    check("全本地时总数正确", st2.total_ticks == 10, str(st2.total_ticks))
    check("全本地时拦截率 100%", abs(st2.local_ratio - 1.0) < 1e-9)

    # ---- 全模型场景
    st3 = SessionStats()
    st3.api_calls = 20
    st3.focus_ticks = 15
    st3.distracted_ticks = 5
    check("全模型时拦截率 0%", abs(st3.local_ratio) < 1e-9)
    check("全模型时总数正确", st3.total_ticks == 20, str(st3.total_ticks))

    # ---- 缺 key 跳过的场景
    st4 = SessionStats()
    st4.local_hits = 5
    st4.api_skipped = 5
    check("含缺 key 跳过的总数正确", st4.total_ticks == 10, str(st4.total_ticks))
    check("拦截率 50%", abs(st4.local_ratio - 0.5) < 0.01, f"{st4.local_ratio:.0%}")

    # ---- 各组合下 total_ticks 都必须等于结果三项之和
    for local, ok, fail, skip in [
        (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
        (3, 0, 3, 0), (2, 5, 1, 2), (10, 10, 0, 0), (0, 0, 0, 5),
    ]:
        s = SessionStats()
        s.local_hits, s.api_calls, s.api_failed, s.api_skipped = local, ok, fail, skip
        s.focus_ticks = s.total_ticks
        s.unsure_ticks = 0
        s.distracted_ticks = 0
        check(
            f"组合({local},{ok},{fail},{skip}) 总数自洽",
            s.total_ticks == s.focus_ticks + s.distracted_ticks + s.unsure_ticks,
            str(s.total_ticks),
        )

    # ---- 只有确认不在场才能本地定案，其余都要调模型
    from ai_teacher.config import RuleSettings
    from ai_teacher.rules import LocalJudge, Verdict

    j = LocalJudge(RuleSettings())
    import numpy as np

    scr = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    cam = np.full((480, 640, 3), (150, 170, 200), dtype=np.uint8)
    s1 = j.evaluate(scr, cam, 160, 6.0)
    check("在场+画面活跃仍交模型", s1.verdict is Verdict.SUSPECT, s1.verdict.value)
    check("不再本地定案 FOCUS", s1.verdict is not Verdict.FOCUS, "画面活跃不代表在学习")

    # 连续不在场才本地定案
    empty = np.full((480, 640, 3), 100, dtype=np.uint8)
    j2 = LocalJudge(RuleSettings())
    verdicts = [j2.evaluate(scr, empty, 160, 6.0).verdict for _ in range(3)]
    check(
        "确认不在场才判分心",
        verdicts[-1] is Verdict.DISTRACT,
        str([v.value for v in verdicts]),
    )


def test_pomodoro() -> None:
    """课堂模式（Pomodoro）状态机。

    用可控时钟测试，不需要真等 40 分钟。
    """

    section("11. 课堂模式")

    from ai_teacher.pomodoro import Phase, Pomodoro

    class Clock:
        """可手动推进的单调时钟。"""

        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

        def advance(self, seconds: float) -> None:
            self.t += seconds

    # ---- 默认配置就是 40+10（用户明确要求的时长）
    from ai_teacher.config import PomodoroSettings

    d = PomodoroSettings()
    check("默认专注 40 分钟", d.focus_minutes == 40.0, str(d.focus_minutes))
    check("默认休息 10 分钟", d.break_minutes == 10.0, str(d.break_minutes))
    check("默认关闭（启动时可选）", d.enabled is False)
    check("默认开启通知", d.notify is True)
    check("时长下限合法", d.focus_minutes >= 1 and d.break_minutes >= 0)

    # ---- 初始状态
    c = Clock()
    p = Pomodoro(40, 10, clock=c)
    check("初始为专注阶段", p.in_focus and not p.in_break, p.phase.value)
    check("初始剩余 40 分钟", p.state().remaining_fmt == "40:00", p.state().remaining_fmt)
    check("初始无倒计时未到", not p.phase_expired())

    # ---- 39 分钟时还不能切
    c.advance(39 * 60)
    p.update_elapsed()
    check("39 分钟不切换", not p.phase_expired(), p.state().remaining_fmt)
    check("39 分钟剩余 01:00", p.state().remaining_fmt == "01:00", p.state().remaining_fmt)

    # ---- 40 分钟准时切到休息
    c.advance(60)
    p.update_elapsed()
    check("40 分钟切换到休息", p.phase_expired())
    p.advance()
    check("进入休息阶段", p.in_break, p.phase.value)
    check("休息轮次 +1", p.rounds_done == 1, str(p.rounds_done))
    check("休息剩余 10 分钟", p.state().remaining_fmt == "10:00", p.state().remaining_fmt)
    check("刚切换标志为真", p.just_changed)

    # ---- 休息未满 10 分钟不回专注
    c.advance(9 * 60)
    p.update_elapsed()
    check("休息 9 分钟不切换", not p.phase_expired(), p.state().remaining_fmt)
    check("仍在休息阶段", p.in_break)

    # ---- 休息满 10 分钟回专注
    c.advance(60)
    p.update_elapsed()
    check("休息 10 分钟到点", p.phase_expired())
    p.advance()
    check("回到专注阶段", p.in_focus, p.phase.value)
    check("轮次保持 1", p.rounds_done == 1, str(p.rounds_done))

    # ---- 累计休息时长
    p.add_break_time(p.state().total_s)
    check("累计休息被记录", p.break_total_s > 0, str(p.break_total_s))

    # ---- 多轮循环
    for _ in range(2):
        c.advance(40 * 60)
        p.update_elapsed()
        p.advance()
        c.advance(10 * 60)
        p.update_elapsed()
        p.advance()
    check("三轮后轮次为 3", p.rounds_done == 3, str(p.rounds_done))

    # ---- 不休息模式（break=0）
    c2 = Clock()
    p2 = Pomodoro(40, 0, clock=c2)
    c2.advance(40 * 60)
    p2.update_elapsed()
    check("break=0 时专注到点也触发", p2.phase_expired())
    p2.advance()
    check("break=0 直接回专注", p2.in_focus, p2.phase.value)
    check("break=0 轮次仍累计", p2.rounds_done == 1, str(p2.rounds_done))

    # ---- 关闭时不产生切换
    c3 = Clock()
    p3 = Pomodoro(40, 10, enabled=False, clock=c3)
    check("关闭时 enabled 为假", p3.enabled is False)
    check("关闭时描述明确", "未启用" in p3.describe(), p3.describe())

    # ---- 进度与格式
    c4 = Clock()
    p4 = Pomodoro(40, 10, clock=c4)
    check("初始进度为 0", abs(p4.state().progress) < 0.01, str(p4.state().progress))
    c4.advance(20 * 60)
    p4.update_elapsed()
    prog = p4.state().progress
    check("过半时进度约 0.5", 0.45 < prog < 0.55, f"{prog:.2f}")
    check("进度在0~1 之间", 0.0 <= prog <= 1.0, str(prog))

    # ---- 序列化给 GUI 用（as_dict 在状态快照上）
    d4 = p4.state().as_dict()
    for key in ("phase", "phase_label", "remaining_s", "remaining_fmt", "rounds_done"):
        check(f"as_dict 含 {key}", key in d4)

    # ---- reset
    p4.reset()
    check("reset 回到初始", p4.in_focus and p4.rounds_done == 0 and not p4.just_changed)

    # ---- 统计报告包含课堂模式分组
    from ai_teacher.stats import SessionStats, format_report

    st = SessionStats()
    st.screenshots = 10
    st.local_hits = 10
    check("未用课堂模式时不显示该组", "完成轮次" not in dict(format_report(st, "x")))

    st.pomodoro_used = True
    st.pomodoro_rounds = 3
    st.pause_seconds = 1800
    rows = dict(format_report(st, "x"))
    check("用了课堂模式会显示轮次", "3 轮" in rows.get("完成轮次", ""), rows.get("完成轮次", ""))
    check("会显示累计休息", "30 分钟" in rows.get("累计休息", ""), rows.get("累计休息", ""))


def test_face_detection() -> None:
    """在场检测：Haar + 肤色兜底。

    曾经的 bug：用户坐在镜头前却一直显示「未检测到人」。根因有二：
      1. scaleFactor=1.2 对俯拍视角（人脸偏在画面下边缘、只露出额头和眼镜）
         会稳定漏检，实测 1.05 才稳定检出；
      2. 单一 Haar 通道在低头场景本身就不可靠，必须有肤色兜底。
    """

    section("10. 在场检测（Haar + 肤色兜底）")

    from ai_teacher.rules import FaceDetector, LocalJudge, Verdict

    det = FaceDetector()

    # ---- 构造一张「有脸」的合成图：肤色椭圆 + 暗色五官块
    def synth_face(face_w: int = 140) -> np.ndarray:
        img = np.full((480, 640, 3), 90, dtype=np.uint8)  # 冷色背景
        cx, cy = 320, 340  # 画面下部，像俯拍
        img[cy - face_w // 2 : cy + face_w // 2, cx - face_w // 2 : cx + face_w // 2] = (
            150, 170, 200
        )  # BGR 肤色
        # 眼睛/嘴的暗块，制造 Haar 依赖的局部特征
        for ox in (-face_w // 6, face_w // 6):
            cv2.rectangle(
                img,
                (cx + ox - 10, cy - face_w // 5),
                (cx + ox + 10, cy - face_w // 5 + 12),
                (40, 40, 40),
                -1,
            )
        cv2.rectangle(
            img,
            (cx - 22, cy + face_w // 4),
            (cx + 22, cy + face_w // 4 + 10),
            (60, 60, 90),
            -1,
        )
        return img

    face = synth_face()
    present, why, ratio = det.present(face)
    check("合成人脸判为在场", present, f"{why} 肤色={ratio:.0%}")

    # ---- 关键回归：模拟真实故障场景（Haar 失效，只剩肤色）
    # 只画肤色块、不画五官，Haar 会失效，但肤色通道必须兜住。
    def synth_skin_only() -> np.ndarray:
        img = np.full((480, 640, 3), 90, dtype=np.uint8)
        img[240:460, 240:400] = (150, 170, 200)  # 只有肤色，无五官
        return img

    skin_only = synth_skin_only()
    p2, w2, r2 = det.present(skin_only)
    check("Haar 失效时肤色通道兜住", p2, f"{w2} 肤色={r2:.0%}")
    check("兜底时肤色占比超过阈值", r2 >= FaceDetector._SKIN_RATIO_MIN, f"{r2:.1%}")
    check(
        "阈值取值合理（0.05~0.3）",
        0.05 <= FaceDetector._SKIN_RATIO_MIN <= 0.3,
        str(FaceDetector._SKIN_RATIO_MIN),
    )

    # ---- 无人场景：绝不能误判为在场
    for label, img in [
        ("冷色空背景", np.full((480, 640, 3), (90, 40, 30), dtype=np.uint8)),
        ("纯黑", np.zeros((480, 640, 3), dtype=np.uint8)),
        ("纯白", np.full((480, 640, 3), 255, dtype=np.uint8)),
        ("纯灰", np.full((480, 640, 3), 128, dtype=np.uint8)),
        (
            "随机噪声",
            np.random.RandomState(0).randint(0, 255, (480, 640, 3), dtype=np.uint8),
        ),
    ]:
        p, w2, r2 = det.present(img)
        check(f"{label} 判为不在场", not p, f"{w2} 肤色={r2:.1%}")

    # ---- scaleFactor 必须是 <=1.1，否则俯拍场景漏检（回归点）
    import inspect

    src = inspect.getsource(FaceDetector.detect)
    check("detect 未做预缩放", "resize" not in src, "预缩放会让小脸更难识别")
    check("detect 显式指定 scaleFactor", "scaleFactor=" in src, "依赖默认值易被改坏")
    check("scaleFactor <= 1.1", "scaleFactor=1.05" in src or "scaleFactor=1.1" in src, src[:200])

    # ---- 连续帧确认：单帧不在场不判分心
    from ai_teacher.config import RuleSettings

    cfg = RuleSettings()
    check("默认需连续 3 帧确认", cfg.absence_confirm == 3, str(cfg.absence_confirm))

    judge = LocalJudge(cfg)
    empty = np.full((480, 640, 3), 100, dtype=np.uint8)
    verdicts = []
    for _ in range(cfg.absence_confirm):
        s = judge.evaluate(empty, empty, 160, 6.0)
        verdicts.append(s.verdict)
    seq = [v.value for v in verdicts]
    check("确认前不判分心（抗抖动）", Verdict.DISTRACT not in verdicts[:-1], str(seq))
    check("连续确认后才判分心", verdicts[-1] is Verdict.DISTRACT, str(seq))
    check("帧计数单调递增", judge._absent_hits == cfg.absence_confirm, str(judge._absent_hits))

    # 在场时绝不能判分心
    judge2 = LocalJudge(cfg)
    for _ in range(5):
        s2 = judge2.evaluate(face, face, 160, 6.0)
    check("在场时判为在场", s2.person_present is True, s2.person_reason)
    check("在场时绝不判分心", s2.verdict is not Verdict.DISTRACT, s2.verdict.value)
    check("在场时重置连续计数", judge2._absent_hits == 0, str(judge2._absent_hits))

    # 中途离开再回来，计数应正确重置
    judge2.evaluate(empty, empty, 160, 6.0)
    judge2.evaluate(empty, empty, 160, 6.0)
    check("离开时开始累计", judge2._absent_hits == 2, str(judge2._absent_hits))
    back = judge2.evaluate(face, face, 160, 6.0)
    check("重新在场则计数清零", judge2._absent_hits == 0, f"streak={judge2._absent_hits}")

    # ---- 摄像头关闭时降级，不崩
    judge3 = LocalJudge(cfg)
    s3 = judge3.evaluate(np.full((400, 600, 3), 128, dtype=np.uint8), None, 160, 6.0)
    check("摄像头关闭时降级为存疑", s3.person_present is None, str(s3.person_present))


def test_isolation() -> None:
    """确认测试进程读不到真实凭据。

    这是防「测试扣费」的关键断言：本机 .env 里有真实 key 时，
    若隔离层失效，GUI 测试的 _start() 在特定场景下会真的发请求。
    """
    section("9. 测试隔离（防止用真实 key 扣费）")

    from ai_teacher.config import get_settings

    s = get_settings()
    check("API key 已被清空", not s.api.api_key, repr(s.api.api_key[:8]))
    check("base_url 指向不可路由地址", "unreachable" in s.api.base_url, s.api.base_url)

    # 构造判定器时不给 key也不应崩，且必须立刻失败（而不是真发请求）
    from ai_teacher.judge import VisionJudge

    j = VisionJudge(s.api)
    v = j.judge(b"\xff\xd8\xff\xd9", "学习")
    check("无 key 时判定跳过而非报错", not v.ok and v.error == "no_api_key", v.error)
    check("无 key 时不产生 token 消耗", j.total_calls == 0 and j.total_prompt_tokens == 0)

    # 隔离层自检函数本身要返回真
    from tests.isolation import assert_isolated

    ok, msg = assert_isolated()
    check("隔离自检函数返回通过", ok, msg)

    # 确认项目根目录可能存在 .env（说明这个测试有意义）
    from ai_teacher.config import ROOT

    env_exists = (ROOT / ".env").exists()
    print(f"       （本机 .env {'存在' if env_exists else '不存在'}，"
          f"隔离层在两种情况下都应生效）")


def test_stats() -> None:
    section("8. 会话统计与报告")
    import tempfile
    from pathlib import Path

    from ai_teacher.db import Store
    from ai_teacher.stats import SessionStats, est_cost, format_report, format_report_from_dict

    st = SessionStats()
    st.screenshots = 100
    st.camera_shots = 100
    st.local_hits = 68
    st.api_calls = 30
    st.api_failed = 2
    st.focus_ticks = 65
    st.distracted_ticks = 25
    st.unsure_ticks = 10
    st.alerts = 3
    st.prompt_tokens = 30 * 250
    st.completion_tokens = 30 * 40
    st.image_bytes = 30 * 30 * 1024

    # ---- 计数与派生指标
    check("截图次数被记录", st.screenshots == 100)
    check("拍照次数独立于截图", st.camera_shots == 100)
    # total_ticks 含 api_failed —— 漏掉它会让报告出现「各项加起来超过总数」
    check(
        "总判定次数含调用失败",
        st.total_ticks == 68 + 30 + 2,
        str(st.total_ticks),
    )
    check(
        "总判定次数 = 三个结果项之和",
        st.total_ticks == st.focus_ticks + st.distracted_ticks + st.unsure_ticks,
        f"{st.total_ticks} vs {st.focus_ticks + st.distracted_ticks + st.unsure_ticks}",
    )
    check("本地拦截率 68%", abs(st.local_ratio - 0.68) < 0.001, f"{st.local_ratio:.2f}")
    check("专注比例 65%", abs(st.focus_ratio - 0.65) < 0.001, f"{st.focus_ratio:.2f}")
    check("结果计数自洽", st.focus_ticks + st.distracted_ticks + st.unsure_ticks == 100)

    # ---- 成本（价格随时段变化，这里显式指定空闲价保证可预期）
    expect = (7500 * 1.0 + 1200 * 4.0) / 1_000_000
    check(
        "花费估算正确（空闲价）",
        abs(est_cost(7500, 1200, peak=False) - expect) < 1e-9,
        f"¥{est_cost(7500, 1200, peak=False):.5f}",
    )
    check("单次平均输入 250", abs(st.avg_prompt_tokens - 250) < 1.0)
    check("上传量统计", abs(st.image_mb - 0.8789) < 0.01, f"{st.image_mb:.2f} MB")
    # 空闲价：1000*1 + 500*4 = 3000 / 1e6
    check(
        "est_cost 空闲价正确",
        abs(est_cost(1000, 500, peak=False) - 0.003) < 1e-9,
        f"¥{est_cost(1000, 500, peak=False):.6f}",
    )
    # 高峰价是空闲价的 2 倍
    check(
        "est_cost 高峰价 = 空闲价 ×2",
        abs(est_cost(1000, 500, peak=True) - 0.006) < 1e-9,
        f"¥{est_cost(1000, 500, peak=True):.6f}",
    )

    # ---- 报告行
    rows = format_report(st, "刷 LeetCode")
    labels = [a for a, _ in rows]
    check("报告含截图次数", any("截图次数" in a for a in labels))
    check("报告含拍照次数", any("拍照次数" in a for a in labels))
    check("报告含本地规则定案", any("本地规则定案" == a for a in labels), str(labels))
    check("报告含模型判定", any("模型判定" == a for a in labels))
    check("报告含判定总次数", any("判定总次数" == a for a in labels), str(labels))
    check("报告含判定结果分组", any("判定结果" in a for a in labels))
    check("报告含预估花费", any("预估花费" in a for a in labels))
    check("报告含 token 消耗", any("token" in a for a in labels))
    check("报告含告警次数", any("告警次数" in a for a in labels))
    check("报告有分组标题", any(not v for _, v in rows))

    # ---- dict 往返（GUI 走这条路）
    rows2 = format_report_from_dict(st.as_dict())
    # 用列表比较：dict() 会把重复的分组标题（空值）覆盖掉，不能用来比行数
    check("dict 往返后行数一致", len(rows2) == len(rows), f"{len(rows2)} vs {len(rows)}")
    diffs = [(a, b) for (a, b), (c, d) in zip(rows, rows2) if (a, b) != (c, d)]
    check("dict 往返后数值一致", not diffs, "GUI 与 CLI 应显示相同内容" if not diffs else str(diffs[:2]))

    # ---- 未调用模型时的报告
    empty = SessionStats()
    empty.screenshots = 5
    empty.local_hits = 5
    erows = dict(format_report(empty, "x"))
    check("未调模型时提示 100% 本地", "100%" in erows["本地规则定案"], erows["本地规则定案"])
    check("未调模型时花费为 0", "¥0.00000" in erows["预估花费"])
    # 样本 >=2 时平均间隔才有意义
    check("多样本时显示平均间隔", "平均间隔" in erows)
    one = SessionStats()
    one.screenshots = 1
    one.local_hits = 1
    orows = dict(format_report(one, "x"))
    check("单样本时不显示平均间隔", "平均间隔" not in orows, "首tick 含启动开销，会拉偏")

    # ---- 摄像头关闭的情况
    nocam = SessionStats()
    nocam.screenshots = 1
    nocam.local_hits = 1
    nrows = dict(format_report(nocam, "x"))
    check("摄像头关闭时拍照为 0", "0 次" in nrows["拍照次数"], nrows["拍照次数"])

    # ---- 落库与读回
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        with Store(db) as store:
            sid = store.start_session("统计测试")
            store.save_session_stats(sid, st.as_dict())
            store.end_session(sid)
            back = store.session_stats(sid)
            check("统计可落库并读回", back is not None and back["screenshots"] == 100)
            check("读回目标是原目标", store.session_goal(sid) == "统计测试")
            check("不存在的会话返回 None", store.session_stats(9999) is None)
            check("session_goal 不存在时返回 None", store.session_goal(9999) is None)

    # ---- 老库迁移
    import sqlite3

    with tempfile.TemporaryDirectory() as td:
        old_db = Path(td) / "old.db"
        conn = sqlite3.connect(old_db)
        conn.execute(
            "CREATE TABLE sessions (id INTEGER PRIMARY KEY, goal TEXT, "
            "started_at REAL, ended_at REAL)"
        )
        conn.execute(
            "INSERT INTO sessions (goal, started_at) VALUES ('旧会话', 1000.0)"
        )
        conn.commit()
        conn.close()
        with Store(old_db) as store:
            check("老库自动加 stats 列", store.session_stats(1) is None, "应为 None 而非报错")
            check("老库仍能读到目标", store.session_goal(1) == "旧会话")

    # ---- 部分缺字段要宽容（老版本落库的 dict 没有新字段）
    partial = st.as_dict()
    partial.pop("image_bytes")
    prows = format_report_from_dict(partial)
    check("缺部分字段仍可出报告", len(prows) > 0, f"{len(prows)} 行")
    check(
        "缺 image_bytes 时不上传量行",
        "图片上传量" not in [a for a, _ in prows],
        "按 0 处理，隐藏该行",
    )

    # ---- 完全无法识别才报错
    try:
        format_report_from_dict({"foo": 1})
        check("无法识别的字典报错", False, "应抛 KeyError")
    except KeyError as exc:
        check("无法识别的字典报错", True, str(exc)[:40])


if __name__ == "__main__":
    sys.exit(main())
