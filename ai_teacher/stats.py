"""会话统计：采集次数、本地/API 判定比例、token 与花费。

设计要点：
- 截图次数和拍照次数分开统计——摄像头可能被关闭或不可用，两者不一定相等
- 本地判定 / API 判定分开统计，这是本项目成本控制效果最直观的体现
- 单次判定的 token 用滑动平均，避免只有一两次调用时被单点数字误导
- **价格区分高峰/空闲时段**：同一张价目表下单价差2 倍，算错会严重低估
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone, time as dtime
from dataclasses import dataclass, field

# DeepSeek 官方定价（元/ 百万 tokens），deepseek-flash
#   高峰时段：输入 ¥2/M（缓存未命中），输出 ¥8/M
#   空闲时段：输入 ¥1/M，输出 ¥4/M
# 高峰 = 周一至周五（不含法定节假日）9:00-12:00 与 14:00-18:00（北京时间）
PRICE_IN_PEAK = 2.0
PRICE_OUT_PEAK = 8.0
PRICE_IN_OFFPEAK = 1.0
PRICE_OUT_OFFPEAK = 4.0

# 中国全境单一时区 UTC+8 且无夏令时，用固定偏移即可。
# 不用 zoneinfo 是为了避开 tzdata 依赖（Windows 上 zoneinfo 需要额外装包）。
_CN_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")

# 高峰时段区间（北京时间）
_PEAK_WINDOWS = ((dtime(9, 0), dtime(12, 0)), (dtime(14, 0), dtime(18, 0)))


def is_peak_time(moment: datetime | None = None) -> bool:
    """当前是否处于高峰时段（按北京时间判断）。

    注意：官方说明明确「不含中国法定节假日」，这里只按工作日判断——
    法定节假日需要额外维护日历表，成本收益不划算，节假日估算偏高一点可接受。
    """

    now = moment or datetime.now(_CN_TZ)
    if now.weekday() >= 5:  # 周末全天空闲
        return False
    t = now.time()
    return any(start <= t < end for start, end in _PEAK_WINDOWS)


def current_prices() -> tuple[float, float]:
    """返回当前时段的（输入价, 输出价），单位：元/ 百万 tokens。"""

    if is_peak_time():
        return PRICE_IN_PEAK, PRICE_OUT_PEAK
    return PRICE_IN_OFFPEAK, PRICE_OUT_OFFPEAK

# SessionStats 中可直接赋值的计数字段（其余都是派生属性）。
# 刻意手写而不是用 dataclasses.fields 自动推导：自动推导会把 started_at
# 也算进去，from_dict 还原时若 dict 里没有该键就会覆盖掉默认的计时起点。
_COUNTER_FIELDS = (
    "screenshots",
    "camera_shots",
    "camera_miss",
    "local_hits",
    "api_calls",
    "api_failed",
    "api_skipped",
    "focus_ticks",
    "distracted_ticks",
    "unsure_ticks",
    "alerts",
    "prompt_tokens",
    "completion_tokens",
    "image_bytes",
    "pause_seconds",
    "pomodoro_rounds",
)


def est_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    peak: bool | None = None,
) -> float:
    """按当前时段价格估算花费（元）。

    Args:
        peak: 是否按高峰价计算。None = 自动判断当前时段。
              显式传 True/False 便于测试与回溯历史会话。
    """

    in_price, out_price = (
        (PRICE_IN_PEAK, PRICE_OUT_PEAK)
        if (is_peak_time() if peak is None else peak)
        else (PRICE_IN_OFFPEAK, PRICE_OUT_OFFPEAK)
    )
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


@dataclass
class SessionStats:
    """一次学习会话的累计统计。

    全部是进程内计数，不依赖数据库查询——这样即使数据库出问题，
    结束时的报告依然能给出准确数字。
    """

    started_at: float = field(default_factory=time.monotonic)

    # 采集
    screenshots: int = 0  # 屏幕截图次数（每个 tick 必截一次）
    camera_shots: int = 0  # 摄像头拍照次数（摄像头关闭/不可用时为 0）
    camera_miss: int = 0  # 摄像头读取失败次数

    # 判定来源
    local_hits: int = 0  # L0 本地规则直接定案，未调API
    api_calls: int = 0  # 实际提交给模型的次数
    api_failed: int = 0  # 调用失败（含网络异常、解析失败）
    api_skipped: int = 0  # 存疑但因缺 key 而跳过

    # 判定结果
    focus_ticks: int = 0
    distracted_ticks: int = 0
    unsure_ticks: int = 0

    # 告警
    alerts: int = 0

    # token
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # 上传体积（用于说明"图片压缩省了多少"）
    image_bytes: int = 0

    # 课堂模式：累计休息秒数与完成轮数。
    # 休息这段时间不抓拍，单独列出来更清楚。
    pause_seconds: float = 0.0
    pomodoro_rounds: int = 0
    # 本次会话是否用了课堂模式（决定报告里显不显示这一组）
    pomodoro_used: bool = False

    @property
    def total_ticks(self) -> int:
        """总判定次数 = 本地定案 + 模型成功 + 模型失败 + 因缺 key 跳过。

        **别漏掉 api_failed**。模型调用失败（网络错误、返回无法解析等）
        也是一次真实发生的判定，只是没拿到结果。漏掉它会让 total_ticks
        小于 focus+distracted+unsure，报告出现「各项加起来超过总数」的矛盾。
        """

        return self.local_hits + self.api_calls + self.api_failed + self.api_skipped

    @property
    def local_ratio(self) -> float:
        """本地判定占总判定的比例。省钱效果就看这个数。

        分母与 total_ticks 保持一致（同样含 api_failed），
        否则会出现「本地判定 3 次却显示占100%」这种误导性结论。
        """

        return self.local_hits / self.total_ticks if self.total_ticks else 0.0

    @property
    def api_attempts(self) -> int:
        """实际发起的模型请求次数（含失败与跳过）。"""

        return self.api_calls + self.api_failed + self.api_skipped

    @property
    def api_success_rate(self) -> float:
        if self.api_calls == 0:
            return 0.0
        return (self.api_calls - self.api_failed) / self.api_calls

    @property
    def focus_ratio(self) -> float:
        """判定为「在学习」的比例。分母含 unsure，口径与DB 查询一致。"""

        return self.focus_ticks / self.total_ticks if self.total_ticks else 0.0

    @property
    def avg_prompt_tokens(self) -> float:
        if self.api_calls == 0:
            return 0.0
        return self.prompt_tokens / self.api_calls

    @property
    def est_cost(self) -> float:
        return est_cost(self.prompt_tokens, self.completion_tokens)

    @property
    def image_mb(self) -> float:
        return self.image_bytes / 1024 / 1024

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def elapsed_fmt(self) -> str:
        s = int(self.elapsed_s)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    @property
    def avg_interval_s(self) -> float:
        """平均采样间隔，能直观反映自适应节奏。"""

        return self.elapsed_s / self.total_ticks if self.total_ticks else 0.0
    def as_dict(self) -> dict:
        """导出为可 JSON 化的字典，供 GUI 事件使用。"""

        return {
            "screenshots": self.screenshots,
            "camera_shots": self.camera_shots,
            "camera_miss": self.camera_miss,
            "local_hits": self.local_hits,
            "api_calls": self.api_calls,
            "api_failed": self.api_failed,
            "api_skipped": self.api_skipped,
            "total_ticks": self.total_ticks,
            "local_ratio": self.local_ratio,
            "focus_ticks": self.focus_ticks,
            "distracted_ticks": self.distracted_ticks,
            "unsure_ticks": self.unsure_ticks,
            "focus_ratio": self.focus_ratio,
            "alerts": self.alerts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "est_cost": self.est_cost,
            "image_bytes": self.image_bytes,
            "image_mb": self.image_mb,
            "pause_seconds": self.pause_seconds,
            "pomodoro_rounds": self.pomodoro_rounds,
            "pomodoro_used": self.pomodoro_used,
            "elapsed_s": self.elapsed_s,
            "elapsed_fmt": self.elapsed_fmt,
            "avg_interval_s": self.avg_interval_s,
        }


def format_report_from_dict(st: dict) -> list[tuple[str, str]]:
    """从 as_dict() 的结果重建报告行。

    GUI 侧只能拿到 dict（事件已序列化），这里还原成临时对象后复用
    同一套格式化逻辑，保证 CLI 和 GUI 显示的数字完全一致。
    """

    tmp = SessionStats()
    # 只还原可写的基础计数；派生属性（比例、均值、花费）一律重算，
    # 避免直接赋值造成数字与实际不符。
    #
    # 缺字段要宽容：早期版本落库的 dict 没有后来新增的字段（如 image_bytes），
    # 直接抛错会让历史会话全部无法查看。缺失项按 0 补齐即可——数字略保守，
    # 但报告结构完整可用。完全无法识别（一个字段都没有）才报错。
    if not any(k in st for k in _COUNTER_FIELDS):
        raise KeyError(f"统计字典不包含任何可识别的计数字段：{sorted(st)[:6]}")
    for key in _COUNTER_FIELDS:
        if key in st:
            setattr(tmp, key, st[key])

    # 用真实已用时长校正 elapsed_s
    elapsed = st.get("elapsed_s")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        tmp.started_at = time.monotonic() - float(elapsed)

    return format_report(tmp, "")


def format_report(st: SessionStats, goal: str) -> list[tuple[str, str]]:
    """生成 (标签, 值) 列表，供 CLI 表格和 GUI 复用。

    分组顺序：采集 -> 判定 -> 结果 -> 成本。
    CLI 侧用rich 表格渲染，GUI 侧用文本行渲染，保证两处数字完全一致。
    """

    rows: list[tuple[str, str]] = []

    rows.append(("── 采集 ──", ""))
    rows.append(("截图次数", f"{st.screenshots} 次"))
    if st.camera_shots or st.camera_miss:
        cam_txt = f"{st.camera_shots} 次"
        if st.camera_miss:
            cam_txt += f"（失败 {st.camera_miss} 次）"
        rows.append(("拍照次数", cam_txt))
    else:
        rows.append(("拍照次数", "0 次（摄像头未启用）"))
    # 样本太少时平均间隔没意义（首tick 含启动开销，会严重拉偏）
    if st.total_ticks >= 2:
        rows.append(("平均间隔", f"{st.avg_interval_s:.0f} 秒"))

    rows.append(("── 判定来源 ──", ""))
    if st.api_calls or st.api_failed or st.api_skipped:
        # 有过真实调用：显示拦截率，这是成本控制效果的直观体现
        rows.append(
            (
                "本地规则定案",
                f"{st.local_hits} 次（占 {st.local_ratio * 100:.0f}%，零成本）",
            )
        )
        rows.append(("模型判定", f"{st.api_calls} 次（成功取到结果）"))
        if st.api_failed:
            rows.append(("  模型失败", f"{st.api_failed} 次（未取到结果）"))
        if st.api_skipped:
            rows.append(("  缺 key 跳过", f"{st.api_skipped} 次"))
        rows.append(("判定总次数", f"{st.total_ticks} 次"))
        if st.api_failed or st.api_skipped:
            # 失败/跳过不产生状态，容易让人怀疑「各项加起来超过总数」，
            # 这里主动说明清楚
            lost = st.api_failed + st.api_skipped
            rows.append(
                (
                    "  其中未出结果",
                    f"{lost} 次（计入总数，但计入下方「不确定」）",
                )
            )
    else:
        rows.append(("本地规则定案", f"{st.local_hits} 次（100%，未调用模型）"))
        rows.append(("模型判定", "0 次（未触发模型调用）"))
        rows.append(("判定总次数", f"{st.total_ticks} 次"))

    rows.append(("── 判定结果 ──", ""))
    rows.append(("学习中", f"{st.focus_ticks} 次"))
    rows.append(("分心", f"{st.distracted_ticks} 次"))
    rows.append(("不确定", f"{st.unsure_ticks} 次"))
    rows.append(("专注比例", f"{st.focus_ratio * 100:.0f}%"))
    rows.append(("告警次数", f"{st.alerts} 次"))

    # 课堂模式：只在用过时才显示，不占版面
    if st.pomodoro_used or st.pomodoro_rounds or st.pause_seconds:
        rows.append(("── 课堂模式 ──", ""))
        rows.append(("完成轮次", f"{st.pomodoro_rounds} 轮"))
        if st.pause_seconds:
            mins = int(st.pause_seconds // 60)
            secs = int(st.pause_seconds % 60)
            txt = f"{mins} 分钟" if mins else f"{secs} 秒"
            rows.append(("累计休息", f"{txt}（期间未抓拍）"))

    rows.append(("── 成本 ──", ""))
    if st.api_calls:
        rows.append(
            (
                "token 消耗",
                f"输入 {st.prompt_tokens:,} / 输出 {st.completion_tokens:,}",
            )
        )
        rows.append(("单次平均输入", f"{st.avg_prompt_tokens:.0f} tokens"))
        rows.append(("预估花费", f"¥{st.est_cost:.5f}"))
        if st.image_mb:
            rows.append(("图片上传量", f"{st.image_mb:.2f} MB"))
    else:
        rows.append(("token 消耗", "0（未调用模型）"))
        rows.append(("预估花费", "¥0.00000"))

    return rows
