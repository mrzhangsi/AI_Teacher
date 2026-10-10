"""L2 状态机：抗抖动 + 自适应采样间隔 + 告警触发。

单帧误报是必然的（回条消息、切个窗口都会被拍到），所以：
- L0 硬规则分心 -> 立即告警（确定性高）
- L1 模型判定分心 -> 需连续 N 次才告警
- 启动有宽限期，期间不打扰
- 告警有冷却，避免刷屏
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .config import JudgeSettings
from .rules import Verdict


class FocusState(str, Enum):
    FOCUSING = "focusing"
    UNSURE = "unsure"
    DISTRACTED = "distracted"


@dataclass(slots=True)
class Decision:
    """单次 tick 的综合结论。"""

    state: FocusState
    reason: str
    should_alert: bool
    next_interval_s: float
    used_api: bool
    from_l0: Verdict | None = None


@dataclass(slots=True)
class FocusTracker:
    """维护学习状态与采样节奏。"""

    cfg: JudgeSettings
    started_at: float = field(default_factory=time.monotonic)
    strikes: int = 0
    focus_streak: int = 0
    last_alert_at: float = 0.0
    interval_s: float = 0.0
    ewma_focus: float = 1.0
    alert_count: int = 0
    samples: int = 0
    api_calls: int = 0
    total_samples: int = 0

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            self.interval_s = self.cfg.interval_min_s

    # ------------------------------------------------------------ 采样间隔

    def _grow_interval(self) -> None:
        """状态好 -> 逐步拉长采样间隔，省调用次数。"""

        if self.focus_streak < self.cfg.interval_grow_after:
            return
        if self.interval_s < self.cfg.interval_max_s:
            self.interval_s = min(self.cfg.interval_max_s, self.interval_s * 1.6)
        self.focus_streak = 0  # 用掉一次成长机会

    def _shrink_interval(self) -> None:
        """可疑 -> 立刻回落到最短间隔，尽快确认。"""

        self.interval_s = self.cfg.interval_min_s
        self.focus_streak = 0

    # ------------------------------------------------------------ 主入口

    def update(self, l0: Verdict, reason: str, model_studying: bool | None, used_api: bool) -> Decision:
        """吃进本tick 的信号，产出综合决策。"""

        self.samples += 1
        self.total_samples += 1
        if used_api:
            self.api_calls += 1

        # 1) 汇总本 tick 的是否在学习
        if l0 is Verdict.FOCUS:
            studying = True
        elif l0 is Verdict.DISTRACT:
            studying = False
        else:
            studying = bool(model_studying) if model_studying is not None else None

        if studying is None:
            state = FocusState.UNSURE
        else:
            self.ewma_focus = (
                self.cfg.focus_ewma_alpha * float(studying)
                + (1 - self.cfg.focus_ewma_alpha) * self.ewma_focus
            )
            if not studying:
                self.strikes += 1
                self.focus_streak = 0
            else:
                self.strikes = 0
                self.focus_streak += 1

            if self.ewma_focus >= self.cfg.focus_healthy_ratio:
                state = FocusState.FOCUSING
            elif self.ewma_focus <= self.cfg.focus_bad_ratio:
                state = FocusState.DISTRACTED
            else:
                state = FocusState.UNSURE

        # 2) 判定是否该告警
        should_alert = self._should_alert(state, l0, studying)

        # 3) 调整下次间隔
        if state is FocusState.FOCUSING:
            self._grow_interval()
        else:
            self._shrink_interval()

        return Decision(
            state=state,
            reason=reason,
            should_alert=should_alert,
            next_interval_s=self.interval_s,
            used_api=used_api,
            from_l0=l0,
        )

    def _should_alert(self, state: FocusState, l0: Verdict, studying: bool | None) -> bool:
        if state is FocusState.FOCUSING:
            return False

        now = time.monotonic()
        # 宽限期
        if now - self.started_at < self.cfg.grace_period_s:
            return False
        # 冷却
        if self.last_alert_at and now - self.last_alert_at < self.cfg.alert_cooldown_s:
            return False

        # L0 硬规则：确定性高，立即告警
        if l0 is Verdict.DISTRACT:
            self.last_alert_at = now
            self.alert_count += 1
            return True

        # L1 模型判定：需连续 N 次
        if studying is False and self.strikes >= self.cfg.strikes_to_warn:
            self.strikes = 0  # 消费掉，避免冷却结束后立刻再报
            self.last_alert_at = now
            self.alert_count += 1
            return True

        return False

    # ------------------------------------------------------------ 统计

    def focus_ratio(self) -> float:
        return self.ewma_focus

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at
