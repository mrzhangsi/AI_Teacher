"""课堂模式（Pomodoro）：专注与休息的循环状态机。

设计要点：
- **不负责睡觉**，只回答两个问题：现在该专注还是休息？还剩多久？
  实际等待由 Monitor 的可中断 sleep 完成，所以用户随时能停止。
- 用 monotonic 时钟，不受系统时间调整影响（改系统时间不会让计时乱掉）。
- 休息期间完全不截图不拍照，这是「省电 + 保护隐私 + 不打扰」的关键。

状态流转：
    FOCUS --(focus_minutes 到)--> BREAK --(break_minutes 到)--> FOCUS
                (break_minutes == 0 时直接跳回 FOCUS)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    """当前处于哪个阶段。"""

    FOCUS = "focus"
    BREAK = "break"


# 阶段的中文名，供界面显示
PHASE_LABEL = {Phase.FOCUS: "专注中", Phase.BREAK: "休息中"}


@dataclass
class PomodoroState:
    """课堂模式运行时状态（纯数据，便于测试与界面展示）。"""

    phase: Phase
    # 本阶段剩余秒数
    remaining_s: float
    # 本阶段总时长（秒），用于算进度条
    total_s: float
    # 已完成的专注轮次
    rounds_done: int = 0
    # 累计休息秒数
    break_total_s: float = 0.0

    @property
    def progress(self) -> float:
        """已完成比例 0.0~1.0。"""

        if self.total_s <= 0:
            return 1.0
        done = self.total_s - max(0.0, self.remaining_s)
        return min(1.0, max(0.0, done / self.total_s))

    @property
    def remaining_fmt(self) -> str:
        """剩余时间，形如 ``04:59``（不足 1 小时）。超过 1 小时用 ``1:05:00``。"""

        s = int(max(0.0, self.remaining_s))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    def as_dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "phase_label": PHASE_LABEL[self.phase],
            "remaining_s": self.remaining_s,
            "remaining_fmt": self.remaining_fmt,
            "total_s": self.total_s,
            "progress": self.progress,
            "rounds_done": self.rounds_done,
            "break_total_s": self.break_total_s,
        }


class Pomodoro:
    """课堂模式计时器。

    用法：
        pom = Pomodoro(focus_minutes=40, break_minutes=10)
        state = pom.tick()# 取当前状态
        if pom.should_notify_break(): # 该弹休息提示了
            ...
        pom.advance()# 时间到了，推进到下一阶段
    """

    def __init__(
        self,
        focus_minutes: float = 40.0,
        break_minutes: float = 10.0,
        *,
        enabled: bool = True,
        clock=time.monotonic,
    ) -> None:
        self.enabled = enabled
        self.focus_s = max(1.0, focus_minutes * 60)
        self.break_s = max(0.0, break_minutes * 60)
        self._clock = clock

        self.phase = Phase.FOCUS
        # 本阶段开始的 monotonic 时刻
        self._phase_started = clock()
        self._elapsed_in_phase = 0.0  # 跨多个 tick 累计的已过秒数
        self.rounds_done = 0
        self.break_total_s = 0.0
        # 阶段刚切换时置True，供调用方决定要不要发通知
        self._just_changed = False

    # ------------------------------------------------------------ 推进

    def update_elapsed(self) -> float:
        """累计本阶段已过秒数。

        每次 tick 前调用一次即可——即使中间停了 30 分钟也能正确累计，
        不会出现「重启就重置」的问题。
        """

        self._elapsed_in_phase += self._clock() - self._phase_started
        self._phase_started = self._clock()
        return self._elapsed_in_phase

    def _remaining(self, total: float) -> float:
        return max(0.0, total - self._elapsed_in_phase)

    def state(self) -> PomodoroState:
        """取当前状态快照。"""

        total = self.focus_s if self.phase is Phase.FOCUS else self.break_s
        return PomodoroState(
            phase=self.phase,
            remaining_s=self._remaining(total),
            total_s=total,
            rounds_done=self.rounds_done,
            break_total_s=self.break_total_s,
        )

    def phase_expired(self) -> bool:
        """本阶段是否已到时间。"""

        if self.phase is Phase.FOCUS:
            return self._elapsed_in_phase >= self.focus_s
        return self.break_s > 0 and self._elapsed_in_phase >= self.break_s

    def advance(self) -> Phase:
        """时间到了，推进到下一阶段。返回新阶段。

        - FOCUS -> BREAK：若 break_s == 0 则直接跳回 FOCUS（不计轮次跳跃）
        - BREAK -> FOCUS：专注轮次 +1
        """

        self._just_changed = True
        if self.phase is Phase.FOCUS:
            self.rounds_done += 1
            if self.break_s <= 0:
                # 不休息，直接开下一轮
                self.phase = Phase.FOCUS
            else:
                self.phase = Phase.BREAK
        else:
            self.phase = Phase.FOCUS

        self._elapsed_in_phase = 0.0
        self._phase_started = self._clock()
        return self.phase

    @property
    def just_changed(self) -> bool:
        """本 tick 是否刚发生了阶段切换（据此决定发不发通知）。"""

        return self._just_changed

    def clear_changed(self) -> None:
        """读完just_changed 后调用，复位标记。"""

        self._just_changed = False

    # ------------------------------------------------------------ 便捷属性

    @property
    def in_break(self) -> bool:
        return self.phase is Phase.BREAK

    @property
    def in_focus(self) -> bool:
        return self.phase is Phase.FOCUS

    def describe(self) -> str:
        """人类可读的一行描述，用于日志。"""

        if not self.enabled:
            return "课堂模式未启用"
        st = self.state()
        if st.phase is Phase.FOCUS:
            return f"专注中，剩余 {st.remaining_fmt}"
        return f"休息中，剩余 {st.remaining_fmt}"

    def add_break_time(self, seconds: float) -> None:
        """累加累计休息时长（用于统计）。"""

        self.break_total_s += max(0.0, seconds)

    def reset(self) -> None:
        """回到初始状态。"""

        self.phase = Phase.FOCUS
        self._elapsed_in_phase = 0.0
        self._phase_started = self._clock()
        self.rounds_done = 0
        self.break_total_s = 0.0
        self._just_changed = False