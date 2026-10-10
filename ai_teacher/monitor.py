"""监控主循环：串联 L0 -> L1 -> L2。

设计要点：
- L0 判FOCUS/DISTRACT 时根本不调 API，直接落地
- 只有 SUSPECT/UNKNOWN 才合成图片调模型
- 每tick 结束按tracker 给的间隔 sleep，而不是固定间隔
- Ctrl+C 优雅退出并打印本次统计
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable

from rich.console import Console
from rich.table import Table

from . import alert, journal
from .capture import CameraReader, ScreenGrabber, compose
from .config import Settings, get_settings
from .db import Store
from .judge import VisionJudge
from .rules import LocalJudge, Verdict
from .pomodoro import Pomodoro
from .stats import SessionStats, format_report
from .tracker import Decision, FocusState, FocusTracker

console = Console()


class Monitor:
    """监控器。

    不用 dataclass(slots=True)：本类要持有 mss/摄像头/httpx 等资源句柄，
    slots 生成的类没有 __dict__，__post_init__ 里无法动态挂属性。
    """

    def __init__(
        self,
        settings: Settings,
        store: Store,
        session_id: int,
        goal: str,
        on_event: "Callable[[str, dict], None] | None" = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.session_id = session_id
        self._goal = goal
        # 可选事件回调，供 GUI 使用；CLI 不传则只打印到终端
        self.on_event = on_event

        cfg = self.settings.app
        self.grabber = ScreenGrabber()
        self.camera = CameraReader(cfg.capture.camera_index) if cfg.monitor_camera else None
        self.local = LocalJudge(cfg.rules)
        self.judge = VisionJudge(self.settings.api)
        self.tracker = FocusTracker(cfg.judge)
        self.stats = SessionStats()
        self._pomodoro: Pomodoro | None = None
        self._break_started = 0.0
        # 本地日志（滚动写data/logs/ai_teacher.log）
        self.logger = journal.get_logger()
        # 供日志记录用的最近一次 tick 中间值
        self._last_person_present: bool | None = None
        self._last_image_bytes = 0
        self._last_error = ""
        self._stop = False
        self._stopped = False
        self._warned_cam_missing = False

    # ---- 目标可在运行中修改

    @property
    def goal(self) -> str:
        """当前学习目标。

        做成 property 而不是普通属性，这样 GUI 在运行中改目标能立刻生效——
        下一次 tick 送给模型的提示词就用新目标，不需要重启监控。
        """

        return self._goal

    @goal.setter
    def goal(self, value: str) -> None:
        value = (value or "").strip()
        if not value or value == self._goal:
            return
        old = self._goal
        self._goal = value
        journal.log_event(self.logger, "目标已修改", f"{old}-> {value}")
        console.print(f"[cyan]目标已更新：[/]{old} [dim]->[/] {value}")
        self._emit("goal_changed", goal=value, old=old)

    # ------------------------------------------------------------ 事件

    def _emit(self, kind: str, **payload: object) -> None:
        """广播事件。回调异常绝不能影响监控主流程。"""

        if self.on_event is None:
            return
        try:
            self.on_event(kind, payload)
        except Exception:
            pass

    def stop(self) -> None:
        """请求停止。线程安全，可在任意线程调用。"""

        self._stop = True

    # ------------------------------------------------------------ 主循环

    def run(self) -> None:
        self._install_signals()
        pom = self.settings.app.pomodoro

        console.print(
            f"[bold green]电子教官已启动[/]  目标：[bold]{self.goal}[/]\n"
            f"采样间隔 {self.settings.app.judge.interval_min_s:.0f}~"
            f"{self.settings.app.judge.interval_max_s:.0f}s 自适应，"
            f"宽限期 {self.settings.app.judge.grace_period_s:.0f}s"
        )
        if pom.enabled:
            console.print(
                f"[cyan]课堂模式[/] 每 {pom.focus_minutes:.0f} 分钟提醒休息 "
                f"{pom.break_minutes:.0f} 分钟，休息期间不抓拍"
            )
        if not self.settings.api.api_key:
            console.print(
                "[yellow]未配置 API key（.env 里设 AI_TEACHER_API_KEY）。"
                "将只走本地规则，模型判定会自动跳过。[/]"
            )

        journal.log_event(
            self.logger,
            "监控启动",
            f"目标={self.goal} | api={'有' if self.settings.api.api_key else '无'}"
            f" | 课堂模式={'开' if pom.enabled else '关'}"
            f" | 间隔 {self.settings.app.judge.interval_min_s:.0f}~"
            f"{self.settings.app.judge.interval_max_s:.0f}s",
        )
        self._emit(
            "started",
            goal=self.goal,
            has_key=bool(self.settings.api.api_key),
            pomodoro=pom.enabled,
        )

        self._pomodoro = Pomodoro(
            pom.focus_minutes, pom.break_minutes, enabled=pom.enabled
        )

        try:
            while not self._stop:
                # ---- 休息阶段：不抓拍、不判定，只倒计时
                if self._pomodoro.in_break:
                    self._run_break()
                    continue

                decision = self.tick()
                self._print_tick(decision)
                self._emit("tick", decision=decision, stats=self.stats.as_dict())

                # ---- 专注时长到了？切换到休息
                if pom.enabled:
                    self._pomodoro.update_elapsed()
                    if self._pomodoro.phase_expired():
                        self._pomodoro.advance()
                        self._emit("pomodoro", state=self._pomodoro.state().as_dict())
                        self._enter_break_phase()
                        continue

                if self._stop:
                    break
                # 分段sleep，保证 Ctrl+C / stop() 能及时响应
                self._sleep(decision.next_interval_s)
        except KeyboardInterrupt:
            console.print("\n[yellow]收到中断，正在退出…[/]")
        except Exception as exc:
            # 任何未预期的异常都要通知到 GUI，否则界面会一直转圈
            self._emit("error", message=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.shutdown()

    # ------------------------------------------------------------ 课堂模式

    def _enter_break_phase(self) -> None:
        """进入休息阶段：弹右下角提示，然后开始倒计时。"""

        pom = self.settings.app.pomodoro
        mins = pom.break_minutes
        if mins <= 0:
            return
        msg = f"已专注 {pom.focus_minutes:.0f} 分钟，休息 {mins:.0f} 分钟吧。\n休息期间不会抓拍。"
        journal.log_event(
            self.logger, "进入休息", f"{mins:.0f} 分钟，第 {self._pomodoro.rounds_done} 轮完成"
        )
        if pom.notify:
            alert.notify(title="该休息一下了", message=msg, sound=True)
        console.print(f"\n[cyan]{msg}[/]")
        self._emit(
            "pomodoro_break", message=msg, state=self._pomodoro.state().as_dict()
        )

    def _run_break(self) -> None:
        """休息阶段：只倒计时，不截图不拍照。

        每秒检查一次停止标志，保证「点停止」能立刻退出，
        不会傻等 10 分钟。
        """

        self._break_started = time.monotonic()

        while not self._stop:
            self._pomodoro.update_elapsed()
            if self._pomodoro.phase_expired():
                self._pomodoro.advance()
                st = self._pomodoro.state()
                self._pomodoro.add_break_time(st.total_s)
                pom = self.settings.app.pomodoro
                msg = f"休息结束，已完成 {self._pomodoro.rounds_done} 轮专注。继续加油。"
                if pom.notify:
                    alert.notify(title="休息结束", message=msg, sound=True)
                journal.log_event(self.logger, "休息结束", f"第 {self._pomodoro.rounds_done} 轮开始")
                console.print(f"[green]{msg}[/]\n")
                self._emit("pomodoro_focus", message=msg, state=st.as_dict())
                return
            self._sleep(1.0)
            # 每 10 秒广播一次状态，让界面上的倒计时走起来
            if int(time.monotonic() - self._break_started) % 10 == 0:
                self._emit("pomodoro", state=self._pomodoro.state().as_dict())

    def tick(self) -> Decision:
        """跑一个 tick。"""

        cfg = self.settings.app.capture
        screen = self.grabber.grab()
        self.stats.screenshots += 1

        camera = self.camera.read() if self.camera else None
        if self.camera:
            # 只统计真正开启了摄像头时的拍照次数
            if camera is not None:
                self.stats.camera_shots += 1
            else:
                self.stats.camera_miss += 1
        if camera is None and self.camera and not self._warned_cam_missing:
            console.print(
                "[yellow]摄像头不可用（被占用或无设备），将仅凭屏幕判定。[/]"
            )
            self._warned_cam_missing = True

        # ---- L0 本地规则
        signals = self.local.evaluate(
            screen, camera, cfg.diff_width, cfg.diff_threshold
        )

        # ---- L1模型判定（仅在L0 拿不准时才花钱）
        upload_cam = camera is not None and self.settings.app.upload_camera
        model_result = None
        if signals.verdict in (Verdict.SUSPECT, Verdict.UNKNOWN):
            payload_cam = camera if upload_cam else None
            jpeg = compose(screen, payload_cam, cfg)
            self.stats.image_bytes += len(jpeg)
            model_result = self.judge.judge(jpeg, self.goal)
            # 统计判定来源：成功 / 失败 / 因缺 key 跳过
            if model_result.ok:
                self.stats.api_calls += 1
            elif model_result.error == "no_api_key":
                self.stats.api_skipped += 1
            else:
                self.stats.api_failed += 1
        else:
            # L0 直接定案，完全没有花钱
            self.stats.local_hits += 1

        studying: bool | None = None
        reason = signals.reason
        # 供日志使用：这几个值在tick 内多处产生，先存到实例上
        self._last_person_present = signals.person_present
        self._last_image_bytes = len(jpeg) if model_result is not None else 0
        self._last_error = ""
        if model_result is not None:
            if model_result.ok:
                studying = model_result.studying
                reason = model_result.comment or reason
                self.stats.prompt_tokens += model_result.prompt_tokens
                self.stats.completion_tokens += model_result.completion_tokens
            else:
                # 调用失败时保守处理：不动状态，也不打扰用户
                self._last_error = model_result.error
                if model_result.error == "budget_exceeded":
                    # 预算护栏触发是「主动的保护」，不是故障，用不同措辞
                    reason = f"{reason}（已达调用预算，本次不再调用模型）"
                else:
                    reason = f"{reason}（模型不可用：{model_result.error[:30]}）"

        decision = self.tracker.update(
            l0=signals.verdict,
            reason=reason,
            model_studying=studying,
            used_api=model_result is not None and model_result.ok,
        )

        # ---- 统计判定结果
        if decision.state is FocusState.FOCUSING:
            self.stats.focus_ticks += 1
        elif decision.state is FocusState.DISTRACTED:
            self.stats.distracted_ticks += 1
        else:
            self.stats.unsure_ticks += 1

        # ---- 告警
        if decision.should_alert:
            self.stats.alerts += 1
            journal.log_alert(self.logger, decision.reason, self.stats.elapsed_s)
            alert.notify(
                title="⚠️ 疑似离开学习",
                message=f"已连续未在学习：{reason}\n目标：{self.goal}",
                sound=True,
            )
            self._emit("alert", reason=reason, state=decision.state.value)

        # ---- 落库
        self.store.add_tick(
            self.session_id,
            l0_verdict=signals.verdict.value,
            model_used=bool(model_result and model_result.ok),
            studying=studying,
            state=decision.state.value,
            reason=reason,
            interval_s=decision.next_interval_s,
            prompt_tokens=model_result.prompt_tokens if model_result else None,
            completion_tokens=model_result.completion_tokens if model_result else None,
        )
        return decision

    # ------------------------------------------------------------ 辅助

    def _sleep(self, seconds: float) -> None:
        """可中断的 sleep。"""

        end = time.monotonic() + seconds
        while time.monotonic() < end and not self._stop:
            time.sleep(min(0.5, max(0.0, end - time.monotonic())))

    def _install_signals(self) -> None:
        def handler(signum: int, frame: object) -> None:
            self._stop = True

        try:
            signal.signal(signal.SIGINT, handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):
            # 非主线程里运行时忽略
            pass

    def _state_mark(self, state: FocusState) -> str:
        return {
            FocusState.FOCUSING: "[green]学习中[/]",
            FocusState.UNSURE: "[yellow]不确定[/]",
            FocusState.DISTRACTED: "[red]分心[/]",
        }[state]

    def _print_tick(self, d: Decision) -> None:
        src = "模型" if d.used_api else "本地"
        console.print(
            f"{time.strftime('%H:%M:%S')} {self._state_mark(d.state)} "
            f"[dim]{src}[/] {d.reason} "
            f"[dim]下次 {d.next_interval_s:.0f}s[/]"
        )
        # 落本地日志，便于事后复盘。**不记录图片内容与请求体**
        journal.log_tick(
            self.logger,
            index=self.stats.screenshots,
            state=d.state.value,
            reason=d.reason,
            used_api=d.used_api,
            interval_s=d.next_interval_s,
            person_present=self._last_person_present,
            screenshot_bytes=self._last_image_bytes,
            elapsed_s=self.stats.elapsed_s,
            prompt_tokens=self.judge.last_prompt_tokens,
            completion_tokens=self.judge.last_completion_tokens,
            error=self._last_error,
        )

    def shutdown(self) -> None:
        # 幂等：GUI 可能在关窗和线程退出时各调一次
        if self._stopped:
            return
        self._stopped = True
        self._stop = True

        self.grabber.close()
        if self.camera:
            self.camera.close()
        self.judge.close()
        self.store.end_session(self.session_id)

        st = self.stats
        st.prompt_tokens = self.judge.total_prompt_tokens or st.prompt_tokens
        st.completion_tokens = self.judge.total_completion_tokens or st.completion_tokens
        if self._pomodoro is not None and self._pomodoro.enabled:
            st.pomodoro_used = True
            st.pomodoro_rounds = self._pomodoro.rounds_done
            if self._pomodoro.break_total_s:
                st.pause_seconds = self._pomodoro.break_total_s

        # 先落库会话汇总，便于事后用 report 命令查历史
        try:
            self.store.save_session_stats(self.session_id, st.as_dict())
        except Exception:
            pass  # 统计写库失败不能影响正常退出

        # 完整统计落日志，便于事后查账
        summary = st.as_dict()
        from .stats import is_peak_time

        summary["peak"] = is_peak_time()
        journal.dump_session(self.logger, summary)

        self._emit("finished", stats=st.as_dict(), goal=self.goal)

        rows = format_report(st, self.goal)
        table = Table(
            title=f"本次学习会话 · {self.goal}",
            show_header=False,
            box=None,
            title_style="bold",
        )
        for label, value in rows:
            if not value:  # 分组标题
                table.add_row(f"[bold cyan]{label}[/]", "")
            else:
                table.add_row(label, value)
        console.print()
        console.print(table)

    def __enter__(self) -> "Monitor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
