"""tkinter 图形界面。

设计要点：
- 监控跑在后台线程（截图/摄像头/网络都是阻塞操作，不能放主线程，否则界面卡死）
- 后台线程只往queue 里塞事件，UI 定时轮询取事件 —— 跨线程更新 tkinter 的正解
- 关窗时先 request_stop 再等线程收尾，不硬杀线程
- 无 tkinter 时给出可读的降级提示，而不是 ImportError 崩栈
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from .config import get_settings
from .db import Store
from .monitor import Monitor
from .tracker import FocusState

# 配色
BG = "#1e1f26"
FG = "#e6e6e6"
SUB = "#9a9aa0"
GREEN = "#5dc98a"
AMBER = "#e8b64c"
RED = "#e5646e"
BLUE = "#5b9dd9"
CARD = "#2a2b34"

# ---------------------------------------------------------------- 字体
# 统一在这里定义，不要在控件里硬编码字号：
# 用户反馈过「字体太小」，散落各处的font=("...", 9) 改起来容易漏，
# 改成命名字体后只改这一处就能全局放大。
FONT_FAMILY = "Microsoft YaHei UI"
MONO_FAMILY = "Consolas"

# 基准字号。系统若有 133% 缩放，这里的 pt 值需要相应补偿，
# 否则在高分屏上会显得偏小（tkinter 的 size 单位是 pt，受 DPI 影响）。
_S = 1  # 缩放系数，统一放大时只改这里

F_TITLE = (FONT_FAMILY, 22 * _S, "bold")  # 主标题
F_STATE = (FONT_FAMILY, 34 * _S, "bold")  # 状态大字
F_INPUT = (FONT_FAMILY, 15 * _S)  # 目标输入框
F_BTN = (FONT_FAMILY, 13 * _S, "bold")  # 主按钮
F_BTN2 = (FONT_FAMILY, 13 * _S)  # 次按钮
F_BODY = (FONT_FAMILY, 12 * _S)  # 正文
F_LABEL = (FONT_FAMILY, 11 * _S)  # 小标签
F_SUBTITLE = (FONT_FAMILY, 15 * _S, "bold")  # 报告窗口标题
F_MONO = (MONO_FAMILY, 12 * _S)  # 计时器/日志

# 日志区要够宽，否则长文本换行后行数暴涨，看着挤
LOG_LINES = 12

STATE_TEXT = {
    FocusState.FOCUSING: ("学习中", GREEN),
    FocusState.UNSURE: ("不确定", AMBER),
    FocusState.DISTRACTED: ("分心", RED),
}


class MonitorApp:
    """电子教官主窗口。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.settings = get_settings()

        self.events: queue.Queue = queue.Queue()
        self.monitor: Monitor | None = None
        self.thread: threading.Thread | None = None
        self._closing = False

        # 配置里没有目标时不填任何内容，改为显示灰色占位提示，
        # 用户一开始输入就自动清空占位符，避免把提示语当成真目标保存。
        self.goal_var = tk.StringVar(value=self.settings.app.goal or "")
        self._goal_placeholder = "例如：刷 LeetCode / 看机器学习网课 / 读论文"
        self._goal_is_placeholder = not self.goal_var.get().strip()
        self.state_var = tk.StringVar(value="未启动")
        self.timer_var = tk.StringVar(value="00:00")
        self.stat_var = tk.StringVar(value="采样 0 次· 模型调用 0 次 · 告警 0 次")
        self.cost_var = tk.StringVar(value="预计花费 ¥0.0000")
        self.camera_var = tk.BooleanVar(value=self.settings.app.monitor_camera)
        self.upload_var = tk.BooleanVar(value=self.settings.app.upload_camera)

        # 课堂模式
        _pom = self.settings.app.pomodoro
        self.pomo_var = tk.BooleanVar(value=_pom.enabled)
        self.focus_min_var = tk.IntVar(value=int(_pom.focus_minutes))
        self.break_min_var = tk.IntVar(value=int(_pom.break_minutes))
        self.pomo_state: dict | None = None  # 运行中的倒计时状态

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # UI 轮询：把后台事件搬到主线程
        self.root.after(120, self._poll)

    # ------------------------------------------------------------ 界面

    def _build(self) -> None:
        self.root.title("电子教官")
        self.root.configure(bg=BG)
        self.root.geometry("660x760")
        self.root.minsize(560, 640)

        pad = tk.Frame(self.root, bg=BG)
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        # ---- 标题
        tk.Label(
            pad, text="电子教官", bg=BG, fg=FG, font=F_TITLE
        ).pack(anchor="w")
        tk.Label(
            pad,
            text="设定目标后启动，程序会定期截屏与拍照判断学习状态",
            bg=BG,
            fg=SUB,
            font=F_BODY,
        ).pack(anchor="w", pady=(2, 14))

        # ---- 目标卡片
        goal_card = self._card(pad)
        tk.Label(
            goal_card, text="学习目标", bg=CARD, fg=SUB, font=F_LABEL
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.goal_entry = tk.Entry(
            goal_card,
            textvariable=self.goal_var,
            bg=BG,
            fg=FG,
            insertbackground=FG,
            font=F_INPUT,
            relief="flat",
            highlightthickness=1,
            highlightbackground="#3d3e4a",
        )
        self.goal_entry.pack(fill="x", padx=14, pady=(0, 12))
        self.goal_entry.bind("<Return>", lambda _e: self._start())
        self.goal_entry.bind("<FocusIn>", self._on_goal_focus_in)
        self.goal_entry.bind("<FocusOut>", self._on_goal_focus_out)
        self._sync_goal_placeholder()

        # ---- 状态卡片
        state_card = self._card(pad)
        tk.Label(
            state_card, text="当前状态", bg=CARD, fg=SUB, font=F_LABEL
        ).pack(anchor="w", padx=14, pady=(12, 4))
        self.state_label = tk.Label(
            state_card,
            textvariable=self.state_var,
            bg=CARD,
            fg=SUB,
            font=F_STATE,
        )
        self.state_label.pack(anchor="w", padx=14)
        tk.Label(
            state_card, textvariable=self.timer_var, bg=CARD, fg=SUB, font=F_MONO
        ).pack(anchor="w", padx=14, pady=(2, 10))
        tk.Label(
            state_card, textvariable=self.stat_var, bg=CARD, fg=SUB, font=F_BODY
        ).pack(anchor="w", padx=14)
        tk.Label(
            state_card, textvariable=self.cost_var, bg=CARD, fg=SUB, font=F_BODY
        ).pack(anchor="w", padx=14, pady=(2, 12))

        # ---- 摄像头开关
        cam_card = self._card(pad)
        tk.Checkbutton(
            cam_card,
            text="使用摄像头判断是否在场",
            variable=self.camera_var,
            command=self._on_camera_toggle,
            bg=CARD,
            fg=FG,
            selectcolor=CARD,
            activebackground=CARD,
            activeforeground=FG,
            font=F_LABEL,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))
        self.upload_chk = tk.Checkbutton(
            cam_card,
            text="将摄像头画面上传给模型（关闭则仅本地检测，不外传）",
            variable=self.upload_var,
            bg=CARD,
            fg=SUB,
            selectcolor=CARD,
            activebackground=CARD,
            activeforeground=FG,
            font=F_LABEL,
            anchor="w",
        )
        self.upload_chk.pack(fill="x", padx=14, pady=(0, 10))

        # ---- 课堂模式（番茄钟）
        self._build_pomodoro_card(pad)

        # ---- 运行中改目标：不用重启，下一个 tick 就用新目标
        self.retarget_btn = tk.Button(
            pad,
            text="修改当前目标",
            command=self._retarget,
            bg=CARD,
            fg=FG,
            activebackground="#3a3b46",
            font=F_BODY,
            relief="flat",
            cursor="hand2",
            state="disabled",
        )
        self.retarget_btn.pack(fill="x", pady=(0, 10))

        # ---- 按钮
        btn_row = tk.Frame(pad, bg=BG)
        btn_row.pack(fill="x", pady=14)
        self.start_btn = tk.Button(
            btn_row,
            text="启动监控",
            command=self._start,
            bg=GREEN,
            fg="#101216",
            activebackground="#4db377",
            font=F_BTN,
            relief="flat",
            cursor="hand2",
            height=1,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.stop_btn = tk.Button(
            btn_row,
            text="停止",
            command=self._stop,
            bg=CARD,
            fg=FG,
            activebackground="#3a3b46",
            font=F_BTN2,
            relief="flat",
            cursor="hand2",
            state="disabled",
            height=1,
        )
        self.stop_btn.pack(side="left", fill="x", expand=True)

        # ---- 日志
        tk.Label(pad, text="运行日志", bg=BG, fg=SUB, font=F_LABEL).pack(
            anchor="w"
        )
        log_frame = tk.Frame(pad, bg=CARD)
        log_frame.pack(fill="both", expand=True, pady=(4, 0))
        self.log = tk.Text(
            log_frame,
            bg=CARD,
            fg=SUB,
            insertbackground=FG,
            font=F_MONO,
            relief="flat",
            wrap="word",
            padx=8,
            pady=6,
            height=LOG_LINES,
        )
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ---- API 状态
        self.api_lbl = tk.Label(pad, bg=BG, fg=SUB, font=F_LABEL, anchor="w")
        self.api_lbl.pack(fill="x", pady=(8, 0))
        self._refresh_api_status()

    def _build_pomodoro_card(self, parent: tk.Widget) -> None:
        """课堂模式卡片：开关 + 时长设置 + 倒计时。"""

        pom = self.settings.app.pomodoro
        card = self._card(parent)

        tk.Checkbutton(
            card,
            text="课堂模式（定时提醒休息）",
            variable=self.pomo_var,
            command=self._on_pomodoro_toggle,
            bg=CARD,
            fg=FG,
            selectcolor=CARD,
            activebackground=CARD,
            activeforeground=FG,
            font=F_BODY,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 4))

        # 时长设置：两个 Spinbox
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x", padx=14, pady=(0, 4))

        tk.Label(row, text="专注", bg=CARD, fg=SUB, font=F_LABEL).pack(side="left")
        tk.Spinbox(
            row,
            from_=1,
            to=120,
            textvariable=self.focus_min_var,
            width=4,
            bg=BG,
            fg=FG,
            buttonbackground=CARD,
            font=F_LABEL,
            relief="flat",
            highlightthickness=1,
            highlightbackground="#3d3e4a",
        ).pack(side="left", padx=(4, 2))
        tk.Label(row, text="分钟", bg=CARD, fg=SUB, font=F_LABEL).pack(side="left")
        tk.Label(row, text="休息", bg=CARD, fg=SUB, font=F_LABEL).pack(side="left", padx=(14, 0))
        tk.Spinbox(
            row,
            from_=0,
            to=60,
            textvariable=self.break_min_var,
            width=4,
            bg=BG,
            fg=FG,
            buttonbackground=CARD,
            font=F_LABEL,
            relief="flat",
            highlightthickness=1,
            highlightbackground="#3d3e4a",
        ).pack(side="left", padx=(4, 2))
        tk.Label(row, text="分钟", bg=CARD, fg=SUB, font=F_LABEL).pack(side="left")

        self.pomo_detail = tk.Label(
            card,
            text="",
            bg=CARD,
            fg=SUB,
            font=F_LABEL,
            anchor="w",
        )
        self.pomo_detail.pack(fill="x", padx=14, pady=(4, 10))

        self._on_pomodoro_toggle()

    def _card(self, parent: tk.Widget) -> tk.Frame:
        f = tk.Frame(parent, bg=CARD, highlightthickness=0)
        f.pack(fill="x", pady=(0, 10))
        return f

    def _refresh_api_status(self) -> None:
        if self.settings.api.api_key:
            txt = f"模型：{self.settings.api.model} · API key 已配置"
            self.api_lbl.configure(fg=SUB, text=txt)
        else:
            self.api_lbl.configure(
                fg=AMBER,
                text="未配置 API key —— 将在项目根目录建 .env 写入 AI_TEACHER_API_KEY 启用模型判定",
            )

    # ------------------------------------------------------------ 日志

    def _log(self, text: str, color: str = SUB) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.insert("end", f"{ts}  {text}\n")
        # 分色：给整行上色
        line_no = int(self.log.index("end-1c").split(".")[0])
        self.log.tag_add(f"c_{color}", f"{line_no}.0", f"{line_no}.end")
        self.log.tag_config(f"c_{color}", foreground=color)
        self.log.see("end")

    # ------------------------------------------------------------ 占位提示

    def _sync_goal_placeholder(self) -> None:
        """空目标时显示灰字占位符，有内容时恢复正常色。"""

        if self._goal_is_placeholder:
            self.goal_var.set(self._goal_placeholder)
            self.goal_entry.configure(fg="#6f6f78")
        else:
            self.goal_entry.configure(fg=FG)

    def _on_goal_focus_in(self, _e: object) -> None:
        if self._goal_is_placeholder:
            self.goal_var.set("")
            self._goal_is_placeholder = False
            self.goal_entry.configure(fg=FG)

    def _on_goal_focus_out(self, _e: object) -> None:
        if not self.goal_var.get().strip():
            self._goal_is_placeholder = True
            self._sync_goal_placeholder()

    def _current_goal(self) -> str:
        """取真实目标：占位符状态下一律当作空。"""

        if self._goal_is_placeholder:
            return ""
        return self.goal_var.get().strip()

    # ------------------------------------------------------------ 控制

    def _on_camera_toggle(self) -> None:
        """摄像头开关会影响上传项是否可用。"""

        state = "normal" if self.camera_var.get() else "disabled"
        self.upload_chk.configure(state=state)
        if not self.camera_var.get():
            self.upload_var.set(False)
        self.settings.app.monitor_camera = self.camera_var.get()
        self.settings.app.upload_camera = self.upload_var.get()

    # ------------------------------------------------------------ 课堂模式

    def _retarget(self) -> None:
        """运行中修改目标：下一个 tick 就生效，不用重启。

        用户明确要求「运行过程中可以改目标，不影响运行流程」。
        实现上依赖 Monitor.goal 的 property setter——只改内存值，
        不动线程、不动采样节奏，下一次 judge() 会自动带上新目标。
        """

        if not (self.thread and self.thread.is_alive()) or not self.monitor:
            self._set_state("未在运行中，无法修改目标", AMBER)
            return

        new_goal = self._current_goal()
        if not new_goal:
            self._set_state("目标不能为空", AMBER)
            return
        if new_goal == self.monitor.goal:
            self._log(f"目标未变：{new_goal}", SUB)
            return

        # setter 内部会记日志并广播 goal_changed 事件
        self.monitor.goal = new_goal
        self._log(f"目标已改为「{new_goal}」，下一个判定即刻生效", BLUE)
        # 直接设状态区，不依赖异步事件——事件可能在队列里被后续 tick 覆盖
        self._set_state(f"目标已改为 {new_goal}", BLUE)
        # 同步落盘，下次启动仍用这个目标
        self.settings.app.goal = new_goal
        try:
            self.settings.save()
        except Exception as exc:
            self._log(f"配置保存失败：{exc}", RED)

    def _on_pomodoro_toggle(self) -> None:
        """课堂模式开关。关闭时收起说明文字，减少视觉干扰。"""

        if not hasattr(self, "pomo_detail"):
            return
        if self.pomo_var.get():
            f = self.focus_min_var.get() or 40
            b = self.break_min_var.get() or 0
            if b > 0:
                txt = f"每专注 {f} 分钟，提醒休息 {b} 分钟（期间不抓拍）"
            else:
                txt = f"每专注 {f} 分钟，提醒一次（不安排休息）"
            self.pomo_detail.configure(fg=SUB, text=txt)
        else:
            self.pomo_detail.configure(text="")

    def _render_pomodoro(self, st: dict) -> None:
        """更新倒计时显示。"""

        if not st:
            return
        self.pomo_state = st
        if not self.pomo_var.get():
            return
        # 只有监控真正在跑时才改状态区，避免停止后仍被残留事件影响
        running = bool(self.thread and self.thread.is_alive())
        label = st.get("phase_label", "")
        remain = st.get("remaining_fmt", "")
        rounds = st.get("rounds_done", 0)
        if st.get("phase") == "break":
            self.pomo_detail.configure(
                fg=AMBER,
                text=f"● {label} {remain}　·　第 {rounds} 轮已完成 · 期间不抓拍",
            )
            if running:
                # 状态区跟着倒计时走，不依赖事件是否被漏处理
                self._set_state("休息中", AMBER)
        else:
            self.pomo_detail.configure(
                fg=SUB, text=f"● {label} {remain}　·　已完成 {rounds} 轮"
            )

    def _start(self) -> None:
        if self.thread and self.thread.is_alive():
            return

        goal = self._current_goal()
        if not goal:
            # 不用 messagebox：弹窗会打断操作且反复触发很难看。
            # 改成内联提示 + 自动聚焦目标框，让用户直接改正。
            self._set_state("请先设定目标", AMBER)
            self._log("目标为空，无法启动。请在上面的输入框里填写学习目标。", AMBER)
            self.goal_entry.focus_set()
            self.goal_entry.select_range(0, "end")
            return

        # 把界面上的设置落盘，保证下次 CLI 启动也一致
        self.settings.app.goal = goal
        self.settings.app.monitor_camera = self.camera_var.get()
        self.settings.app.upload_camera = self.upload_var.get() and self.camera_var.get()
        # 课堂模式设置
        _p = self.settings.app.pomodoro
        _p.enabled = self.pomo_var.get()
        try:
            _p.focus_minutes = float(max(1, self.focus_min_var.get() or 40))
            _p.break_minutes = float(max(0, self.break_min_var.get()))
        except (tk.TclError, ValueError):
            pass  # Spinbox 里输入了非数字，保持原值
        try:
            self.settings.save()
        except Exception as exc:
            self._log(f"配置保存失败：{exc}", RED)

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_state("运行中", BLUE)
        self._log(f"目标「{goal}」· 开始监控", BLUE)

        def on_event(kind: str, payload: dict) -> None:
            # 运行在工作线程，只入队不碰 UI
            self.events.put((kind, payload))

        def worker() -> None:
            store = None
            try:
                store = Store()
                sid = store.start_session(goal)
                # Monitor 构造会开摄像头/网络，放后台线程执行
                mon = Monitor(
                    settings=self.settings,
                    store=store,
                    session_id=sid,
                    goal=goal,
                    on_event=on_event,
                )
                self.monitor = mon
                mon.run()
            except Exception as exc:
                self.events.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
            finally:
                # Monitor 正常结束时会自己关；构造失败时这里兜底，避免泄漏句柄
                if store is not None:
                    try:
                        store.close()
                    except Exception:
                        pass
                self.events.put(("done", {}))

        self.thread = threading.Thread(target=worker, daemon=True, name="monitor")
        self.thread.start()

    def _stop(self) -> None:
        if not self.thread or not self.thread.is_alive():
            return
        if self.monitor:
            self.monitor.stop()
        self._log("正在停止…", AMBER)
        self.stop_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")
        self.retarget_btn.configure(state="disabled")

    def _set_state(self, text: str, color: str) -> None:
        self.state_var.set(text)
        self.state_label.configure(fg=color)

    # ------------------------------------------------------------ 事件泵

    def _poll(self) -> None:
        """把后台事件搬到主线程更新界面。"""

        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass

        # 计时器
        if self.thread and self.thread.is_alive():
            m = self.monitor
            if m is not None:
                s = int(m.tracker.elapsed_s())
                self.timer_var.set(f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}")

        if not self._closing:
            self.root.after(120, self._poll)

    def _render_stats(self, st: dict) -> None:
        """刷新状态卡上的实时统计行。"""

        if not st:
            return
        shots = st.get("screenshots", 0)
        cam = st.get("camera_shots", 0)
        local = st.get("local_hits", 0)
        api = st.get("api_calls", 0)
        ratio = (st.get("local_ratio") or 0) * 100

        self.stat_var.set(
            f"截图 {shots} 次 · 拍照 {cam} 次 · 本地判定 {local} / 模型判定 {api}"
        )
        self.cost_var.set(
            f"本地拦截率 {ratio:.0f}%　·　告警 {st.get('alerts', 0)} 次"
            f"　·　花费 ¥{st.get('est_cost', 0):.5f}"
        )

    def _show_report(self, st: dict) -> None:
        """会话结束时的完整统计窗口。"""

        from .stats import format_report_from_dict

        goal = self.goal_var.get().strip() or "学习"
        win = tk.Toplevel(self.root)
        win.title("本次学习报告")
        win.configure(bg=BG)
        win.geometry("560x640")
        win.minsize(460, 420)
        win.transient(self.root)

        frm = tk.Frame(win, bg=BG)
        frm.pack(fill="both", expand=True, padx=20, pady=16)

        tk.Label(
            frm,
            text=f"学习报告 · {goal}",
            bg=BG,
            fg=FG,
            font=F_SUBTITLE,
        ).pack(anchor="w", pady=(0, 4))
        tk.Label(
            frm,
            text=f"总时长 {st.get('elapsed_fmt', '00:00')}",
            bg=BG,
            fg=SUB,
            font=F_BODY,
        ).pack(anchor="w", pady=(0, 10))

        # 统计缺字段时降级为一行提示，不能让报告窗口开不出来
        try:
            rows = format_report_from_dict(st)
        except (KeyError, TypeError) as exc:
            rows = [("统计异常", str(exc)[:60])]

        # 报告可能很长（分组标题 + 十几行），必须放进可滚动区域，
        # 否则固定高度窗口里下半部分会被裁掉且无法查看。
        body = tk.Frame(frm, bg=BG)
        body.pack(fill="both", expand=True)

        canvas = tk.Canvas(body, bg=CARD, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=CARD)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_resize(event: object) -> None:
            # 宽度跟随窗口，高度按内容自适应
            canvas.itemconfigure(win_id, width=event.width)
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", on_resize)

        # 滚轮支持：Windows 上 MouseWheel 正常，但焦点不在窗口内时要手动绑定
        def on_wheel(event: object) -> None:
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")

        for widget in (canvas, inner):
            widget.bind("<MouseWheel>", on_wheel)
            widget.bind("<Button-4>", on_wheel)
            widget.bind("<Button-5>", on_wheel)

        for label, value in rows:
            if not value:
                tk.Label(
                    inner, text=label, bg=CARD, fg=BLUE,
                    font=F_BODY, anchor="w",
                ).pack(fill="x", padx=14, pady=(10, 3))
            else:
                row = tk.Frame(inner, bg=CARD)
                row.pack(fill="x", padx=14, pady=2)
                tk.Label(
                    row, text=label, bg=CARD, fg=SUB,
                    font=F_BODY, anchor="w", width=16,
                ).pack(side="left")
                tk.Label(
                    row, text=value, bg=CARD, fg=FG,
                    font=F_BODY, anchor="w",
                ).pack(side="left")

        tk.Button(
            frm, text="关闭", command=win.destroy, bg=CARD, fg=FG,
            activebackground="#3a3b46", relief="flat", font=F_BODY,
            cursor="hand2",
        ).pack(fill="x", pady=(12, 0), ipady=6)

        # 告诉用户日志在哪，事后才能复盘
        from .journal import LOG_DIR

        tk.Label(
            frm,
            text=f"详细日志：{LOG_DIR / 'ai_teacher.log'}",
            bg=BG,
            fg=SUB,
            font=F_LABEL,
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

    def _handle(self, kind: str, p: dict) -> None:
        if kind == "tick":
            d = p["decision"]
            text, color = STATE_TEXT.get(d.state, ("未知", SUB))
            self._set_state(text, color)
            src = "模型" if d.used_api else "本地"
            self._log(f"{text} · {src}判定 · {d.reason}", color)
            self._render_stats(p.get("stats", {}))

        elif kind == "alert":
            self._log(f"⚠ 告警：{p.get('reason', '')}", RED)

        elif kind == "started":
            if not p.get("has_key"):
                self._log("未配置 API key，仅使用本地规则判定", AMBER)
            # 运行中可改目标
            self.retarget_btn.configure(state="normal")
            if p.get("pomodoro"):
                _p = self.settings.app.pomodoro
                self._log(
                    f"课堂模式已开启 · 每 {_p.focus_minutes:.0f} 分钟提醒休息 "
                    f"{_p.break_minutes:.0f} 分钟",
                    BLUE,
                )

        elif kind == "goal_changed":
            # 界面提示已在 _retarget() 里同步做完，这里故意什么都不做。
            # 异步事件晚到一步，若在这里改状态区会把刚设的提示覆盖掉。
            pass

        elif kind == "pomodoro":
            self._render_pomodoro(p.get("state", {}))

        elif kind == "pomodoro_break":
            self._log(f"● 进入休息 · {p.get('message', '')}", AMBER)
            self._set_state("休息中", AMBER)
            st = p.get("state")
            if st:
                self._render_pomodoro(st)

        elif kind == "pomodoro_focus":
            self._log(f"● 休息结束 · {p.get('message', '')}", GREEN)
            self._set_state("运行中", BLUE)
            self._render_pomodoro(p.get("state", {}))

        elif kind == "finished":
            st = p.get("stats", {})
            self._render_stats(st)
            self._log(
                f"会话结束 · 截图 {st.get('screenshots', 0)} 次 · "
                f"本地判定 {st.get('local_hits', 0)} / 模型判定 {st.get('api_calls', 0)} · "
                f"专注 {(st.get('focus_ratio') or 0) * 100:.0f}% · "
                f"告警 {st.get('alerts', 0)} 次 · 花费 ¥{st.get('est_cost', 0):.5f}",
                BLUE,
            )
            self._show_report(st)

        elif kind == "error":
            self._log(f"错误：{p.get('message', '')}", RED)
            messagebox.showerror("运行出错", str(p.get("message", "未知错误"))[:400])

        elif kind == "done":
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.monitor = None
            if not self._closing:
                self._set_state("已停止", SUB)
                self.timer_var.set("00:00:00")

    # ------------------------------------------------------------ 关闭

    def _on_close(self) -> None:
        """关窗：先让监控线程自己收尾，避免数据库/摄像头句柄泄漏。"""

        if self.thread and self.thread.is_alive():
            if not messagebox.askokcancel("退出", "监控正在运行，确定要退出吗？"):
                return
            self._closing = True
            if self.monitor:
                self.monitor.stop()
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            # 给线程一点时间优雅收尾
            self.thread.join(timeout=8.0)
        self._closing = True
        self.root.destroy()


def _enable_dpi_awareness() -> None:
    """开启 Per-Monitor DPI 感知。

    **必须在创建 Tk() 之前调用**。tkinter 在Tk() 里完成自身的 DPI 初始化，
    之后再设置进程 DPI 感知已无效果——之前就踩过这个坑：
    调用位置写在了 Tk() 后面，导致 133% 缩放下字体偏小且发虚。
    """

    if sys.platform != "win32":
        return
    try:
        # 先试 Win10 1703+ 的 Per-Monitor V2
        from ctypes import windll  # type: ignore[attr-defined]

        try:
            windll.user32.SetProcessDpiAwarenessContext(-4)  # PER_MONITOR_AWARE_V2
            return
        except Exception:
            pass
        try:
            windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
            return
        except Exception:
            pass
        windll.user32.SetProcessDPIAware()  # 最老API兜底
    except Exception:
        pass


def main() -> None:
    """GUI 入口。"""

    # 必须在 Tk() 之前
    _enable_dpi_awareness()

    try:
        root = tk.Tk()
    except Exception as exc:
        print(f"[错误] 无法创建图形窗口：{exc}")
        print("如果你在无桌面的环境里，请改用命令行：uv run ai-teacher start")
        return

    MonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
