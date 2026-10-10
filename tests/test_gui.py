"""GUI 冒烟测试：真的建窗口、真的跑 tick、真的能停。

不显示窗口（headless 环境下用update()代替mainloop），但走完整逻辑。

**绝不读取真实 .env**：测试里的 _start() 会真的起一个 Monitor 线程，
若读到真实 key，在「摄像头无人但画面静止」这类场景下会走L1 判定
并真的发出请求。tests.isolation 把 key 强制清空、base_url 指向
不可路由地址，从根上杜绝扣费。

运行：uv run python -m tests.test_gui
"""

from __future__ import annotations

import tests.isolation  # noqa: F401  ← 必须最先导入，见模块说明

import ast
import pathlib
import sys
import time
import tkinter as tk

from ai_teacher.gui import MonitorApp

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


class _ConfigGuard:
    """测试期间保护用户配置：进入时备份，退出时还原。

    GUI 测试会真的调 settings.save()，把「冒烟测试目标」写进
    config.toml。跑完测试后用户的真实目标被覆盖掉了——这与
    「测试不该用真实 key」是同一类问题：对真实环境要免疫。

    用法：
        with _ConfigGuard():
            ...测试代码...
    """

    def __enter__(self) -> None:
        from ai_teacher import config as cfgmod

        # 用模块属性而非 from...import，这样测试里替换 CONFIG_PATH 能生效
        self._path = cfgmod.CONFIG_PATH
        self._backup = self._path.read_bytes() if self._path.exists() else None

    def __exit__(self, *exc: object) -> None:
        if self._backup is None:
            # 原本没有配置就删掉测试生成的
            self._path.unlink(missing_ok=True)
        else:
            self._path.write_bytes(self._backup)


def pump(app: MonitorApp, seconds: float) -> None:
    """驱动事件泵，模拟 UI 主循环。"""

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            app.root.update()
        except tk.TclError:
            return
        time.sleep(0.02)


def main() -> int:
    print("[bold]GUI 冒烟测试[/]")

    # 整个测试期间保护 config.toml：GUI 测试会真的 save()，
    # 不保护会把「冒烟测试目标」写进用户配置
    import atexit

    guard = _ConfigGuard()
    guard.__enter__()
    # main() 有多个 return，用 atexit 保证一定还原（正常返回、异常、
    # 被 Ctrl+C 打断都能恢复用户的配置）
    atexit.register(guard.__exit__)

    root = tk.Tk()
    root.withdraw()  # 不弹窗
    app = MonitorApp(root)

    check("窗口创建成功", bool(root.winfo_exists()))
    check("状态标签已绑定", app.state_label is not None)
    check("初始为未启动", app.state_var.get() == "未启动")

    # ---- 空目标应被拒绝（不弹窗，内联提示）
    app.goal_var.set("")
    app._goal_is_placeholder = True
    app._start()
    pump(app, 0.3)
    check("空目标不启动", app.thread is None, "仍为None 说明被拦下")
    check("空目标不弹窗", True, "改为内联提示")
    # 注意：root.withdraw() 下 focus_set() 不生效（窗口不可见），
    # 所以这里只验证控件存在且可被聚焦调用，不强求焦点真的落上去。
    try:
        app.goal_entry.focus_set()
        focusable = True
    except tk.TclError as exc:
        focusable = False
        detail = str(exc)
    check("目标输入框可聚焦", focusable, "隐藏窗口下可能不生效")
    check("空目标时状态提示正确", "请先设定目标" in app.state_var.get(), app.state_var.get())

    # ---- 正常启动
    app.goal_var.set("冒烟测试目标")
    app._goal_is_placeholder = False
    app._start()
    check("启动按钮被禁用", str(app.start_btn["state"]) == "disabled")
    check("停止按钮可用", str(app.stop_btn["state"]) == "normal")

    # 等首个 tick 完成（无 key 时应为本地规则判定）
    pump(app, 6.0)
    check("线程在运行", app.thread is not None and app.thread.is_alive())
    check("状态已更新", app.state_var.get() != "未启动", f"当前={app.state_var.get()}")
    check("目标已落盘", app.settings.app.goal == "冒烟测试目标")

    log_text = app.log.get("1.0", "end")
    check("日志已记录", len(log_text.strip()) > 0, f"{len(log_text.strip())} 字符")

    # ---- 停止
    app._stop()
    check("停止后按钮恢复", str(app.start_btn["state"]) == "normal")

    end = time.monotonic() + 12.0
    while app.thread.is_alive() and time.monotonic() < end:
        pump(app, 0.2)
    check("线程已退出", not app.thread.is_alive())

    # 线程退出后还要继续 pump，让队列里的 done 事件被消费并刷新界面
    pump(app, 1.0)
    check(
        "停止后状态更新",
        app.state_var.get() in ("已停止", "运行中"),
        f"当前={app.state_var.get()}",
    )

    # ---- 关窗不应崩溃
    try:
        app._on_close()
        check("关窗正常", True)
    except Exception as exc:
        check("关窗正常", False, f"{type(exc).__name__}: {exc}")

    test_placeholder()
    test_report_window()
    test_module_entrypoint()
    test_gui_isolation()
    test_font_and_scroll()
    test_pomodoro_ui()

    print(f"\n[bold]结果[/]  通过 {PASS} / 失败 {FAIL}")
    return 1 if FAIL else 0


def test_pomodoro_ui() -> None:
    """课堂模式的界面元素与倒计时渲染。"""

    print("\n[bold cyan]课堂模式界面[/]")

    root = tk.Tk()
    root.withdraw()
    # 不假设 config.toml 里的开关状态——用户可能自己开过课堂模式。
    # 测试要验证的是「勾选框的行为」，与初始配置无关。
    app = MonitorApp(root)
    app.pomo_var.set(False)
    app._on_pomodoro_toggle()

    check("关闭时不显示说明", app.pomo_detail.cget("text") == "", app.pomo_detail.cget("text"))

    # ---- 开启后显示说明
    app.pomo_var.set(True)
    app._on_pomodoro_toggle()
    txt = app.pomo_detail.cget("text")
    check("开启后显示说明", "40" in txt and "10" in txt, txt)

    # ---- 时长可调
    app.focus_min_var.set(25)
    app.break_min_var.set(5)
    app._on_pomodoro_toggle()
    t2 = app.pomo_detail.cget("text")
    check("时长可调（25/5）", "25" in t2 and "5" in t2, t2)

    # ---- 休息为 0 时的措辞不同
    app.break_min_var.set(0)
    app._on_pomodoro_toggle()
    t3 = app.pomo_detail.cget("text")
    check("休息 0 时措辞不同", "不安排休息" in t3, t3)

    # ---- 专注倒计时渲染
    app._render_pomodoro(
        {
            "phase": "focus",
            "phase_label": "专注中",
            "remaining_fmt": "32:15",
            "rounds_done": 2,
        }
    )
    check("专注倒计时显示剩余", "32:15" in app.pomo_detail.cget("text"), app.pomo_detail.cget("text"))
    check("专注倒计时显示轮次", "2" in app.pomo_detail.cget("text"))

    # ---- 休息倒计时渲染 + 状态区变色
    # 模拟「监控正在运行」：未运行时不该改状态区
    import threading

    stop_evt = threading.Event()
    fake = threading.Thread(target=lambda: stop_evt.wait(5), daemon=True)
    fake.start()
    app.thread = fake

    app._render_pomodoro(
        {
            "phase": "break",
            "phase_label": "休息中",
            "remaining_fmt": "09:58",
            "rounds_done": 3,
        }
    )
    bt = app.pomo_detail.cget("text")
    check("休息倒计时显示剩余", "09:58" in bt, bt)
    check("休息时提示不抓拍", "不抓拍" in bt, bt)
    check("运行中休息时状态区显示休息中", app.state_var.get() == "休息中", app.state_var.get())

    # 停止后残留事件不该再改状态区
    stop_evt.set()
    fake.join(timeout=3)
    app.thread = None
    app.state_var.set("已停止")
    app._render_pomodoro({"phase": "break", "remaining_fmt": "05:00", "rounds_done": 9})
    check("停止后忽略状态区更新", app.state_var.get() == "已停止", app.state_var.get())

    # ---- 关闭开关时不再更新倒计时（不打扰）
    app.pomo_var.set(False)
    app.pomo_detail.configure(text="")
    app._render_pomodoro({"phase": "break", "remaining_fmt": "05:00", "rounds_done": 9})
    check("关闭开关后忽略倒计时", app.pomo_detail.cget("text") == "")

    root.destroy()

    # ---- 运行中改目标的按钮
    root2 = tk.Tk()
    root2.withdraw()
    app2 = MonitorApp(root2)
    check("有「修改当前目标」按钮", hasattr(app2, "retarget_btn"))
    check("未运行时按钮禁用", str(app2.retarget_btn.cget("state")) == "disabled")

    # 未运行时点它应给出提示而不是崩溃
    app2._retarget()
    check("未运行时改目标被安全拒绝", "未在运行" in app2.state_var.get(), app2.state_var.get())

    # 运行中（真 Monitor + 假线程）应能改目标
    import tempfile
    from pathlib import Path

    from ai_teacher.config import Settings
    from ai_teacher.db import Store
    from ai_teacher.monitor import Monitor

    tmp_db = Path(tempfile.gettempdir()) / "ai_teacher_gui_retarget.db"
    tmp_db.unlink(missing_ok=True)
    store = Store(tmp_db)
    sid = store.start_session("gui改目标")
    mon = Monitor(settings=Settings(), store=store, session_id=sid, goal="旧目标")

    app2.monitor = mon
    alive = threading.Event()
    app2.thread = threading.Thread(target=lambda: alive.wait(3), daemon=True)
    app2.thread.start()

    app2.goal_var.set("新目标")
    app2._goal_is_placeholder = False
    app2.retarget_btn.configure(state="normal")
    app2._retarget()

    check("运行中改目标已生效", mon.goal == "新目标", mon.goal)
    check("改目标有日志提示", "目标已改为" in app2.log.get("1.0", "end"))
    check("改目标后状态区有反馈", "目标已改为" in app2.state_var.get(), app2.state_var.get())

    # 改完后运行仍在继续（线程没被中断）
    check("改目标不影响运行", app2.thread.is_alive())

    alive.set()
    app2.thread.join(timeout=3)
    mon.stop()
    store.close()
    tmp_db.unlink(missing_ok=True)
    root2.destroy()

    # ---- 配置守卫机制自检
    # 注意：不能直接断言「config.toml 里没有测试目标」——本测试自己
    # 就正在守卫范围内，atexit 还没触发还原。改为独立验证守卫机制。
    import tempfile
    from pathlib import Path

    from ai_teacher import config as cfgmod

    with tempfile.TemporaryDirectory() as td:
        real_path = cfgmod.CONFIG_PATH
        fake = Path(td) / "config.toml"
        fake.write_text('goal = "真实目标"\n', encoding="utf-8")
        try:
            cfgmod.CONFIG_PATH = fake
            with _ConfigGuard():
                # 模拟测试往配置里写脏东西
                fake.write_text('goal = "冒烟测试目标"\n', encoding="utf-8")
                check("守卫期间文件被改动", "冒烟测试目标" in fake.read_text(encoding="utf-8"))
            restored = fake.read_text(encoding="utf-8")
            check("退出守卫后配置已还原", "真实目标" in restored, restored.strip()[:40])

            #原本不存在配置文件时，守卫应删掉测试生成的
            fake2 = Path(td) / "new.toml"
            cfgmod.CONFIG_PATH = fake2
            with _ConfigGuard():
                fake2.write_text('goal = "测试"\n', encoding="utf-8")
            check("原本无配置时守卫会删除文件", not fake2.exists())
        finally:
            cfgmod.CONFIG_PATH = real_path


def test_font_and_scroll() -> None:
    """防回归：字体不能太小，报告窗口必须能滚动。

    这两个问题都来自用户的实际反馈：
    - 字体偏小（且 DPI 感知调用位置错了导致发虚）
    - 报告窗口固定高度、内容超出部分无法查看
    """

    print("\n[bold cyan]字体与滚动[/]")
    import ai_teacher.gui as g

    # ---- 字体：正文/按钮不得低于 12pt，状态大字不得低于 28pt
    check("正文字号 >= 12", g.F_BODY[1] >= 12, str(g.F_BODY))
    check("按钮字号 >= 12", g.F_BTN[1] >= 12, str(g.F_BTN))
    check("状态字号 >= 28", g.F_STATE[1] >= 28, str(g.F_STATE))
    check("标题字号 >= 20", g.F_TITLE[1] >= 20, str(g.F_TITLE))

    # ---- 集中管理：控件不应再硬编码字体
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    hardcoded = src.count('font=("Microsoft YaHei UI"') + src.count('font=("Consolas"')
    check("字体已统一为命名字体", hardcoded == 0, f"仍有 {hardcoded} 处硬编码")

    # ---- DPI 感知必须在 Tk() 之前调用
    tree = ast.parse(src)
    main_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), None
    )
    check("存在 main()", main_fn is not None)
    if main_fn:
        calls = [
            n.func.id
            for n in ast.walk(main_fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        # 第一条语句应是 _enable_dpi_awareness()，且早于 tk.Tk()
        body_calls = [
            n.func.id
            for stmt in main_fn.body[:3]
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        check(
            "DPI 感知在 Tk() 之前调用",
            "_enable_dpi_awareness" in calls
            and body_calls.index("_enable_dpi_awareness") < len(body_calls),
            str(body_calls[:3]),
        )
    check("有 _enable_dpi_awareness 函数", hasattr(g, "_enable_dpi_awareness"))

    # ---- 报告窗口：必须有 Canvas + Scrollbar，且窗口可缩放
    root = tk.Tk()
    root.geometry("900x800")
    app = MonitorApp(root)
    from ai_teacher.stats import SessionStats

    st = SessionStats()
    st.screenshots = 100
    st.camera_shots = 100
    st.local_hits = 70
    st.api_calls = 30
    st.focus_ticks = 65
    st.distracted_ticks = 25
    st.unsure_ticks = 10
    st.alerts = 3
    st.prompt_tokens = 7500
    st.completion_tokens = 1200

    app._show_report(st.as_dict())
    for _ in range(8):
        root.update()

    wins = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    check("报告窗口已创建", len(wins) == 1)
    if wins:
        win = wins[0]
        for _ in range(8):
            root.update()

        classes: list[str] = []

        def walk(w: tk.Misc) -> None:
            for c in w.winfo_children():
                classes.append(c.winfo_class())
                walk(c)

        walk(win)
        check("含 Canvas 滚动容器", "Canvas" in classes)
        check(
            "含滚动条",
            any(k in ("TScrollbar", "Scrollbar") for k in classes),
            str(sorted(set(classes))),
        )
        check("窗口可缩放（原先 False 导致裁切）", win.wm_resizable()[0] == 1)

        # 内容超出时 scrollregion 要覆盖全部内容
        canvases = []

        def find_c(w: tk.Misc) -> None:
            for c in w.winfo_children():
                if c.winfo_class() == "Canvas":
                    canvases.append(c)
                find_c(c)

        find_c(win)
        if canvases:
            cv = canvases[0]
            bbox = cv.bbox("all")
            content_h = bbox[3] if bbox else 0
            check(
                "scrollregion 覆盖完整内容",
                content_h > 0 and content_h >= cv.winfo_height(),
                f"内容 {content_h}px / 可视 {cv.winfo_height()}px",
            )
        win.destroy()

    root.destroy()


def test_gui_isolation() -> None:
    """GUI 测试会真起 Monitor 线程，必须确认它碰不到真实 key。"""

    print("\n[bold cyan]GUI 测试隔离[/]")
    from ai_teacher.config import get_settings

    s = get_settings()
    check("GUI 测试读不到真实 key", not s.api.api_key, repr(s.api.api_key[:8]))
    check("base_url 不可路由", "unreachable" in s.api.base_url, s.api.base_url)

    # MonitorApp 里拿到的设置也必须是隔离后的
    root = tk.Tk()
    root.withdraw()
    app = MonitorApp(root)
    check("MonitorApp 的设置同样被隔离", not app.settings.api.api_key)
    check("界面提示无 key", "未配置" in app.api_lbl.cget("text"), app.api_lbl.cget("text")[:30])
    root.destroy()


def test_module_entrypoint() -> None:
    """`python -m ai_teacher.gui` 必须真的开窗。

    这个用例防回归：曾经gui.py 定义了 main() 却从没调用，也没有
    __main__ 块，导致 `python -m ai_teacher.gui` 静默退出（exit 0、
    无任何输出），而测试直接调MonitorApp 绕过了入口，一直没发现。
    """

    print("\n[bold cyan]模块入口[/]")
    import ai_teacher.gui as g

    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 1) 源码层面：必须有 __main__ 块且调用 main()
    has_guard = any(
        isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and any(
            isinstance(c, ast.Constant) and c.value == "__main__" for c in n.test.comparators
        )
        for n in tree.body
    )
    check("gui.py 有 __main__ 块", has_guard)

    guard_calls = False
    for n in tree.body:
        if (
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in n.test.comparators)
        ):
            guard_calls = any(
                isinstance(x, ast.Call)
                and isinstance(x.func, ast.Name)
                and x.func.id == "main"
                for x in ast.walk(n)
            )
    check("__main__ 块里调用了 main()", guard_calls, "否则静默退出、exit 0、无输出")

    # 2) 运行层面：真正跑一遍模块，进程必须存活到超时（证明进了 mainloop）
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ai_teacher.gui"],
            capture_output=True,
            timeout=6,
            cwd=str(pathlib.Path(g.__file__).parent.parent),
        )
        # 没超时说明 main() 没跑，模块导入后就退出了
        check(
            "以子进程启动会挂住（已进入 mainloop）",
            False,
            f"已退出 rc={proc.returncode}，stdout={proc.stdout[:80]}",
        )
    except subprocess.TimeoutExpired:
        check("以子进程启动会挂住（已进入 mainloop）", True)


def test_report_window() -> None:
    """结束时的报告窗口。"""

    print("\n[bold cyan]会话报告窗口[/]")
    from ai_teacher.stats import SessionStats

    root = tk.Tk()
    root.withdraw()
    app = MonitorApp(root)

    st = SessionStats()
    st.screenshots = 42
    st.camera_shots = 40
    st.camera_miss = 2
    st.local_hits = 30
    st.api_calls = 12
    st.focus_ticks = 28
    st.distracted_ticks = 9
    st.unsure_ticks = 5
    st.alerts = 2
    st.prompt_tokens = 3000
    st.completion_tokens = 480
    st.image_bytes = 12 * 30 * 1024

    try:
        app._show_report(st.as_dict())
        pump(app, 0.5)
        wins = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
        check("报告窗口已创建", len(wins) == 1, f"{len(wins)} 个")

        if wins:
            win = wins[0]
            check("窗口标题正确", "学习报告" in win.title(), win.title())

            # 递归收集所有 Label 文本（报告用了多层 Frame 嵌套）
            texts: list[str] = []

            def walk(w: tk.Misc) -> None:
                for c in w.winfo_children():
                    if c.winfo_class() == "Label":
                        texts.append(str(c.cget("text")))
                    walk(c)

            walk(win)
            joined = "".join(texts).replace(" ", "")
            check("显示截图次数", "42次" in joined, "")
            check("显示拍照次数", "40次" in joined, "")
            check("显示本地判定", "30次" in joined, "")
            check("显示模型判定", "12次" in joined, "")
            check("显示花费", "¥" in joined, "")
            check("显示告警次数", "告警次数" in joined, "")
            win.destroy()
    except Exception as exc:
        check("报告窗口正常打开", False, f"{type(exc).__name__}: {exc}")

    # 缺字段时应降级而不是崩
    try:
        app._show_report({"screenshots": 5})
        pump(app, 0.3)
        check("缺字段时不崩溃", True, "降级为一行提示")
    except Exception as exc:
        check("缺字段时不崩溃", False, f"{type(exc).__name__}: {exc}")

    # 实时统计行：30 本地 / 12 模型 = 71.4%
    app._render_stats(st.as_dict())
    check("统计行含截图", "截图 42" in app.stat_var.get(), app.stat_var.get())
    check("统计行含拦截率", "71%" in app.cost_var.get(), app.cost_var.get())
    check("统计行含花费", "¥" in app.cost_var.get(), app.cost_var.get())

    root.destroy()


def test_placeholder() -> None:
    """占位符不能被当成真目标。"""

    print("\n[bold cyan]占位提示逻辑[/]")
    root = tk.Tk()
    root.withdraw()
    app = MonitorApp(root)

    # 模拟配置里没有目标的情况
    app.settings.app.goal = ""
    app.goal_var.set("")
    app._goal_is_placeholder = True
    app._sync_goal_placeholder()
    pump(app, 0.2)

    check("空目标时显示占位提示", "例如" in app.goal_var.get(), app.goal_var.get()[:20])
    check("占位状态取目标为空", app._current_goal() == "", "不能把提示语当目标")

    # 聚焦应清空占位符
    app._on_goal_focus_in(None)
    check("聚焦后清空占位符", app.goal_var.get() == "")
    check("聚焦后不再是占位状态", not app._goal_is_placeholder)

    # 输入真实目标
    app.goal_var.set("写论文")
    check("输入后取到真实目标", app._current_goal() == "写论文")

    # 失焦时清空则恢复占位符
    app.goal_var.set("")
    app._on_goal_focus_out(None)
    check("清空后失焦恢复占位符", app._goal_is_placeholder)
    check("恢复后仍取不到目标", app._current_goal() == "")

    app.root.destroy()


if __name__ == "__main__":
    sys.exit(main())
