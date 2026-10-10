"""命令行入口。

命令：
  ai-teacher goal "刷 LeetCode"   设定学习目标（持久化到 config.toml）
  ai-teacher start              启动监控
  ai-teacher status             查看当前状态与最近一次会话
  ai-teacher cost               打印一次判定的成本估算
"""

from __future__ import annotations

import json
import time

import typer
from rich.console import Console
from rich.table import Table

from .config import ROOT, get_settings
from .db import Store
from .stats import est_cost, format_report_from_dict

app = typer.Typer(add_completion=False, help="电子教官 —— 学习状态监控与提醒")
console = Console()


@app.command()
def goal(
    text: str = typer.Argument(..., help="学习目标，如「刷 LeetCode」「看机器学习网课」"),
) -> None:
    """设定学习目标。"""

    settings = get_settings()
    settings.app.goal = text
    settings.save()
    console.print(f"[green]目标已设定：[/]{text}")


@app.command()
def pomodoro(
    action: str = typer.Argument(
        None,
        help="on 开启 / off 关闭 / 不填则显示当前设置",
    ),
    focus: int = typer.Option(None, "--focus", help="专注时长（分钟），默认 40"),
    rest: int = typer.Option(None, "--rest", help="休息时长（分钟），默认 10"),
) -> None:
    """课堂模式：每专注一段时间提醒休息。"""

    settings = get_settings()
    pom = settings.app.pomodoro

    if action in (None, "", "show"):
        if pom.enabled:
            console.print(
                f"[green]课堂模式已开启[/] 每 {pom.focus_minutes:.0f} 分钟"
                f"休息 {pom.break_minutes:.0f} 分钟（期间不抓拍）"
            )
        else:
            console.print("[dim]课堂模式未启用[/]　用 [cyan]pomodoro on[/] 开启")
        return

    if action in ("on", "开启", "启用"):
        pom.enabled = True
        if focus:
            pom.focus_minutes = float(focus)
        if rest is not None:
            pom.break_minutes = float(rest)
        console.print(
            f"[green]课堂模式已开启[/] 每 {pom.focus_minutes:.0f} 分钟"
            f"休息 {pom.break_minutes:.0f} 分钟"
        )
    elif action in ("off", "关闭", "禁用"):
        pom.enabled = False
        console.print("[dim]课堂模式已关闭[/]")
    else:
        console.print(f"[red]无法识别的操作：{action}[/]　可用：on / off")
        raise typer.Exit(code=1)

    settings.save()


@app.command()
def start(
    goal_override: str = typer.Option(None, "--goal", "-g", help="临时目标，不写入配置"),
    once: bool = typer.Option(False, "--once", help="只跑一个 tick，用于自检"),
) -> None:
    """启动监控。Ctrl+C 停止。"""

    from .monitor import Monitor

    settings = get_settings()
    goal_text = goal_override or settings.app.goal

    if not settings.api.api_key:
        console.print(
            "[yellow]提示：未检测到 API key。[/] 在项目根目录建 .env 写入：\n"
            "  [bold]AI_TEACHER_API_KEY=sk-你的key[/]"
        )

    with Store() as store:
        session_id = store.start_session(goal_text)
        with Monitor(settings=settings, store=store, session_id=session_id, goal=goal_text) as m:
            if once:
                d = m.tick()
                m._print_tick(d)
                m.shutdown()
            else:
                m.run()


@app.command()
def gui() -> None:
    """启动图形界面。"""

    from .gui import main as gui_main

    gui_main()


@app.command()
def provider(
    action: str = typer.Argument(None, help="show 查看 / set 设置 / reset 恢复默认"),
    base_url: str = typer.Option(None, "--url", help="OpenAI 兼容端点，如 https://api.moonshot.cn/v1"),
    model: str = typer.Option(None, "--model", help="模型名，如 kimi-k2-5"),
    thinking: str = typer.Option(
        None, "--thinking", help="auto / disable / omit——是否发送 DeepSeek 的思考模式参数"
    ),
) -> None:
    """切换模型服务。任何 OpenAI 兼容端点都可用。

    base_url / model 写入项目根目录的 .env（API key 也在那里，不进 config.toml）。
    """

    env_path = ROOT / ".env"
    settings = get_settings()
    api = settings.api

    if action in (None, "", "show"):
        t = Table(title="模型服务", show_header=False, box=None)
        t.add_row("base_url", api.base_url)
        t.add_row("model", api.model)
        t.add_row("是否 DeepSeek 官方", "是" if api.is_deepseek else "否")
        t.add_row("发送 thinking 参数", "是" if api.send_thinking else "否")
        t.add_row("JSON Output", "开" if api.json_output else "关")
        t.add_row("输出上限", f"{api.max_output_tokens} tokens")
        t.add_row("配置文件", str(env_path))
        console.print(t)
        console.print(
            "[dim]提示：API key 请在 .env 里设AI_TEACHER_API_KEY=xxx[/]"
        )
        return

    def _upsert_env(key: str, value: str) -> None:
        """更新 .env 里的某个键，保留其他行。"""

        lines = (
            env_path.read_text(encoding="utf-8").splitlines()
            if env_path.exists()
            else []
        )
        prefix = f"{key}="
        replaced = False
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if action == "reset":
        _upsert_env("AI_TEACHER_BASE_URL", "https://api.deepseek.com")
        _upsert_env("AI_TEACHER_MODEL", "deepseek-flash")
        console.print("[green]已恢复默认：DeepSeek deepseek-flash[/]")
        return

    if action != "set":
        console.print(f"[red]无法识别：{action}[/]　可用：show / set / reset")
        raise typer.Exit(code=1)

    if not base_url and not model:
        console.print("[red]请至少提供 --url 或 --model[/]")
        raise typer.Exit(code=1)

    if base_url:
        _upsert_env("AI_TEACHER_BASE_URL", base_url)
    if model:
        _upsert_env("AI_TEACHER_MODEL", model)
    if thinking:
        _upsert_env("AI_TEACHER_THINKING", thinking)

    # 立刻用新值重读，确认生效
    import ai_teacher.config as cfgmod

    cfgmod._settings = None  # type: ignore[attr-defined]
    fresh = cfgmod.get_settings()
    console.print(
        f"[green]已切换[/] {fresh.api.model} @ {fresh.api.base_url}"
        f"（thinking 参数：{'发送' if fresh.api.send_thinking else '不发送'}）"
    )
    console.print("[dim]若换了服务商，记得在 .env 里把 AI_TEACHER_API_KEY 换成对应的 key[/]")


@app.command()
def status() -> None:
    """查看配置与最近一次会话。"""

    settings = get_settings()
    app_cfg = settings.app

    table = Table(title="当前配置", show_header=False, box=None)
    table.add_row("学习目标", app_cfg.goal)
    table.add_row("模型", f"{settings.api.model} @ {settings.api.base_url}")
    table.add_row("图片 detail", settings.api.image_detail)
    table.add_row("采样间隔", f"{app_cfg.judge.interval_min_s:.0f}~{app_cfg.judge.interval_max_s:.0f}s 自适应")
    table.add_row("告警阈值", f"连续 {app_cfg.judge.strikes_to_warn} 次分心")
    table.add_row("摄像头", "开启" if app_cfg.monitor_camera else "关闭")
    table.add_row("上传摄像头", "是" if app_cfg.upload_camera else "否")
    pom = app_cfg.pomodoro
    if pom.enabled:
        table.add_row(
            "课堂模式",
            f"每 {pom.focus_minutes:.0f} 分钟休息 {pom.break_minutes:.0f} 分钟",
        )
    else:
        table.add_row("课堂模式", "[dim]未启用[/]")
    table.add_row("API key", "已配置" if settings.api.api_key else "[red]未配置[/]")
    api = settings.api
    table.add_row("模型", f"{api.model}　[dim]{api.base_url}[/]")
    table.add_row(
        "接口格式",
        "OpenAI 兼容"
        + (
            "（DeepSeek 官方，已自动关思考模式）"
            if api.is_deepseek
            else "（通用，未发送 DeepSeek 专属参数）"
        ),
    )
    console.print(table)

    with Store() as store:
        last = store.last_session()
        if last is None:
            console.print("\n[dim]还没有任何会话记录。[/]")
            return
        sid = int(last["id"])
        s = store.session_summary(sid)
        console.print(
            f"\n[bold]最近会话[/] #{sid} · 目标「{last['goal']}」\n"
            f"采样 {s.get('n') or 0} 次，模型调用 {s.get('api_n') or 0} 次，"
            f"专注比例 {(s.get('focus_ratio') or 0) * 100:.0f}%"
        )
        if s.get("pt"):
            console.print(
                f"token 输入 {s['pt']:,} / 输出 {s.get('ct') or 0:,}，"
                f"预估花费 ¥{est_cost(int(s['pt']), int(s.get('ct') or 0)):.5f}"
            )
        console.print("[dim]用 report 命令看完整报告[/]")


@app.command()
def report(
    session_id: int = typer.Argument(None, help="会话编号，不填则显示最近一次"),
) -> None:
    """查看某次会话的完整统计报告。"""

    with Store() as store:
        if session_id is None:
            last = store.last_session()
            if last is None:
                console.print("[dim]还没有任何会话记录。[/]")
                return
            session_id = int(last["id"])

        goal = store.session_goal(session_id)
        if goal is None:
            console.print(f"[red]找不到会话 #{session_id}[/]")
            raise typer.Exit(code=1)

        st = store.session_stats(session_id)
        if st is None:
            # 老版本会话没存统计，退回 tick 表聚合
            s = store.session_summary(session_id)
            console.print(
                f"[yellow]会话 #{session_id} 建于旧版本，没有完整统计。[/]\n"
                f"以下由 tick 记录聚合：\n"
                f"  采样 {s.get('n') or 0} 次 · 模型调用 {s.get('api_n') or 0} 次 · "
                f"专注 {(s.get('focus_ratio') or 0) * 100:.0f}% · "
                f"token {s.get('pt') or 0:,} 输入"
            )
            return

        try:
            rows = format_report_from_dict(st)
        except (KeyError, TypeError) as exc:
            console.print(f"[red]统计数据格式异常：{exc}[/]")
            raise typer.Exit(code=1) from exc

        table = Table(
            title=f"学习报告 · #{session_id} · {goal}",
            show_header=False,
            box=None,
            title_style="bold",
        )
        for label, value in rows:
            if not value:
                table.add_row(f"[bold cyan]{label}[/]", "")
            else:
                table.add_row(label, value)
        console.print(table)


@app.command()
def history() -> None:
    """列出最近的会话。"""

    with Store() as store:
        rows = store.list_sessions(15)
        if not rows:
            console.print("[dim]还没有任何会话记录。[/]")
            return
        table = Table(title="最近会话", box=None)
        table.add_column("#", justify="right", style="dim")
        table.add_column("目标")
        table.add_column("开始时间")
        table.add_column("时长", justify="right")
        table.add_column("截图", justify="right")
        table.add_column("本地/模型", justify="right")
        table.add_column("花费", justify="right")

        for r in rows:
            st = None
            if r["stats"]:
                try:
                    st = json.loads(r["stats"])
                except (json.JSONDecodeError, TypeError):
                    st = None
            started = time.strftime("%m-%d %H:%M", time.localtime(r["started_at"]))
            if r["ended_at"]:
                dur = int(r["ended_at"] - r["started_at"])
                dur_s = f"{dur // 3600}h{dur % 3600 // 60:02d}m" if dur >= 3600 else f"{dur // 60}m"
            else:
                dur_s = "[yellow]进行中[/]"
            if st:
                table.add_row(
                    str(r["id"]),
                    r["goal"],
                    started,
                    dur_s,
                    str(st.get("screenshots", 0)),
                    f"{st.get('local_hits', 0)}/{st.get('api_calls', 0)}",
                    f"¥{st.get('est_cost', 0):.5f}",
                )
            else:
                table.add_row(str(r["id"]), r["goal"], started, dur_s, "-", "-", "-")
        console.print(table)
        console.print("[dim]用 report <编号> 看某次的完整报告[/]")


@app.command()
def cost() -> None:
    """打印一次判定的成本估算。"""

    from .stats import current_prices, est_cost, is_peak_time

    settings = get_settings()
    peak = is_peak_time()
    in_price, out_price = current_prices()

    # 实测值（关闭思考 + json_object，连续 12 次采样零波动）：
    #   输入 276 = 图片 ~215（detail=low）+ 提示词 ~61
    #   输出 14  = 精简 JSON
    img_tokens = 215
    prompt_tokens = 61
    completion_tokens = 14
    cost = est_cost(276, 14, peak=peak)

    table = Table(title="单次判定成本", show_header=False, box=None)
    table.add_row("定价时段", f"{'高峰' if peak else '空闲'}（输入 ¥{in_price:g}/M，输出 ¥{out_price:g}/M）")
    table.add_row("图片 tokens", f"~{img_tokens}（detail=low，512x512）")
    table.add_row("提示词 tokens", f"~{prompt_tokens}")
    table.add_row("输出 tokens", f"~{completion_tokens}")
    table.add_row("合计", "~276 输入 + 14 输出")
    table.add_row("单次成本", f"¥{cost:.6f}")
    table.add_row(
        "思考模式对比",
        "关闭 ~14 输出 tokens；开启约 613（本项目关闭思考省 40 倍，且贵 6 倍）",
    )
    table.add_row(
        "时段提示",
        "高峰 = 周一至周五 9-12、14-18（北京时间），单价为空闲时段的 2 倍",
    )
    console.print(table)

    # ---- 日成本：取决于学习表现（间隔自适应）
    from .rules import Verdict
    from .tracker import FocusTracker

    j = settings.app.judge

    def calls_per_hour(studying: bool | None) -> float:
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

    print()
    t2 = Table(title="日成本预估（按有效学习 8 小时）", show_header=False, box=None)
    t2.add_row("采样间隔", f"{j.interval_min_s:.0f}~{j.interval_max_s:.0f}s 自适应")
    t2.add_row("当前时段", f"{'高峰' if peak else '空闲'}")
    for label, studying, note in [
        ("持续在学", True, "间隔拉到最长"),
        ("一直分心", False, "间隔最短，密集采样确认"),
    ]:
        c = calls_per_hour(studying)
        per_day = c * 8
        day_cost = est_cost(276 * per_day, 14 * per_day, peak=peak)
        t2.add_row(
            label,
            f"{per_day:.0f} 次/天 = ¥{day_cost:.3f}/天（月约 ¥{day_cost * 30:.1f}）　{note}",
        )
    console.print(t2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
