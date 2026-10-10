"""本地日志：把每次判定、告警、阶段切换落成文件。

为什么要有：
- 控制台输出会滚走，事后无法复盘「昨天到底判定成什么、为什么」
- 排查问题时最需要的是「那一tick 的画面和理由」，日志是最直接的线索

安全约束：
- **绝不记录 API key 或完整请求体**。只记必要的元信息。
- 不落盘图片（隐私），只记字节数
- 单文件有大小上限，超了自动轮转，防止撑爆磁盘
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Any

from .config import DB_PATH

# 日志与数据库同目录（data/），已在 .gitignore
LOG_DIR = DB_PATH.parent / "logs"
# 单文件上限 2MB，保留 3 个历史文件（总共最多约 8MB）
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
DATE_FMT = "%H:%M:%S"

_configured = False


def get_logger(name: str = "ai_teacher") -> logging.Logger:
    """取全局 logger。重复调用不会重复加handler。"""

    global _configured
    logger = logging.getLogger(name)

    if _configured:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "ai_teacher.log"

    # 级别从环境变量读，默认 INFO。
    # DEBUG 会把每个 tick 的画面差异都打出来，噪音太大。
    level_name = os.environ.get("AI_TEACHER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    # 写文件（滚动）
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT))
    logger.addHandler(fh)

    # 控制台只记 WARNING 以上，避免和控制台已有的 rich 输出重复
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT))
    logger.addHandler(sh)

    # 不向root logger 冒泡，否则会污染其他库的输出
    logger.propagate = False
    _configured = True

    if not logger.handlers[0].baseFilename:  # pragma: no cover - 防御
        logger.warning("日志文件创建失败，仅输出到控制台")
    else:
        logger.debug("日志已开启：%s（级别 %s）", log_path, level_name)

    return logger


def log_tick(
    logger: logging.Logger,
    *,
    index: int,
    state: str,
    reason: str,
    used_api: bool,
    interval_s: float,
    person_present: bool | None,
    screenshot_bytes: int,
    elapsed_s: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    error: str = "",
) -> None:
    """记录一个 tick 的判定结果。

    只记必要字段——**不记图像内容、不记请求体**。
    """

    person = {True: "在场", False: "不在", None: "未知"}[person_present]
    src = "模型" if used_api else "本地"
    extra = ""
    if used_api and (prompt_tokens or completion_tokens):
        extra = f" tok={prompt_tokens}+{completion_tokens}"
    if error:
        extra += f" err={error}"
    logger.info(
        "[%3d] %-9s %-4s %-52s 间隔=%5.1fs 人=%-4s 图=%6dB 已用=%5.0fs%s",
        index,
        state,
        src,
        reason[:52],
        interval_s,
        person,
        screenshot_bytes,
        elapsed_s,
        extra,
    )


def log_event(logger: logging.Logger, event: str, detail: str = "") -> None:
    """记录会话级事件（开始、结束、告警、阶段切换）。"""

    if detail:
        logger.info("[事件] %s | %s", event, detail)
    else:
        logger.info("[事件] %s", event)


def log_alert(logger: logging.Logger, reason: str, elapsed_s: float) -> None:
    """告警单独记 WARNING 级别，方便事后 grep。"""

    logger.warning("[告警] 已用%.0fs | %s", elapsed_s, reason)


def dump_session(logger: logging.Logger, summary: dict[str, Any]) -> None:
    """会话结束时把完整统计打到日志，便于日后查账。"""

    lines = [
        f"截图 {summary.get('screenshots', 0)} | 拍照 {summary.get('camera_shots', 0)}",
        f"本地 {summary.get('local_hits', 0)} / 模型成功 {summary.get('api_calls', 0)}"
        f" / 模型失败 {summary.get('api_failed', 0)}"
        f" / 缺key跳过 {summary.get('api_skipped', 0)}",
        f"学习中 {summary.get('focus_ticks', 0)} / 分心 {summary.get('distracted_ticks', 0)}"
        f" / 不确定 {summary.get('unsure_ticks', 0)}",
        f"专注比例 {(summary.get('focus_ratio') or 0) * 100:.0f}% | 告警 {summary.get('alerts', 0)} 次",
        f"token 输入 {summary.get('prompt_tokens', 0)} / 输出 {summary.get('completion_tokens', 0)}",
        f"预估花费 ¥{summary.get('est_cost', 0):.6f}"
        f"（时段 {'高峰' if summary.get('peak') else '空闲'}）",
    ]
    if summary.get("pomodoro_used"):
        lines.append(
            f"课堂模式 {summary.get('pomodoro_rounds', 0)} 轮"
            f" | 累计休息 {summary.get('pause_seconds', 0) / 60:.1f} 分钟"
        )
    logger.info("[会话结束]\n  " + "\n  ".join(lines))