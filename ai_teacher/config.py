"""全局配置：API 凭据、阈值、采样节奏。

优先级：环境变量 / .env > config.toml > 默认值。
API key只从环境变量读，不写入 config.toml。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"
DB_PATH = ROOT / "data" / "sessions.db"


class ApiSettings(BaseSettings):
    """API 凭据。全部走环境变量，键名前缀 AI_TEACHER_。"""

    model_config = SettingsConfigDict(
        env_prefix="AI_TEACHER_",
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    # OpenAI 兼容端点。任何实现 /v1/chat/completions 的服务都能用：
    #   DeepSeek: https://api.deepseek.com
    #   Moonshot: https://api.moonshot.cn/v1
    #   通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1
    #   本地 Ollama: http://localhost:11434/v1
    # 想换服务只改 base_url + model，provider 保持 deepseek（默认参数最全）。
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"

    # 思考模式开关（仅 DeepSeek 系支持，其他服务必须关，否则请求会被拒）。
    #   "auto"    —— base_url 是 DeepSeek 时发 thinking 参数，否则不发
    #   "disable" —— 总是发送（DeepSeek 用）
    #   "omit"    —— 从不发送（其他 OpenAI 兼容服务用）
    thinking: str = "auto"

    # JSON Output。OpenAI 标准字段，绝大多数兼容服务都支持。
    # 万一某个服务不支持，可设为 False 关闭（代价是要靠正则兜底解析）。
    json_output: bool = True

    # low = 服务端先缩到 512x512 再推理，约 159 tokens/张
    image_detail: str = "low"

    # 输出上限。**别设太小**——DeepSeek 开思考时思考链也计入这个额度，
    # 设48 会导致思考吃光额度、正文为空（bad_json）。64 关闭思考时够用。
    max_output_tokens: int = 64

    timeout_s: float = 30.0
    max_retries: int = 2

    @property
    def is_deepseek(self) -> bool:
        """是否为 DeepSeek 服务。

        **同时看 base_url 和 model**——只看域名会漏判：
        代理/自建网关转发到 DeepSeek 时域名不是 deepseek.com，
        但模型名是 deepseek-flash / deepseek-chat，仍需关思考。
        """

        return "deepseek" in self.base_url.lower() or "deepseek" in self.model.lower()

    @property
    def send_thinking(self) -> bool:
        """是否需要发送 thinking 参数。"""

        if self.thinking == "disable":
            return True
        if self.thinking == "omit":
            return False
        return self.is_deepseek


class CaptureSettings(BaseModel):
    """图像采集与压缩。"""

    # 合成图长边。detail=low 时服务端会再缩到 512，短边给足以免叠加后过小
    compose_width: int = Field(default=640, ge=320, le=1280)
    jpeg_quality: int = Field(default=60, ge=30, le=95)
    # 摄像头画面占总宽的比例，摄像头通常不需要很清晰
    camera_ratio: float = Field(default=0.28, ge=0.15, le=0.5)
    camera_index: int = Field(default=0, ge=0)
    # 帧差检测用的降采样尺寸，越小越快
    diff_width: int = Field(default=160, ge=32, le=320)
    # 两次采样的帧差绝对值之和超过此值，认为画面有变化
    diff_threshold: float = Field(default=6.0, ge=0.0)


class RuleSettings(BaseModel):
    """L0 本地规则阈值。"""

    # 键鼠空闲超过此秒数，视为无交互（不直接判分心，只降权）
    idle_soft_s: float = Field(default=180.0, ge=10.0)
    idle_hard_s: float = Field(default=600.0, ge=60.0)
    # 人脸检测的最小命中比例（保留字段供调试参考，实际判定走 present()）
    face_min_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    # 连续多少帧检测不到人才判定为离开。
    # 单帧不可靠：Haar 在俯拍低头场景本就临界，抖动一次就误报离开很烦人。
    absence_confirm: int = Field(default=3, ge=1, le=10)


class JudgeSettings(BaseModel):
    """L1 模型判定与 L2 状态聚合。"""

    # 模型判定分心后，需连续命中几次才告警
    strikes_to_warn: int = Field(default=2, ge=1, le=10)
    # 启动后的宽限期，此期间不告警
    grace_period_s: float = Field(default=180.0, ge=0.0)
    # 两次告警之间的最小间隔，避免刷屏
    alert_cooldown_s: float = Field(default=300.0, ge=0.0)
    # 自适应采样间隔边界。
    # 实测（关思考模式，单次 pt=276/out=14≈ ¥0.000304）：
    #   持续在学→ 间隔拉到 300s，每天 8h 约 144 次 = ¥0.044
    #   偶尔走神        → 约 224 次 = ¥0.068
    #   一直分心        → 约 480 次 = ¥0.146（分心时需要密集采样来确认）
    # 注意「越不学习越费钱」是合理的：重点是全天 ¥0.15 以内。
    interval_min_s: float = Field(default=60.0, ge=15.0)
    interval_max_s: float = Field(default=300.0, ge=60.0)
    # 连续判定为学习中多少次后拉长间隔。
    # 2 比 3 更省（早一次拉长，多省约 20%），实测无质量损失。
    interval_grow_after: int = Field(default=2, ge=1, le=10)
    # 判定学习中的滑动窗口比例阈值（EWMA）
    focus_ewma_alpha: float = Field(default=0.4, ge=0.05, le=1.0)
    focus_healthy_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    focus_bad_ratio: float = Field(default=0.4, ge=0.0, le=1.0)


class PomodoroSettings(BaseModel):
    """课堂模式：学一段、休息一段，循环往复。

    番茄钟的「传统」是 25+5，这里默认 40+10——按一节课的注意力节奏设的，
    40 分钟专注学习后必须离开屏幕活动一下。
    """

    # 是否启用课堂模式（启动时可选，不启用则完全不打扰）
    enabled: bool = False
    # 专注时长（分钟）
    focus_minutes: float = Field(default=40.0, ge=1.0, le=180.0)
    # 休息时长（分钟）。0 表示跳过后直接进入下一轮专注。
    break_minutes: float = Field(default=10.0, ge=0.0, le=60.0)
    # 休息开始与结束时是否弹右下角通知
    notify: bool = True


class AppSettings(BaseModel):
    goal: str = "学习"
    monitor_camera: bool = True
    upload_camera: bool = True
    headless_notify: bool = False
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    rules: RuleSettings = Field(default_factory=RuleSettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    pomodoro: PomodoroSettings = Field(default_factory=PomodoroSettings)


class Settings:
    """聚合配置。"""

    def __init__(self) -> None:
        self.api = ApiSettings()
        file_cfg: dict[str, Any] = {}
        if CONFIG_PATH.exists():
            file_cfg = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.app = AppSettings.model_validate(file_cfg)

    def save(self) -> None:
        """把当前 app 配置写回 config.toml。

        api 段是凭据，只从环境变量读，不落盘。
        """

        import tomli_w

        dump = self.app.model_dump(mode="json")
        header = (
            "# AI_Teacher 配置文件\n"
            "# API key 请写在项目根目录的 .env 里（AI_TEACHER_API_KEY=...），不要写在这里。\n"
            "# 直接改这个文件后，运行中的程序重启即生效。\n\n"
        )
        CONFIG_PATH.write_text(header + tomli_w.dumps(dump), encoding="utf-8")

    def reload(self) -> None:
        self.app = AppSettings.model_validate(
            tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if CONFIG_PATH.exists()
            else {}
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
