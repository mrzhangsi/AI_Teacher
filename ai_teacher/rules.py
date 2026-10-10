"""L0 本地规则层：零API 成本的确定性判定。

这一层的存在意义：锁屏、锁屏保护、全屏游戏这类事件是确定性的，
用规则判比用视觉模型判更准、更快、更省钱。
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .config import RuleSettings
from .capture import to_thumb


class Verdict(str, Enum):
    """L0 结论。"""

    FOCUS = "focus"  # 明确在学习
    SUSPECT = "suspect"  # 存疑，交给模型
    DISTRACT = "distract"  # 明确不在学习，直接判分心，不调API
    UNKNOWN = "unknown"  # 拿不到本地信号


@dataclass(slots=True)
class LocalSignals:
    """本地采集到的各项信号。"""

    verdict: Verdict = Verdict.UNKNOWN
    reason: str = ""
    face_ratio: float | None = None
    skin_ratio: float | None = None
    person_present: bool | None = None
    person_reason: str = ""
    absent_streak: int = 0
    idle_s: float | None = None
    screen_changed: bool | None = None
    screen_locked: bool = False
    details: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- Win32 空闲

class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]


def get_idle_seconds() -> float | None:
    """用 GetLastInputInfo 取键鼠空闲秒数。拿不到就返回 None。"""

    if not hasattr(ctypes, "windll"):
        return None
    try:
        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)) == 0:
            return None
        # GetTickCount 与 dwTime 同基准，都是开机毫秒数
        elapsed_ms = ctypes.windll.kernel32.GetTickCount() - info.dwTime
        return max(0.0, elapsed_ms / 1000.0)
    except Exception:
        return None


# ------------------------------------------------------------ 屏幕锁定检测

def is_screen_locked() -> bool:
    """检测是否处于锁屏/ UAC 安全桌面。

    Win32 的做法：锁屏时输入桌面被换成 secure desktop，
    OpenInputDesktop 会返回NULL。普通前台窗口下它总能打开。

    这样判定是确定性的，比丢一张黑屏给模型判又快又准，还省一次调用。
    """

    if not hasattr(ctypes, "windll"):
        return False
    try:
        user32 = ctypes.windll.user32
        # 64 位下 HANDLE 是 8 字节，必须显式声明，否则 ctypes 默认按 c_int 截断
        user32.OpenInputDesktop.argtypes = [ctypes.c_uint, ctypes.c_bool, ctypes.c_uint]
        user32.OpenInputDesktop.restype = ctypes.c_void_p
        user32.CloseDesktop.argtypes = [ctypes.c_void_p]
        user32.CloseDesktop.restype = ctypes.c_bool
        # DESKTOP_SWITCHDESKTOP = 0x0100，只读打开即可
        handle = user32.OpenInputDesktop(0, False, 0x0100)
        if handle:
            user32.CloseDesktop(handle)
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 人脸检测

class FaceDetector:
    """在场检测：Haar 级联为主 + 肤色比例为兜底。

    **为什么必须有兜底通道。** 实际用下来发现 Haar 单独用会持续漏检，典型场景是
    摄像头架在显示器上方俯拍：人低头看屏幕时画面里只剩额头和眼镜，下半张脸被
    裁掉，而 Haar 依赖眼-鼻-嘴的完整几何结构，天生识别不了这种脸。

    实测数据（同一个俯拍场景，640x480）：
    - Haar（scaleFactor=1.2）：稳定0 个 —— 这就是「明明在镜头前却判未检测到人」
    - Haar（scaleFactor=1.05）：1~2 个，但仍会偶发 0
    - 肤色像素占比（人在下半画面）：40% ± 1.5%
    - 肤色像素占比（空背景）：5%
    两者区分度约 8 倍，且帧间稳定，因此适合做兜底。

    两个通道取「或」：任一命中就认为在场。肤色通道偏宽松，但它的作用是
    防止把「在场」误判成「离开」——宁可多判在场（继续交给模型看屏幕），
    也不能把认真低头看书的人判成离开而误报。
    """

    # 肤色阈值（YCrCb 空间经典取值，对亚洲肤色友好）
    _CR_LO, _CR_HI = 133, 173
    _CB_LO, _CB_HI = 77, 127
    # 肤色占比阈值：实测人在 ~40%、空背景 ~5%，取 0.12 留足余量
    _SKIN_RATIO_MIN = 0.12

    def __init__(self) -> None:
        self._cascade: cv2.CascadeClassifier | None = None

    def _ensure(self) -> cv2.CascadeClassifier | None:
        if self._cascade is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                return None
            self._cascade = cascade
        return self._cascade

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Haar 人脸框列表 (x, y, w, h)。坐标基于原始图像。"""

        cascade = self._ensure()
        if cascade is None or frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # scaleFactor<=1.1 与 minNeighbors=3 是召回率的关键，不要调大
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
        )
        return [tuple(int(v) for v in b) for b in boxes]  # type: ignore[misc]

    def skin_ratio(self, frame: np.ndarray) -> float:
        """肤色像素占比（下半画面），作为兜底的「有人在场」信号。

        只看下2/3 区域：俯拍时人脸和上半身总是落在画面下部，
        上部多为背景，可减少背景干扰。
        """

        if frame is None or frame.size == 0:
            return 0.0
        h = frame.shape[0]
        region = frame[int(h * 0.38) :, :]
        if region.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        mask = (
            (cr >= self._CR_LO) & (cr <= self._CR_HI) & (cb >= self._CB_LO) & (cb <= self._CB_HI)
        )
        return float(mask.mean())

    def present(self, frame: np.ndarray) -> tuple[bool, str, float]:
        """综合判断是否有人在场。返回 (是否在场, 说明, 肤色占比)。

        Haar 命中即为在场；Haar 漏检时用肤色兜底。
        """

        if frame is None or frame.size == 0:
            return False, "无画面", 0.0

        faces = self.detect(frame)
        ratio = self.skin_ratio(frame)
        if faces:
            return True, f"检测到人脸 {len(faces)} 个", ratio
        if ratio >= self._SKIN_RATIO_MIN:
            # 低头/侧脸等 Haar 失效的场景，靠肤色兜住
            return True, f"肤色 {ratio * 100:.0f}%", ratio
        return False, "未检测到人", ratio


# ---------------------------------------------------------------- 规则聚合

class LocalJudge:
    """把各项本地信号聚合成一个 L0 结论。"""

    def __init__(self, cfg: RuleSettings) -> None:
        self.cfg = cfg
        self.detector = FaceDetector()
        self._prev_thumb: np.ndarray | None = None
        self._last_faces: list[tuple[int, int, int, int]] = []
        self._absent_hits = 0  # 连续判定「不在场」的帧数
        self._skin_cache = 0.0

    def evaluate(
        self,
        screen: np.ndarray,
        camera: np.ndarray | None,
        diff_width: int,
        diff_threshold: float,
    ) -> LocalSignals:
        sig = LocalSignals()

        sig.idle_s = get_idle_seconds()
        sig.screen_locked = is_screen_locked()

        # 帧差：判断画面是否有变化
        if screen is not None and screen.size:
            thumb = to_thumb(screen, diff_width)
            if self._prev_thumb is not None and self._prev_thumb.shape == thumb.shape:
                diff = float(np.abs(thumb.astype(np.int16) - self._prev_thumb.astype(np.int16)).mean())
                sig.screen_changed = diff > diff_threshold
                sig.details["screen_diff"] = diff
            else:
                sig.screen_changed = None  # 第一次没有基准
            self._prev_thumb = thumb

        # 在场检测：Haar 主检 + 肤色兜底，再加连续帧确认防抖
        if camera is not None and camera.size:
            here, why, ratio = self.detector.present(camera)
            self._last_faces = self.detector.detect(camera)
            self._skin_cache = ratio
            sig.person_present = here
            sig.person_reason = why

            cam_area = float(camera.shape[0] * camera.shape[1]) or 1.0
            face_area = sum(float(w * h) for _, _, w, h in self._last_faces)
            sig.face_ratio = face_area / cam_area
            sig.skin_ratio = ratio

            # 连续 absent_hit 次「不在场」才判离开。
            # 单帧可能因为 Haar 临界而抖动，直接判离开会误报。
            if here:
                self._absent_hits = 0
            else:
                self._absent_hits += 1
            sig.absent_streak = self._absent_hits
        else:
            sig.face_ratio = None
            sig.person_present = None

        sig.verdict, sig.reason = self._decide(sig)
        return sig

    def _decide(self, sig: LocalSignals) -> tuple[Verdict, str]:
        # 硬规则 1：锁屏 -> 立即判分心，不花钱
        if sig.screen_locked:
            return Verdict.DISTRACT, "屏幕已锁定"

        # 硬规则 2：确认不在场 -> 离开。这是唯一能在本地定案的场景
        if sig.person_present is False:
            need = self.cfg.absence_confirm
            if sig.absent_streak >= need:
                return Verdict.DISTRACT, f"连续 {sig.absent_streak} 帧{sig.person_reason}"

        # 其余一律交给模型看屏幕。
        #
        # 用户明确要求：只有确认「人不在」才能本地定案，其余情况都要让模型
        # 看桌面到底在干什么。原因是「画面活跃」不等于「在学习」——打游戏时
        # 画面同样剧烈变化，本地规则区分不了，必须由模型判断内容。
        if sig.person_present:
            return Verdict.SUSPECT, f"在场（{sig.person_reason}），请模型看屏幕"
        return Verdict.SUSPECT, "需模型判定"

    @property
    def last_faces(self) -> list[tuple[int, int, int, int]]:
        return self._last_faces

    def reset(self) -> None:
        self._prev_thumb = None
        self._last_faces = []
        self._absent_hits = 0
        self._skin_cache = 0.0
        time.sleep(0)
