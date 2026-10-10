"""图像采集与合成：屏幕截图 + 摄像头画面 -> 单张 JPEG。

设计要点：
1. 屏幕与摄像头上下拼成一张图，只发一次请求、只算一份图片 token。
2. 缩到长边 compose_width、JPEG 压缩后体积约 25-40 KB，上传快、重试成本低。
3. detail=low 时服务端会再缩到 512x512，所以不必压到更小（服务端也会放大回来）。
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import cv2
import mss
import numpy as np

from .config import CaptureSettings


@dataclass(slots=True)
class Capture:
    """一次采集的原始结果。"""

    screen: np.ndarray
    camera: np.ndarray | None


class ScreenGrabber:
    """mss 屏幕抓取器，复用底层句柄以避免每次重建。"""

    def __init__(self) -> None:
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        """抓取主显示器，返回 BGR ndarray。"""

        monitor = self._sct.monitors[1] if len(self._sct.monitors) > 1 else self._sct.monitors[0]
        raw = self._sct.grab(monitor)
        # mss 返回 BGRA
        return np.asarray(raw)[:, :, :3].copy()

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


class CameraReader:
    """摄像头读取器。设备缺失时静默降级为 None，不影响主流程。"""

    def __init__(self, index: int) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._index = index

    def _ensure(self) -> cv2.VideoCapture | None:
        if self._cap is not None:
            return self._cap
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            cap.release()
            return None
        # 降低分辨率以省CPU：判定只需要大致轮廓
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap = cap
        return cap

    def read(self) -> np.ndarray | None:
        cap = self._ensure()
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _resize_width(img: np.ndarray, width: int) -> np.ndarray:
    """按宽度等比缩放。宽度<=1 时原样返回。"""

    h, w = img.shape[:2]
    if w == width or width <= 1:
        return img
    scale = width / w
    return cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def compose(screen: np.ndarray, camera: np.ndarray | None, cfg: CaptureSettings) -> bytes:
    """把屏幕（主）与摄像头（辅）拼成一张 JPEG。

    摄像头按 camera_ratio 分配宽度：判定「人是否在」不需要高分辨率，
    留窄一点既省token 也让主体更清晰。
    """

    total_w = cfg.compose_width
    cam_w = int(total_w * cfg.camera_ratio) if camera is not None else 0

    screen_part = _resize_width(screen, total_w)
    parts = [screen_part]
    if camera is not None and cam_w > 0:
        parts.append(_resize_width(camera, cam_w))

    if len(parts) == 1:
        merged = parts[0]
    else:
        # 顶部对齐，摄像头挂在屏幕下方
        top_h = parts[0].shape[0]
        cam_part = parts[1]
        gap = 4
        canvas = np.zeros((top_h + gap + cam_part.shape[0], total_w, 3), dtype=np.uint8)
        canvas[:top_h, : screen_part.shape[1]] = screen_part
        canvas[top_h + gap :, :cam_part.shape[1]] = cam_part
        merged = canvas

    ok, buf = cv2.imencode(".jpg", merged, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def to_data_url(jpeg: bytes) -> str:
    """转成 API 需要的 data URL。"""

    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def to_thumb(img: np.ndarray, width: int) -> np.ndarray:
    """生成用于帧差检测的小灰度图。"""

    small = _resize_width(img, width)
    if small.shape[1] > width:
        small = small[:, :width]
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
