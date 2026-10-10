"""告警通道：桌面通知 + 提示音。

两个通道都做了失败静默：通知发不出去不该影响监控主流程。
"""

from __future__ import annotations

import sys


def _beep(times: int = 3) -> None:
    """响一声。Windows 用 winsound 蜂鸣器，跨平台退回终端 BEL。"""

    try:
        if sys.platform == "win32":
            import winsound

            for _ in range(times):
                winsound.Beep(880, 220)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def notify(title: str, message: str, *, sound: bool = True, app_name: str = "电子教官") -> bool:
    """发桌面通知。返回是否成功发出。"""

    ok = False
    try:
        from plyer import notification

        notification.notify(
            title=title,
            message=message,
            app_name=app_name,
            timeout=8,
        )
        ok = True
    except Exception:
        ok = False

    if sound:
        _beep()

    if not ok:
        # 通知不可用时，至少把消息打到终端
        print(f"[{title}] {message}")
    return ok
