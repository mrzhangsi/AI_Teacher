"""验证测试套件在任何情况下都不会发出网络请求。

单独运行（不要和其他测试一起跑，会互相干扰）：
    uv run python -m tests.test_no_network

原理：劫持 socket.connect / connect_ex，任何出站 TCP 尝试都被记录并
抛错。然后完整跑一遍 e2e + GUI 套件，最后断言尝试次数为 0。

这是「测试不会用真实 key 扣费」的最终保险——比断言配置值更硬，
因为它直接证明没有任何数据包离开本机。
"""

from __future__ import annotations

import socket
import sys


def main() -> int:
    attempts: list = []

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def spy_connect(self, addr, *a, **k):  # type: ignore[no-untyped-def]
        attempts.append(addr)
        raise AssertionError(f"测试期间尝试连接 {addr} —— 隔离失败！")

    def spy_connect_ex(self, addr, *a, **k):  # type: ignore[no-untyped-def]
        attempts.append(addr)
        raise OSError(f"测试期间尝试连接 {addr} —— 隔离失败！")

    socket.socket.connect = spy_connect
    socket.socket.connect_ex = spy_connect_ex

    try:
        print("[bold]零网络访问验证[/]")
        print("（socket 已劫持，任何出站连接都会被捕获）\n")

        import tests.isolation  # noqa: F401  必须在最前

        from tests import test_e2e

        e2e_rc = test_e2e.main()
        attempts_after_e2e = len(attempts)

        from tests import test_gui

        gui_rc = test_gui.main()
    finally:
        socket.socket.connect = orig_connect
        socket.socket.connect_ex = orig_connect_ex

    print()
    print("[bold]结果[/]")
    print(f"  e2e 套件 exit   : {e2e_rc}")
    print(f"  gui 套件 exit   : {gui_rc}")
    print(f"  e2e 期间出站连接: {attempts_after_e2e}")
    print(f"  总出站连接次数  : {len(attempts)}")

    if attempts:
        print("\n  [red]FAIL[/] 测试期间发生了网络访问：")
        for addr in attempts[:10]:
            print(f"         {addr}")
        return 1

    if e2e_rc or gui_rc:
        print("\n  [red]FAIL[/] 有测试项未通过（见上方输出）")
        return 1

    print("\n  [green]PASS[/] 全程零网络访问，不会产生任何 API 费用")
    return 0


if __name__ == "__main__":
    sys.exit(main())