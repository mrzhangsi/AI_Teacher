"""测试隔离层：保证测试永远不会用到真实 API key、也不会发网络请求。

背景：用户项目根目录可能有真实的 .env。如果测试直接调 get_settings()，
就会读到真实 key；一旦走到 L1 判定分支就会真发请求并扣费。

本模块的导入**必须发生在所有测试代码之前**（在 test_e2e.py / test_gui.py
的最顶部），通过环境变量 + settings 缓存清空双重保险：

1. 把 AI_TEACHER_API_KEY 强制设为空串 —— pydantic-settings 读 .env 时
   优先级低于进程环境变量，所以这个空值会覆盖 .env 里的真实 key
2. 把 base_url 指向一个不可路由的地址，即使有人绕过拦截也会失败快速
3. 清掉 settings 单例缓存，强制下次 get_settings() 重新读取上述设置

用法：
    import tests.isolation  # noqa: F401  必须放在测试文件最顶部
"""

from __future__ import annotations

import os

# 必须在导入 ai_teacher 任何模块之前设置（pydantic-settings 初始化时读）
os.environ["AI_TEACHER_API_KEY"] = ""
os.environ["AI_TEACHER_BASE_URL"] = "http://127.0.0.1:9/unreachable-test-endpoint"

# 再清一次缓存，确保 get_settings() 重新构造并读到上面的环境变量。
# ai_teacher.config 在此时可能还没被导入，先 try 一下。
try:  # pragma: no cover - 取决于导入顺序
    from ai_teacher import config as _cfg

    _cfg._settings = None
except Exception:  # pragma: no cover
    pass


def assert_isolated() -> tuple[bool, str]:
    """自检：确认当前进程读不到真实 key。返回 (是否隔离, 说明)。"""

    try:
        from ai_teacher.config import get_settings

        s = get_settings()
        if s.api.api_key:
            return False, f"仍读到 key: {s.api.api_key[:6]}..."
        if "unreachable" not in s.api.base_url:
            return False, f"base_url 未被隔离: {s.api.base_url}"
        return True, "已隔离"
    except Exception as exc:  # pragma: no cover
        return False, f"自检异常: {exc}"


# 模块导入时立刻自检一次，结果打印出来，便于测试报告里一眼看到
_ISOLATED, _MSG = assert_isolated()
if not _ISOLATED:  # pragma: no cover
    import sys

    print(f"[WARNING] 测试隔离未生效：{_MSG}", file=sys.stderr)