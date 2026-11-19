# Python 启动时会自动 import 名为 sitecustomize 的模块（只要它在 PYTHONPATH 上）。
# vLLM 的每个 worker 进程都会继承 PYTHONPATH，所以补丁能打到所有进程里。
# 只有 GDN_TUNE=1 时才会真正打补丁，否则什么都不做。
import os

if os.environ.get("GDN_TUNE", "0") == "1":
    from gdn_tune_patch import maybe_apply
    maybe_apply()
