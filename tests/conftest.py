"""测试夹具。

给配置项填上假的 Key，让不碰网络的单元测试（切分、DTO、状态机）
不必依赖本地 `.env` 的真实密钥，也不必联网。

注意：真实调模型的测试要显式标记，别混进来 —— 那些会花钱。
"""

from __future__ import annotations

import os

# 环境变量优先级高于 .env 文件，所以这里 setdefault 不会覆盖真实配置
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-used")
os.environ.setdefault("SILICONFLOW_API_KEY", "test-key-not-used")
