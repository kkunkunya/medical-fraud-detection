# Input: 无
# Output: 配置好的 logger 实例
# Pos: 全局日志配置，所有模块导入使用
# Warning: 更新时同步更新注释和 _ARCH.md

"""
项目日志配置模块
- 日志位置: log/app.log
- 轮转策略: 500KB 自动轮转，保留 2 个备份
- 格式: [01-15 14:30:22] INFO | module_name | 消息
- 支持实时查看 (unbuffered writes)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# 项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "log"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

# 日志格式
LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%m-%d %H:%M:%S"


class FlushFileHandler(RotatingFileHandler):
    """支持实时刷新的 RotatingFileHandler"""

    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    获取配置好的 logger 实例

    Args:
        name: 日志模块名称，会显示在日志中
        level: 日志级别，默认 INFO

    Returns:
        配置好的 logger 实例

    Example:
        from src.utils.logger_config import get_logger
        logger = get_logger("trainer")
        logger.info("开始训练")
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 文件处理器 - 500KB 轮转，保留 2 个备份，实时刷新
    file_handler = FlushFileHandler(
        LOG_FILE,
        maxBytes=500_000,  # 500KB
        backupCount=2,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    # 控制台处理器 - 简洁格式，实时刷新
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_training_logger(experiment_name: Optional[str] = None) -> logging.Logger:
    """
    设置训练专用日志器

    Args:
        experiment_name: 实验名称，可选

    Returns:
        配置好的训练日志器
    """
    name = f"train.{experiment_name}" if experiment_name else "train"
    return get_logger(name)


# 默认 logger 实例
logger = get_logger()


if __name__ == "__main__":
    # 测试日志功能
    test_logger = get_logger("test")
    test_logger.info("日志系统测试")
    test_logger.warning("警告测试")
    test_logger.error("错误测试")
    print(f"\n日志文件位置: {LOG_FILE}")
