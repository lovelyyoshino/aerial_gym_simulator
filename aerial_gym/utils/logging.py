import logging
from logging import Logger

# 自定义日志格式化类，继承自logging.Formatter
class CustomFormatter(logging.Formatter):
    # 定义不同颜色的ANSI转义序列，用于终端输出
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[37m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    reset = "\x1b[0m"

    # 日志格式字符串，包括相对创建时间、日志名称、级别和消息内容
    format = (
        "[%(relativeCreated)d ms][%(name)s] - %(levelname)s : %(message)s (%(filename)s:%(lineno)d)"
    )

    # 根据日志级别选择对应的格式
    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    # 重写format方法以应用自定义格式
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)  # 获取当前日志级别对应的格式
        formatter = logging.Formatter(log_fmt)  # 创建新的格式化器
        return formatter.format(record)  # 返回格式化后的日志记录


# 自定义日志记录类，继承自Logger
class CustomLogger(Logger):
    def __init__(self, logger_name):
        # 调用父类构造函数
        super().__init__(logger_name)
        self.setLevel(logging.INFO)  # 设置默认日志级别为INFO
        self.ch = logging.StreamHandler()  # 创建流处理器（控制台输出）
        self.ch.setLevel(logging.INFO)  # 设置流处理器的日志级别为INFO
        self.ch.setFormatter(CustomFormatter())  # 设置流处理器的格式化器为CustomFormatter
        self.addHandler(self.ch)  # 将流处理器添加到当前logger中

    # 设置日志记录器的日志级别
    def setLoggerLevel(self, level) -> None:
        self.setLevel(level)  # 设置logger的日志级别
        self.ch.setLevel(level)  # 设置流处理器的日志级别

    # 打印示例日志信息的方法
    def print_example_message(self):
        self.debug("A Debug message will look like this")  # 打印调试信息
        self.info("An Info message will look like this")  # 打印普通信息
        self.warning("A Warning message will look like this")  # 打印警告信息
        self.error("An Error message will look like this")  # 打印错误信息
        self.critical("A Critical message will look like this")  # 打印严重错误信息
