# MIT License
#
# Copyright (c) 2024-2025 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Optional
from pathlib import Path
import logging as _logging

PathLike = str | Path

FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"

APP_LOGGER: Optional[_logging.Logger] = None
CONSOLE: Optional[_logging.Handler] = None
LOGGING_FORMATTER: Optional[_logging.Formatter] = None
USE_LOGGING: bool = False

DEBUG = _logging.DEBUG
INFO  = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL


def use_logging(value: bool = True) -> bool:
    global USE_LOGGING
    USE_LOGGING = bool(value)
    return USE_LOGGING


def is_logging() -> bool:
    return USE_LOGGING


def set_level(value: int = INFO) -> int:
    _ = get_logger()
    CONSOLE.setLevel(value)
    return CONSOLE.level


def get_logger(
    base_level: int = DEBUG,
    console_level: int = INFO,
) -> _logging.Logger:
    global APP_LOGGER
    if APP_LOGGER is None:
        APP_LOGGER = _init(
            base_level=base_level,
            console_level=console_level,
        )
    return APP_LOGGER


def _init(
    base_level: int = DEBUG,
    console_level: int = INFO,
) -> _logging.Logger:
    global LOGGING_FORMATTER
    global CONSOLE
    bdbc = _logging.getLogger('bdbc')
    existing = list(bdbc.handlers)
    for h in existing:
        bdbc.removeHandler(h)
    bdbc.propagate = False
    bdbc.setLevel(base_level)
    LOGGING_FORMATTER = _logging.Formatter(FORMAT, datefmt=DATEFMT)
    CONSOLE = _logging.StreamHandler()
    CONSOLE.setLevel(console_level)
    CONSOLE.setFormatter(LOGGING_FORMATTER)
    bdbc.addHandler(CONSOLE)
    return bdbc


def critical(msg: str, *args, **kwargs):
    if is_logging():
        get_logger().critical(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    if is_logging():
        get_logger().error(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    if is_logging():
        get_logger().warning(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    if is_logging():
        get_logger().info(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    if is_logging():
        get_logger().debug(msg, *args, **kwargs)


def exception(exc: BaseException, *args, **kwargs):
    if is_logging():
        get_logger().exception(exc, *args, **kwargs)


def test():
    use_logging()
    info('testing info msg')
    debug('testing debug msg')
