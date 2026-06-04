

import logging

import os

import sys

from logging.handlers import RotatingFileHandler

from config import Config

_DETAILED_FMT = logging.Formatter(

    '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',

    datefmt='%Y-%m-%d %H:%M:%S'

)

_SIMPLE_FMT = logging.Formatter(

    '[%(asctime)s] %(levelname)-8s %(message)s',

    datefmt='%H:%M:%S'

)

class FlushStreamHandler(logging.StreamHandler):

    def emit(self, record: logging.LogRecord) -> None:  

        super().emit(record)

        self.flush()

def _bootstrap_root_logger() -> None:

    root = logging.getLogger()

    if root.level == logging.WARNING or root.level == logging.NOTSET:

        root.setLevel(logging.DEBUG)

    for noisy in ('urllib3', 'requests', 'matplotlib', 'PIL', 'yfinance'):

        logging.getLogger(noisy).setLevel(logging.WARNING)

_bootstrap_root_logger()

def setup_logger(name: str, log_file: str = None, level: str = None) -> logging.Logger:

    log_level = getattr(logging, (level or Config.LOG_LEVEL).upper(), logging.INFO)

    logger = logging.getLogger(name)

    logger.setLevel(log_level)

    logger.propagate = False

    has_console = any(isinstance(h, FlushStreamHandler) for h in logger.handlers)

    if not has_console:

        console_handler = FlushStreamHandler(sys.stdout)

        console_handler.setLevel(logging.DEBUG)

        console_handler.setFormatter(_SIMPLE_FMT)

        logger.addHandler(console_handler)

    else:

        for h in logger.handlers:

            if isinstance(h, FlushStreamHandler):

                h.setLevel(logging.DEBUG)

    log_path = log_file or Config.LOG_FILE

    has_file = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)

    if log_path and not has_file:

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        file_handler = RotatingFileHandler(

            log_path,

            maxBytes=10 * 1024 * 1024,  

            backupCount=5

        )

        file_handler.setLevel(log_level)

        file_handler.setFormatter(_DETAILED_FMT)

        logger.addHandler(file_handler)

    return logger

app_logger = setup_logger('stock_prediction')

