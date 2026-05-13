import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "xtts_server.log")
_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_initialized = False


def _ensure_log_dir() -> None:
    os.makedirs(_LOG_DIR, exist_ok=True)


def _resolve_level() -> int:
    raw = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, raw, logging.INFO)


def _setup_root() -> None:
    global _initialized
    if _initialized:
        return

    _ensure_log_dir()

    level = _resolve_level()
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)

    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if called multiple times in the same process.
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    _setup_root()
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level())
    return logger
