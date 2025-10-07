import logging
from typing import Optional

_LEVEL = logging.INFO
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_def_handler = logging.StreamHandler()
_def_handler.setFormatter(logging.Formatter(_FORMAT))

_def_logger = logging.getLogger("fake_news")
_def_logger.setLevel(_LEVEL)
_def_logger.addHandler(_def_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        return _def_logger
    logger = logging.getLogger(f"fake_news.{name}")
    if not logger.handlers:
        logger.setLevel(_LEVEL)
        logger.addHandler(_def_handler)
    return logger
