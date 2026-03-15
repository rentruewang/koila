# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import logging
from logging import Handler

from rich.logging import RichHandler

__all__ = ["enable_logging", "enable_rich_logging"]


@ctxl.contextmanager
def enable_rich_logging(level: str | int, /):
    """
    Enable logging for the duration of the block with rich handlers.
    """

    with enable_logging(level, RichHandler()) as logger:
        yield logger


@ctxl.contextmanager
def enable_logging(level: str | int, /, *handlers: Handler):
    """
    Enable logging for the duration of the block package wide.
    """

    logger = logging.getLogger("aioway")

    before_level = logger.level
    before_handlers = list(logger.handlers)

    for handler in handlers:
        logger.addHandler(handler)
        logger.setLevel(level)

    try:
        yield logger
    finally:
        logger.setLevel(before_level)
        logger.handlers = before_handlers
