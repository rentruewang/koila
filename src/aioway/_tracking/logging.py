# Copyright (c) AIoWay Authors - All Rights Reserved

"Turning on logging globally for a duration."

import contextlib as ctxl
import logging
from logging import Handler

from rich.logging import RichHandler

__all__ = ["enable", "enable_rich"]


@ctxl.contextmanager
def enable_rich(level: str | int, /):
    """
    Enable logging for the duration of the block with rich handlers.
    """

    with enable(level, RichHandler()) as logger:
        yield logger


@ctxl.contextmanager
def enable(level: str | int, /, *handlers: Handler):
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
