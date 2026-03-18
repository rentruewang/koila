# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

import lark
from lark import Lark, Transformer

from aioway import _logging

__all__ = ["lark_transformer_dcls", "parse_and_transform_later"]

LOGGER = _logging.get_logger(__name__)


@typing.dataclass_transform(frozen_default=True)
def lark_transformer_dcls(cls):
    "Make a class dataclass, and wrap in `lark.v_args`."
    cls = dcls.dataclass(frozen=True)(cls)
    cls = lark.v_args(inline=True)(cls)
    assert isinstance(cls, type), type(cls)
    return cls


def parse_and_transform_later(
    parser: Callable[[], Lark],
    transformer: Callable[[dict[str, type]], Transformer],
    text: str,
):
    def do_transform(**types: type):
        LOGGER.debug("Parsing %s", text)
        parsed = parser().parse(text)
        LOGGER.debug("Parsed: %s", parsed)
        result = transformer(types).transform(parsed)
        LOGGER.debug("Result: %s", result)
        return result

    return do_transform
