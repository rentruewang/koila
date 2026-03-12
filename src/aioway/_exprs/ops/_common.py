# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from collections.abc import Callable

import lark
from lark import Lark, Transformer

LOGGER = logging.getLogger(__name__)


@typing.dataclass_transform(frozen_default=True)
def lark_transformer_dcls(cls):
    "Make a class dataclass, and wrap in ``v_args``."
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
