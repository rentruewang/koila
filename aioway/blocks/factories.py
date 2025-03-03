# Copyright (c) RenChu Wang - All Rights Reserved

import typing
from typing import Literal

import deprecated as dprc
from pandas import DataFrame
from tensordict import TensorDict

from aioway.errors import UnknownTypeError

from .blocks import Block
from .pandas import PandasBlock
from .torch import TensorDictBlock

__all__ = ["block"]


@typing.overload
def block(data, kind: Literal["tensordict"]) -> TensorDictBlock: ...


@typing.overload
def block(data, kind: Literal["pandas"]) -> PandasBlock: ...


@dprc.deprecated(reason="See issue #16")
def block(data, kind) -> Block:
    internal_block: Block

    match data:
        case TensorDict():
            internal_block = TensorDictBlock(data)
        case DataFrame():
            internal_block = PandasBlock(data)
        case _:
            raise UnknownTypeError(
                f"Type: {type(data)} is not supported. Must be either a `TensorDict` or a `DataFrame`."
            )

    match kind:
        case "tensordict" | "pandas":
            return internal_block.cast(kind)
        case _:
            raise UnknownTypeError(f"Casting target: '{kind}' is not known.")
