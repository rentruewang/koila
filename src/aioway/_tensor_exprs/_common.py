# Copyright (c) AIoWay Authors - All Rights Reserved

"The common operations."

import dataclasses as dcls
import typing

import torch

from aioway import _tracking

__all__ = ["expr_dcls", "TRACKER"]

TRACKER = _tracking.get_tracker(lambda: torch.Tensor)


@typing.dataclass_transform(eq_default=False)
def expr_dcls[T](cls: T) -> T:
    dataclass: typing.Any = dcls.dataclass
    return dataclass(match_args=False, eq=False, repr=False)(cls)
