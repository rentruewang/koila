# Copyright (c) AIoWay Authors - All Rights Reserved

"The common operations."

import dataclasses as dcls
import typing

import torch

from aioway._tracking import ModuleApiTracker

__all__ = ["expr_dcls", "TRACKER"]

TRACKER = ModuleApiTracker(lambda: torch.Tensor)


@typing.dataclass_transform(eq_default=False)
def expr_dcls[T](cls: T) -> T:
    dataclass: typing.Any = dcls.dataclass
    return dataclass(match_args=False, eq=False, repr=False)(cls)
