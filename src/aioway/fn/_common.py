# Copyright (c) AIoWay Authors - All Rights Reserved

"The common operations."

import dataclasses as dcls
import typing
from typing import Any

from torch import Tensor

from aioway._tracking import ModuleApiTracker

__all__ = ["fn_dcls"]

TRACKER = ModuleApiTracker(lambda: Tensor)


@typing.dataclass_transform(eq_default=False)
def fn_dcls[T](cls: T) -> T:
    dataclass: Any = dcls.dataclass
    return dataclass(eq=False)(cls)
