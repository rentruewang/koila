# Copyright (c) AIoWay Authors - All Rights Reserved

"The common operations."

import dataclasses as dcls
import typing
from typing import Any

__all__ = ["expr_dcls"]


@typing.dataclass_transform(eq_default=False)
def expr_dcls[T](cls: T) -> T:
    dataclass: Any = dcls.dataclass
    return dataclass(match_args=False, eq=False, repr=False)(cls)
