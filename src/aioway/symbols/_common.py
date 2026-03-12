# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing


@typing.dataclass_transform(eq_default=False)
def symbol_dataclass[T: type](cls: T) -> T:
    return typing.cast(T, dcls.dataclass(eq=False)(cls))
