# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing


@typing.dataclass_transform(eq_default=False)
@typing.no_type_check
def symbol_dataclass[T: type](cls: T) -> T:
    cls = dcls.dataclass(eq=False)(cls)
    return cls
