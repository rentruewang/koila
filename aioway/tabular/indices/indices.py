# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Sequence
from weakref import ReferenceType

from aioway.tabular.frames import Frame

__all__ = ["Index"]


@dcls.dataclass(frozen=True)
class Index[T](ABC):
    frame: ReferenceType[Frame]
    columns: Sequence[str]

    @abc.abstractmethod
    def __call__(self, value: T) -> list[int]: ...
