# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from aioway.schemas import TableSchema


class Tranformation(Protocol):
    """
    ``Tranformation`` describes the input and output types of a function.
    It tells the compiler how a function would process its inputs,
    and what types of outputs it would create,
    such that the compiler knows what machine learning models to use.
    """

    @abc.abstractmethod
    def __call__(self, schema: TableSchema, /) -> TableSchema: ...
