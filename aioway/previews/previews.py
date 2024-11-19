# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Callable
from typing import Generic, TypeVar

from aioway.relalg import Relation
from aioway.schemas import DataType

from ..schemas.shapes import EinsumError

__all__ = ["Info", "Preview"]

T = TypeVar("T")


@dcls.dataclass(frozen=True)
class Info:
    """
    The runtime information that is needed at compile time to determine what is available.
    This is very similar to my ``koila`` project's metadata.

    Todo:
        Perhaps update ``koila`` and use the utilities here?
    """

    shape: tuple[int, ...]
    """
    The shape function of the current module.
    """

    dtype: DataType
    """
    The data types of the input tensors.
    """


@dcls.dataclass(frozen=True)
class Preview(Generic[T]):
    relation: type[Relation]
    """
    The relation for which this preview module is for.
    """

    num_params: int
    """
    The number of parameters for a given function.
    """

    initialization: Callable[[], T]
    """
    The hooks to initialize the backend engines directly from ``Preview`` objects.
    """

    def compute(self, *infos: Info) -> Info:
        try:
            return self._compute(*infos)
        except EinsumError:
            return NotImplemented

    def _compute(self, *infos: Info) -> Info:
        """
        Compute the output

        Args:
            *infos: The symbolic representation for the runtime information of tensors.

        Returns:
            The computed information for the runtime.
            If ``NotImplemented`` is returned,
            this means that the input formats are not supported by the operator.
        """

        _ = infos

        return NotImplemented
