# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import operator
from collections.abc import Callable
from numbers import Number
from typing import Any, Self

import torch
from pandas import Series
from torch import Tensor

from aioway.schemas import DataType, DataTypeEnum


@dcls.dataclass(frozen=True)
class Buffer:
    """
    ``Column`` represents a single column in the ``Block`` stored in memory.
    This is the core representation of an in-memory column data type.

    Todo:
        Maybe merge some functionality of this class and ``Block`` class.
    """

    data: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.data, Tensor):
            raise TypeError(
                f"Expected data to be of type Tensor, got {type(self.data)=}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Any) -> Tensor:
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def __str__(self) -> str:
        return str(self.data)

    def __eq__(self, other: Self) -> Self:  # type: ignore[override]
        return self.__bin_op(other, operator.eq)

    def __ne__(self, other: Self) -> Self:  # type: ignore[override]
        return self.__bin_op(other, operator.ne)

    def __add__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.add)

    def __sub__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.sub)

    def __mul__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.mul)

    def __truediv__(self, other: Self) -> Self:
        return self.__bin_op(other, operator.truediv)

    def __bin_op(
        self,
        other: Self | Tensor | Number,
        op: Callable[[Tensor, Tensor | Number], Tensor],
    ) -> Self:
        # Usually, the ``numpy`` family's arithmetic operations would work with python number,
        # therefore, they are included in the type checks for implemented methods.
        if not isinstance(other, (type(self), Tensor, Number)):
            return NotImplemented

        if not callable(op):
            raise ValueError(
                f"Operation {op} is not callable. Must be a callabe with 2 arguments."
            )

        # We want to leverage the existing operators implemented by ``TensorDict``,
        # since they support built-in types and other ``TensorDict``s, but not our types.
        if not isinstance(other, Buffer):
            operand = other
        else:
            operand = other.data

        return type(self)(op(self.data, operand))

    def series(self) -> Series:
        return Series(self.data.detach().cpu().numpy())

    @property
    def dtype(self) -> str | None:
        return str(self.data.dtype) if self.data.dtype is not None else None

    @property
    def device(self) -> str | None:
        return str(self.data.device) if self.data.device is not None else None

    @property
    def datatype(self) -> DataType:
        """
        The column type from the current block.

        Todo:
            Currently, the block is implemented using a mapping from ``torch.dtype``
            to aioway's ``Dtype``. This sin't a great solution, make changes.
        """

        FROM_TORCH_DTYPE = {
            torch.long: DataTypeEnum.INT[64](),
            torch.int: DataTypeEnum.INT[32](),
            torch.float: DataTypeEnum.FLOAT[32](),
            torch.double: DataTypeEnum.FLOAT[64](),
            torch.bool: DataTypeEnum.BOOL(),
        }
        return FROM_TORCH_DTYPE[self.data.dtype]
