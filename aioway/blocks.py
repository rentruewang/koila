# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import operator
import typing
from collections.abc import Callable, Iterator, KeysView, Mapping
from typing import Self

import numpy as np
import sympy as sp
import torch
from numpy.typing import ArrayLike, NDArray
from sympy import Basic, Expr
from tensordict import TensorDict
from torch import Tensor
from torch import device as TorchDevice

from aioway.attrs import Attr, AttrSet, Device, DType, Shape
from aioway.errors import AiowayError

__all__ = ["Block"]


@typing.final
@dcls.dataclass(frozen=True)
class Block(Mapping[str, Tensor]):
    """
    ``Block`` represents a batch that is immutable,
    while providing some additional functionality.

    A ``Block`` is currently a simple wrapper around ``TensorDict``,
    a batch of data that can  move around different devices,
    and some additional checks and utilties.

    In the future, this might be generalized for models
    that use the different length for the same inputs, such as ``torch_geometric``.
    """

    data: TensorDict
    """
    The underlying ``TensorDict`` that is treated as a batch.

    Since ``Block``s are immutable wrappers of ``TensorDict``s that represent a batch,
    only immutable APIs are exposed, and others are wrapped or hidden.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorDict):
            raise BlockTypeError(
                "Underlying data for `Block` should be of type `TensorDict`, "
                f"got {type(self.data)=}"
            )

        # If `batch_dims == 0`, `batch_size[0]` would fail.
        if not self.data.batch_dims:
            raise BlockIndexError(
                "Underlying data for `Block` must have at least 1 batch dimension. Got 0."
            )

    @typing.override
    @typing.no_type_check
    def __eq__(self, other: object) -> Self:
        return self.__td_bin_op(operator.eq, other=other)

    def __ge__(self, other: object) -> Self:
        return self.__td_bin_op(operator.ge, other=other)

    def __gt__(self, other: object) -> Self:
        return self.__td_bin_op(operator.gt, other=other)

    def __le__(self, other: object) -> Self:
        return self.__td_bin_op(operator.le, other=other)

    def __lt__(self, other: object) -> Self:
        return self.__td_bin_op(operator.lt, other=other)

    @typing.override
    def __len__(self) -> int:
        return self.data.batch_size[0]

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    @typing.overload
    def __getitem__(self, idx: str) -> Tensor: ...

    @typing.overload
    def __getitem__(
        self, idx: int | slice | list[str] | list[int] | NDArray | Tensor
    ) -> Self: ...

    @typing.override
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._getitem_str(idx)

        if isinstance(idx, int):
            return self._getitem_int(idx)

        # Normalize the slices before passing in.
        if isinstance(idx, slice):
            return self._getitem_slice(idx)

        # Columns must be a subset of the existing ones.
        if isinstance(idx, list) and all(isinstance(i, str) for i in idx):
            return self._getitem_cols(idx)

        if isinstance(idx, Tensor):
            return self._getitem_tensor(idx)

        # Other are ``ArrayLike``.
        idx = np.array(idx)
        return self._getitem_array(idx)

    # Implemented for ``DataLoader`` to be more efficient.
    __getitems__ = __getitem__

    @typing.override
    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    @typing.override
    def keys(self) -> KeysView[str]:
        return self.data.keys()

    def eval_col_expr(self, str_or_expr: str | Basic, /) -> Tensor:
        """
        Perform evaluate on columns, given a sympy expression.

        Given a ``sympy`` expression, ``lambdify`` would be called,
        and the symbols would be substitued with ``Block``'s columns,
        using symbols as keys, and lambda function would then be called.

        Args:
            str_or_expr:
                The ``sympy`` expression.
                If a string is given, ``sympify`` is called.

        Raises:
            BlockKeyError: If the free symbols in the expression don't exist in columns.
            BlockFilterSympyEvalError: Evaluation is not successful.

        Returns:
            The tensor.
        """

        expr: Basic = sp.sympify(str_or_expr, evaluate=False)

        args: tuple[Expr, ...] = expr.args
        # Convert symbols to their string representations.
        keys = [str(s) for s in expr.free_symbols]

        if any(var not in self.keys() for var in keys):
            raise BlockKeyError(
                f"Expression {expr} contains {keys}, not a subset of {self.keys()}"
            )

        # Create a lambda function that works on self.
        func = sp.lambdify(keys, expr, "numpy")

        try:
            # Unpacking is OK because self is of type ``Mapping``.
            return func(**self[keys])
        except TypeError as te:
            raise BlockSympyEvalError from te

    def filter(self, expr: str | Expr) -> Self:
        idx = self.eval_col_expr(expr).bool()

        if len(idx) != len(self):
            raise BlockIndexError(
                f"The result expression has a different legnth than the current sequence. "
                f"Got {len(idx)=} and {len(self)=}."
            )

        # No conversion needed because we know that ``index`` must be ``Tensor``.
        return self[idx]

    def rename(self, **names: str) -> Self:
        return dcls.replace(
            self,
            data=TensorDict(
                {names.get(key, key): val for key, val in self.data.items()},
                batch_size=self.data.batch_size,
                device=self.data.device,
            ),
        )

    def chain(self, other: Self) -> Self:
        self._check_other_device(other)

        if self.keys() != other.keys():
            raise BlockChainError(
                "Batch keys must match to chain. "
                f"Got {self.keys()=} and {other.keys()=}"
            )

        return type(self)(torch.cat([self.data, other.data], dim=0))

    def zip(self, other: Self) -> Self:
        """
        Concatenate the curernt ``Block`` with another ``Block``.

        Args:
            other: The RHS of the concatenation.

        Raises:
            BlockZipError: If the length of the other is different.

        Returns:
            A new ``Block`` on the same device.
        """

        self._check_other_device(other)

        if len(self) != len(other):
            raise BlockZipError(
                "Can only zip batches with the same length. "
                f"Got {len(self)=} and {len(other)=}"
            )

        return type(self)(
            TensorDict(
                {**self.data, **other.data},
                device=str(self.device) if self.device else None,
                batch_size=[len(self)],
            )
        )

    def require_attrs(self, attr: AttrSet, /) -> None:
        """
        Promises that the current ``Block`` has a given ``TableSchema`` type.
        """

        if attr.keys() != self.keys():
            raise BlockKeyError(
                "Key mismatch. "
                f"Required: {list(attr.keys())}. Actual: {list(self.keys())}"
            )

        if attr.device and self.device and attr.device != self.device:
            raise BlockDeviceError(
                "Device mismatch with schema. "
                f"Required: {attr.device}. Got: {self.device}."
            )

        for key in self.keys():
            if attr[key].dtype == self[key].dtype:
                continue

            raise BlockDTypeError(
                f"For {key=}, {attr[key].dtype=} incompatible with {self[key].dtype=}."
            )

    def _getitem_str(self, idx: str) -> Tensor:
        return self.data[idx]

    def _getitem_cols(self, idx: list[str]) -> Self:
        return type(self)(self.data.select(*idx))

    def _getitem_int(self, idx: int) -> Self:
        # Using a ``list`` instead of passing ``int`` directly,
        # because ``Block`` requires ``batch`` to not be null.
        return type(self)(self.data[[idx]])

    def _getitem_array(self, idx: ArrayLike):
        arr = np.array(idx)

        # For boolean array, be sure that it is of the same length,
        # then convert to an integer index array.
        if np.isdtype(arr.dtype, "bool"):
            if len(arr) != len(self) or arr.ndim != 1:
                raise BlockIndexError(
                    f"Boolean array {arr} is not valid for length: {len(self)}"
                )

            arr = np.arange(len(self))[arr]

        if not np.isdtype(arr.dtype, "integral"):
            raise BlockIndexError(
                f"Array dtype must be boolean or integer. Got {arr.dtype}"
            )

        return type(self)(self.data[arr.tolist()])

    def __getitem_direct(self, idx: object) -> Self:
        return type(self)(self.data[idx])

    _getitem_slice = _getitem_tensor = __getitem_direct

    def to_dict(self) -> dict[str, Tensor]:
        return self.data.to_dict()

    def to_tensor(self) -> Tensor:
        columns: list[Tensor] = []

        for value in self.values():
            columns.append(value.view(len(value), -1))

        return torch.cat(columns, dim=1)

    def detach(self) -> Self:
        return type(self)(self.data.detach())

    def _check_other_device(self, other: Self, /) -> None:
        if self.device == other.device:
            return

        raise BlockDeviceError(
            "Can only merge `TensorDict`s on the same device. "
            f"Got {self.device=} and {other.device=}"
        )

    def _get_device(self) -> Device | None:
        return Device.parse(self.data.device)

    def _set_device(self, device: str | TorchDevice | Device, /) -> None:
        device = Device.parse(device=device)
        self.data.to(device=device.name)

    device = property(fget=_get_device, fset=_set_device)

    def _get_dtype(self) -> DType | None:
        return DType.parse(self.data.dtype)

    dtype = property(fget=_get_dtype)

    def _get_dtypes(self) -> list[DType]:
        return [DType.parse(attr.dtype) for attr in self.attrs.columns.values()]

    dtypes = property(fget=_get_dtypes)

    def _get_shapes(self) -> list[Shape]:
        return [attr.shape for attr in self.attrs.columns.values()]

    shapes = property(fget=_get_shapes)

    @property
    def attrs(self) -> AttrSet:
        return AttrSet(
            columns={key: Attr.parse_tensor(val) for key, val in self.items()},
            device=self.device,
        )

    @property
    def batch_dims(self) -> int:
        return self.data.batch_dims

    @property
    def batch_size(self) -> tuple[int, ...]:
        return self.data.batch_size

    def __td_bin_op(self, op: Callable, /, other: object) -> Self:
        if isinstance(other, Block):
            return type(self)(op(self.data, other.data))

        return type(self)(op(self.data, other))


class BlockTypeError(AiowayError, TypeError): ...


class BlockKeyError(AiowayError, KeyError): ...


class BlockSympyEvalError(AiowayError, RuntimeError): ...


class BlockIndexError(AiowayError, IndexError): ...


class BlockChainError(AiowayError, ValueError): ...


class BlockZipError(AiowayError, ValueError): ...


class BlockDeviceError(AiowayError, ValueError): ...


class BlockDTypeError(AiowayError, ValueError): ...
