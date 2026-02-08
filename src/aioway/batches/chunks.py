# Copyright (c) AIoWay Authors - All Rights Reserved

"Chunk is a heterogenious collection of in-memory tensor batches."

import logging
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Self, TypeIs

import tensordict
from numpy import ndarray as NpArr
from sympy import Expr
from tensordict import TensorDict
from torch import Size, Tensor

from aioway.attrs import Attr, AttrSet, _validation
from aioway.attrs import funcs as atf

from . import funcs

__all__ = ["Chunk"]

LOGGER = logging.getLogger(__name__)

type DataLike = TensorDict | dict[str, Tensor]
type SchemaLike = AttrSet | dict[str, Attr]


class Chunk(Mapping[str, Tensor]):
    """
    A ``Chunk`` represents a batch of data, following a specific scheam.

    For now, it is immutable as implementing mutable interface is harder.
    This can change in the future.
    """

    def __init__(self, data: DataLike, schema: SchemaLike) -> None:
        self._data: TensorDict = _as_tensordict(data)
        """
        Underlying tensordict backing the ``Block``.
        """

        self._schema: AttrSet = _as_schema(schema)
        """
        The schema of the current ``Block``.
        """

        # Check if the inputs of ``__init__`` is ok.
        # Raise ``ValueError`` or ``TypeError`` if the inputs are not as expected.
        self._check_init()

        # Check if the data matches the schema.
        _validation.validate_schema(self._schema, self._data)

    @typing.override
    def __len__(self) -> int:
        "The length of the current batch."

        return len(self._data)

    @typing.override
    def __repr__(self) -> str:
        return f"{self._schema!r}({len(self)})"

    @typing.override
    def __eq__(self, rhs: object) -> bool:
        # Check if schema and data are both equal.
        if isinstance(rhs, Chunk):
            return self.schema == rhs.schema and (self._data == rhs._data).all()

        # If tensordict, don't compare schema.
        if isinstance(rhs, dict | TensorDict):
            return (self._data == rhs).all()

        return NotImplemented

    @typing.overload
    def __getitem__(self, idx: str) -> Tensor: ...

    @typing.overload
    def __getitem__(
        self,
        idx: int | slice | list[int] | list[str] | NpArr | Tensor | tuple[Any, ...],
        /,
    ) -> Self: ...

    @typing.override
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._getitem_str(idx)

        # Doing the batch operations that simply does type check first,
        # to avoid expensive iteration over the input.
        if isinstance(idx, int | tuple | slice | NpArr | Tensor):
            return self._getitem_direct(idx)

        if _is_list_of_str(idx):
            return self._getitem_list_str(idx)

        if _is_list_of_int(idx):
            return self._getitem_direct(idx)

        raise NotImplementedError(f"Unsupported {type(idx)=}")

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return iter(self._schema)

    def rename(self, **renames: str) -> Self:
        if not renames:
            return self

        schema = atf.renames(self.schema, **renames)
        data = funcs.rename(self._data, **renames)
        return type(self)(data=data, schema=schema)

    def filter(self, expr: str | Expr) -> Self:
        return type(self)(data=funcs.filter(self._data, expr), schema=self.schema)

    def zip(self, rhs: Self) -> Self:
        data = tensordict.merge_tensordicts(self._data, rhs._data)
        schema = {**self.schema, **rhs.schema}
        return type(self)(data=data, schema=schema)

    def _getitem_str(self, key: str) -> Tensor:
        return self._data[key]

    def _getitem_list_str(self, keys: list[str]) -> Self:
        data = self._data.select(*keys)
        schema = {key: self.schema[key] for key in keys}
        return type(self)(data=data, schema=schema)

    def _getitem_direct(
        self, idx: slice | list[int] | Tensor | tuple[Any, ...]
    ) -> Self:
        return type(self)(data=self._data[idx], schema=atf.index(self.schema, idx))

    @property
    def schema(self) -> AttrSet:
        "The schema of a ``Chunk``."

        return self._schema

    @property
    def shape(self) -> Size:
        return self._data.shape

    @typing.no_type_check
    def _check_init(self) -> None:
        if not isinstance(self._data, TensorDict):
            raise TypeError(
                f"Expected a `TensorDict`, got type of data: {type(self._data)}"
            )

        if not isinstance(self._schema, AttrSet):
            raise TypeError(
                f"Expected an `AttrSet`, got type of schema: {type(self.schema)}"
            )

    @classmethod
    def cat(cls, chunks: Sequence[Self]) -> Self:
        if not chunks:
            raise ValueError("Given an empty sequence. Not sure what to do.")

        if len({chunk.schema for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same schema before joining.")

        schema = chunks[0].schema

        data = tensordict.cat([c._data for c in chunks])

        return cls(schema=schema, data=data)


def _as_tensordict(data: DataLike) -> TensorDict:
    if isinstance(data, TensorDict):
        return data

    if _is_dict_of_tensor(data):
        return TensorDict(data)

    raise TypeError(
        f"Unknown: {type(data)=}. Only accepts `TensorDict` or `dict[str, Tensor]`."
    )


def _as_schema(schema: SchemaLike) -> AttrSet:
    if isinstance(schema, AttrSet):
        return schema

    if _is_dict_of_attr(schema):
        return AttrSet.from_dict(schema)

    raise TypeError(
        f"Unknown: {type(schema)=}. Only accepts `AttrSet` or `dict[str, Attr]`."
    )


def _is_dict_of_tensor(obj) -> TypeIs[dict[str, Tensor]]:
    return (
        True
        and isinstance(obj, dict)
        and all(isinstance(key, str) for key in obj.keys())
        and all(isinstance(val, Tensor) for val in obj.values())
    )


def _is_dict_of_attr(obj) -> TypeIs[dict[str, Attr]]:
    return (
        True
        and isinstance(obj, dict)
        and all(isinstance(key, str) for key in obj.keys())
        and all(isinstance(val, Attr) for val in obj.values())
    )


def _is_list_of_str(obj) -> TypeIs[list[str]]:
    return isinstance(obj, list) and all(isinstance(i, str) for i in obj)


def _is_list_of_int(obj) -> TypeIs[list[int]]:
    return isinstance(obj, list) and all(isinstance(i, int) for i in obj)
