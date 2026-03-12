# Copyright (c) AIoWay Authors - All Rights Reserved

"Chunk is a heterogenious collection of in-memory tensor batches."

import dataclasses as dcls
import logging
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Self

import tensordict
from numpy import ndarray as NpArr
from tensordict import TensorDict
from torch import Size, Tensor

from aioway import _typing, attrs
from aioway._tables import Table
from aioway.attrs import AttrSet, AttrSetLike, _validation
from aioway.symbols import SymbolExpr

from .vectors import Vector

__all__ = ["Chunk"]

LOGGER = logging.getLogger(__name__)

type TensorDictLike = TensorDict | dict[str, Tensor]
type ChunkLike = Chunk | dict[str, Vector]


@typing.final
@dcls.dataclass(frozen=True)
class Chunk(Mapping[str, Vector], Table):
    """
    A ``Chunk`` represents a batch of data, following a specific scheam.

    For now, it is immutable as implementing mutable interface is harder.
    This can change in the future.
    """

    data: TensorDict
    "The underlying data."

    attrs: AttrSet
    "The schema for the ``Chunk``."

    def __post_init__(self) -> None:
        _validation.validate_schema(self.attrs, self.data)

    @typing.override
    def __len__(self) -> int:
        "The length of the current batch."

        return len(self.data)

    @typing.override
    def __repr__(self) -> str:
        return f"{self.attrs!r}({len(self)})"

    @typing.override
    def __eq__(self, rhs: object) -> bool:
        # Check if schema and data are both equal.
        if isinstance(rhs, Chunk):
            return self.attrs == rhs.attrs and (self.data == rhs.data).all()

        # If tensordict, don't compare schema.
        if isinstance(rhs, dict | TensorDict):
            return (self.data == rhs).all()

        return NotImplemented

    @typing.overload
    def __getitem__(self, idx: str) -> Vector: ...

    @typing.overload
    def __getitem__(
        self, idx: int | slice | list[int] | list[str] | NpArr | Tensor | Vector, /
    ) -> Self: ...

    @typing.override
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.column(idx)

        # Doing the batch operations that simply does type check first,
        # to avoid expensive iteration over the input.
        if isinstance(idx, int | slice | NpArr | Tensor):
            return self._getitem_direct(idx)

        if isinstance(idx, Vector):
            return self._getitem_direct(idx.data)

        if _typing.is_list_of(str)(idx):
            return self.select(*idx)

        if _typing.is_list_of(int)(idx):
            return self._getitem_direct(idx)

        raise TypeError(type(idx))

    @typing.override
    def __iter__(self) -> Iterator[str]:
        return iter(self.attrs)

    def rename(self, **renames: str) -> Self:
        if not renames:
            return self

        schema = self.attrs.rename(**renames)

        data = _rename(self.data, **renames)
        return self.from_data_schema(data=data, schema=schema)

    def zip(self, rhs: Self) -> Self:
        data = tensordict.merge_tensordicts(self.data, rhs.data)
        schema = {**self.attrs, **rhs.attrs}
        return self.from_data_schema(data=data, schema=schema)

    @typing.override
    def column(self, key: str) -> Vector:
        tensor = self.data[key]
        attr = self.attrs[key]

        # No need to validate as it's already verified in ``Chunk.__post_init__``.
        return Vector(data=tensor, attr=attr)

    @typing.override
    def select(self, *keys: str) -> Self:
        data = self.data.select(*keys)
        schema = {key: self.attrs[key] for key in keys}
        return self.from_data_schema(data=data, schema=schema)

    def _getitem_direct(self, idx: int | slice | list[int] | NpArr | Tensor) -> Self:
        return self.from_data_schema(data=self.data[idx], schema=self.attrs[idx])

    @property
    def shape(self) -> Size:
        return self.data.shape

    def torch(self):
        return self.data

    @classmethod
    def cat(cls, chunks: Sequence[Self]) -> Self:
        if not chunks:
            raise ValueError("Given an empty sequence. Not sure what to do.")

        if len({chunk.attrs for chunk in chunks}) != 1:
            raise ValueError("Chunks should have the same schema before joining.")

        schema = chunks[0].attrs

        data = tensordict.cat([c.data for c in chunks])

        return cls.from_data_schema(schema=schema, data=data)

    @classmethod
    def from_data_schema(cls, data: TensorDictLike, schema: AttrSetLike) -> Self:
        td = _as_tensordict(data)
        td.auto_batch_size_()
        td.auto_device_()
        aset = attrs.attr_set(schema)
        return cls(data=td, attrs=aset)

    @classmethod
    def from_mapping(cls, chunk: ChunkLike) -> Self:
        if isinstance(chunk, cls):
            return chunk

        _is_mapping_of_vector = _typing.is_dict_of_str_to(Vector)
        if _is_mapping_of_vector(chunk):
            data = {key: chunk[key].data for key in chunk.keys()}
            schema = {key: chunk[key].attr for key in chunk.keys()}
            return cls.from_data_schema(data=data, schema=schema)

        raise TypeError(type(chunk))


def _as_tensordict(data: TensorDictLike, /) -> TensorDict:
    if isinstance(data, TensorDict):
        return data

    _is_dict_of_tensor = _typing.is_dict_of_str_to(Tensor)
    if _is_dict_of_tensor(data):
        return TensorDict(data)

    raise TypeError(type(data))


def _col_expr(td: TensorDict, expr: SymbolExpr, /) -> Tensor:
    """
    Given a ``SymbolExpr`` expression, the ``TensorDict`` would be transformed with a tree walk.

    Args:
        td: The ``DictOfTensor`` to manipulate.
        expr:
            The ``SymbolExpr`` expression.
            If a string is given, ``sympify`` is called.

    Raises:
        ValueError: If the free symbols in the expression don't exist in columns.

    Returns:
        The tensor.
    """

    LOGGER.debug("Column expression: %s", expr)

    raise NotImplementedError
    # expr: Basic = sym.sympify(expr, evaluate=False)

    # # Convert symbols to their string representations.
    # keys = [str(s) for s in expr.free_symbols]

    # if any(var not in td for var in keys):
    #     raise ValueError(
    #         f"Expression {expr} contains {keys}, not a subset of {td.keys()}"
    #     )

    # LOGGER.debug("Creating a function with expr=%s, args=%s", expr, keys)
    # # Create a lambda function that works on self.
    # func = sym.lambdify(keys, expr, "numpy")

    # try:
    #     # Unpacking is OK because self is of type `Mapping`.
    #     return func(**td.select(*keys))
    # except TypeError as te:
    #     raise FrameworkUnexpected(sym) from te


def _filter(td: TensorDict, expr: SymbolExpr) -> TensorDict:
    """
    Filter the current ``Block`` with a given expression.
    """

    LOGGER.debug("Filter called with expr=%s", expr)
    idx = _col_expr(td, expr).bool()

    if len(idx) != len(td):
        raise AssertionError(
            f"The result expression has a different legnth than the current sequence. "
            f"Got {len(idx)=} and {len(td)=}."
        )

    # No conversion needed because we know that `index` must be `Tensor`.
    return td[idx]


def _rename(td: TensorDict, **names: str) -> TensorDict:
    """
    Rename the columns of the current ``Block``.
    """

    LOGGER.debug("Renamed called with names=%s", names)
    return TensorDict(
        {names.get(key, key): val for key, val in td.items()},
        batch_size=td.batch_size,
        device=td.device,
    )
