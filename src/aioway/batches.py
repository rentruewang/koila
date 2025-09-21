# Copyright (c) AIoWay Authors - All Rights Reserved

import logging

import sympy as sym
import torch
from sympy import Basic, Expr
from tensordict import TensorDict
from torch import Tensor

from aioway.attrs import AttrSet
from aioway.errors import AiowayError

__all__ = ["Block"]

LOGGER = logging.getLogger(__name__)


def tensordict_col_expr(td: TensorDict, str_or_expr: str | Basic, /) -> Tensor:
    """
    Perform evaluate on columns, given a sympy expression.

    Given a ``sympy`` expression, ``lambdify`` would be called,
    and the symbols would be substitued with ``Block``'s columns,
    using symbols as keys, and lambda function would then be called.

    Args:
        td: The ``TensorDict`` to manipulate.
        str_or_expr:
            The ``sympy`` expression.
            If a string is given, ``sympify`` is called.

    Raises:
        DictKeyError: If the free symbols in the expression don't exist in columns.
        SympyEvalError: Evaluation is not successful.

    Returns:
        The tensor.
    """

    LOGGER.debug("Column expression: %s", str_or_expr)

    expr: Basic = sym.sympify(str_or_expr, evaluate=False)

    # Convert symbols to their string representations.
    keys = [str(s) for s in expr.free_symbols]

    if any(var not in td.keys() for var in keys):
        raise DictKeyError(
            f"Expression {expr} contains {keys}, not a subset of {td.keys()}"
        )

    LOGGER.debug("Creating a function with expr=%s, args=%s", expr, keys)
    # Create a lambda function that works on self.
    func = sym.lambdify(keys, expr, "numpy")

    try:
        # Unpacking is OK because self is of type `Mapping`.
        return func(**td.select(*keys))
    except TypeError as te:
        raise SympyEvalError from te


def tensordict_filter(td: TensorDict, expr: str | Expr) -> TensorDict:
    """
    Filter the current ``Block`` with a given expression.
    """

    LOGGER.debug("Filter called with expr=%s", expr)
    idx = tensordict_col_expr(td, expr).bool()

    if len(idx) != len(td):
        raise BlockIndexError(
            f"The result expression has a different legnth than the current sequence. "
            f"Got {len(idx)=} and {len(td)=}."
        )

    # No conversion needed because we know that `index` must be `Tensor`.
    return td[idx]


def tensordict_rename(td: TensorDict, **names: str) -> TensorDict:
    """
    Rename the columns of the current ``Block``.
    """

    LOGGER.debug("Renamed called with names=%s", names)
    return TensorDict(
        {names.get(key, key): val for key, val in td.data.items()},
        batch_size=td.data.batch_size,
        device=td.data.device,
    )


def tensordict_chain(left: TensorDict, right: TensorDict) -> TensorDict:
    """
    Concatenate the current ``TensorDict`` with another ``TensorDict``, vertically.

    Args:
        left: The LHS of the concatenation.
        right: The RHS of the concatenation.

    Raises:
        BlockChainError: If the length of the other is different.

    Returns:
        A new ``Block`` on the same device.
    """

    LOGGER.debug("Chain called with self=%s, other=%s", left, right)

    if left.keys() != right.keys():
        raise BatchChainError(
            "Batch keys must match to chain. " f"Got {left.keys()=} and {right.keys()=}"
        )

    return torch.cat([left.data, right.data], dim=0)


def tensordict_require_attrs(self: TensorDict, attrs: AttrSet, /) -> None:
    """
    Promises that the current ``Block`` has a given ``TableSchema`` type.
    """

    LOGGER.debug(
        "Requiring attrs of self: %s, other: %s to be equal", self.attrs, attrs
    )

    if attrs.keys() != self.keys():
        raise DictKeyError(
            "Key mismatch. "
            f"Required: {list(attrs.keys())}. Actual: {list(self.keys())}"
        )

    if attrs.device and self.device and attrs.device != self.device:
        raise BlockDeviceError(
            "Device mismatch with schema. "
            f"Required: {attrs.device}. Got: {self.device}."
        )

    for key in self.keys():
        if attrs[key].dtype == self[key].dtype:
            continue

        raise BlockDTypeError(
            f"For {key=}, {attrs[key].dtype=} incompatible with {self[key].dtype=}."
        )


def tensordict_to_tensor(td: TensorDict) -> Tensor:
    columns: list[Tensor] = []

    for value in td.values():
        columns.append(value.view(len(value), -1))

    return torch.cat(columns, dim=1)


class BlockTypeError(AiowayError, TypeError): ...


class DictKeyError(AiowayError, KeyError): ...


class SympyEvalError(AiowayError, RuntimeError): ...


class BlockIndexError(AiowayError, IndexError): ...


class BatchChainError(AiowayError, ValueError): ...


class BlockZipError(AiowayError, ValueError): ...


class BlockDeviceError(AiowayError, ValueError): ...


class BlockDTypeError(AiowayError, ValueError): ...
