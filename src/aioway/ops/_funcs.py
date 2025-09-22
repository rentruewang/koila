# Copyright (c) AIoWay Authors - All Rights Reserved

import logging

import sympy as sym
import torch
from sympy import Basic, Expr
from tensordict import TensorDict
from torch import Tensor

from aioway._errors import AiowayError

LOGGER = logging.getLogger(__name__)


def col_expr(td: TensorDict, str_or_expr: str | Basic, /) -> Tensor:
    """
    Perform evaluate on columns, given a sympy expression.

    Given a ``sympy`` expression, ``lambdify`` would be called,
    and the symbols would be substitued with ``Block``'s columns,
    using symbols as keys, and lambda function would then be called.

    Args:
        td: The ``DictOfTensor`` to manipulate.
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

    if any(var not in td for var in keys):
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


def filter(td: TensorDict, expr: str | Expr) -> TensorDict:
    """
    Filter the current ``Block`` with a given expression.
    """

    LOGGER.debug("Filter called with expr=%s", expr)
    idx = col_expr(td, expr).bool()

    if len(idx) != len(td):
        raise BlockIndexError(
            f"The result expression has a different legnth than the current sequence. "
            f"Got {len(idx)=} and {len(td)=}."
        )

    # No conversion needed because we know that `index` must be `Tensor`.
    return td[idx]


def rename(td: TensorDict, **names: str) -> TensorDict:
    """
    Rename the columns of the current ``Block``.
    """

    LOGGER.debug("Renamed called with names=%s", names)
    return TensorDict(
        {names.get(key, key): val for key, val in td.items()},
        batch_size=td.batch_size,
        device=td.device,
    )


def to_tensor(td: TensorDict) -> Tensor:
    columns: list[Tensor] = []

    for value in td.values():
        columns.append(value.view(len(value), -1))

    return torch.cat(columns, dim=1)


class BlockTypeError(AiowayError, TypeError): ...


class DictKeyError(AiowayError, KeyError): ...


class SympyEvalError(AiowayError, RuntimeError): ...


class BlockIndexError(AiowayError, IndexError): ...


class BlockZipError(AiowayError, ValueError): ...


class BlockDeviceError(AiowayError, ValueError): ...


class BlockDTypeError(AiowayError, ValueError): ...
